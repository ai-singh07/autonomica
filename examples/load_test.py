#!/usr/bin/env python3
"""
Autonomica — governance latency load test.

Test 1 (Sequential): 1,000 evaluate_action() calls one at a time.
Test 2 (Concurrent): 1,000 calls dispatched as asyncio.gather batches of 10.

Both tests use SQLite storage (realistic I/O conditions) and a realistic
mix of action types: 70% reads, 20% writes, 10% financial.

Usage:
    python3 examples/load_test.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import tempfile
import time
from typing import Sequence

from autonomica.config import AutonomicaConfig
from autonomica.escalation.base import BaseEscalation
from autonomica.interceptor import Autonomica
from autonomica.models import ActionType, AgentAction, GovernanceMode, RiskScore
from autonomica.storage.sqlite import SQLiteStorage


# ---------------------------------------------------------------------------
# Silent escalation — suppresses all console output during the load test
# ---------------------------------------------------------------------------

class _SilentEscalation(BaseEscalation):
    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> None:
        pass

    async def wait_for_response(self, action_id: str, timeout: float):
        return None  # SOFT_GATE → auto-proceed; HARD_GATE → auto-block

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ACTIONS = 1_000
BATCH_SIZE = 10          # concurrent goroutines per gather wave
N_AGENTS   = 10          # distinct agent IDs used in concurrent test

# Action-type distribution (must sum to 10 for modulo indexing)
#   0-6 → READ (70%), 7-8 → WRITE (20%), 9 → FINANCIAL (10%)
_TYPE_MAP: dict[int, tuple[ActionType, str, dict]] = {
    **{i: (ActionType.READ, "db_read",
           {"query": f"SELECT id FROM logs WHERE bucket = {i}"})
       for i in range(7)},
    7: (ActionType.WRITE, "write_record",
        {"table": "events", "payload": "status=ok"}),
    8: (ActionType.WRITE, "update_config",
        {"key": "feature_flag", "value": "true"}),
    9: (ActionType.FINANCIAL, "process_payment",
        {"amount": 1_500, "vendor": "supplier@corp.com"}),
}


def _make_action(idx: int, agent_id: str = "load-test-agent") -> AgentAction:
    action_type, tool_name, tool_input = _TYPE_MAP[idx % 10]
    return AgentAction(
        agent_id=agent_id,
        agent_name="Load Test Agent",
        tool_name=tool_name,
        tool_input={**tool_input, "_idx": idx},   # unique input per call
        action_type=action_type,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _stats(latencies: list[float], wall_s: float) -> dict:
    return {
        "n":          len(latencies),
        "min":        min(latencies),
        "p50":        _percentile(latencies, 50),
        "p95":        _percentile(latencies, 95),
        "p99":        _percentile(latencies, 99),
        "max":        max(latencies),
        "mean":       statistics.mean(latencies),
        "throughput": len(latencies) / wall_s,
    }


# ---------------------------------------------------------------------------
# Test 1 — Sequential
# ---------------------------------------------------------------------------

async def test_sequential(gov: Autonomica) -> dict:
    latencies: list[float] = []

    wall_start = time.perf_counter()
    for i in range(N_ACTIONS):
        action = _make_action(i)
        t0 = time.perf_counter()
        await gov.evaluate_action(action)
        latencies.append((time.perf_counter() - t0) * 1_000)
    wall_s = time.perf_counter() - wall_start

    return _stats(latencies, wall_s)


# ---------------------------------------------------------------------------
# Test 2 — Concurrent (batches of BATCH_SIZE, N_AGENTS distinct agent IDs)
# ---------------------------------------------------------------------------

async def test_concurrent(gov: Autonomica) -> dict:
    latencies: list[float] = []

    async def _timed(action: AgentAction) -> float:
        t0 = time.perf_counter()
        await gov.evaluate_action(action)
        return (time.perf_counter() - t0) * 1_000

    wall_start = time.perf_counter()
    for batch_start in range(0, N_ACTIONS, BATCH_SIZE):
        batch = [
            _make_action(i, agent_id=f"agent-{i % N_AGENTS}")
            for i in range(batch_start, batch_start + BATCH_SIZE)
        ]
        results = await asyncio.gather(*[_timed(a) for a in batch])
        latencies.extend(results)
    wall_s = time.perf_counter() - wall_start

    return _stats(latencies, wall_s)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(label: str, s: dict, action_mix: str) -> None:
    W = 62
    claim_ok = s["p99"] < 10.0

    print()
    print("═" * W)
    print(f"  {label}")
    print(f"  {action_mix}")
    print("─" * W)
    print(f"  {'Metric':<28}  {'Value':>14}  {'Target':>12}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*12}")

    def row(name: str, val: float, unit: str = "ms",
            target: str = "", ok: bool | None = None) -> None:
        val_str = f"{val:>10.3f} {unit}"
        tgt_str = target if target else ""
        flag = ""
        if ok is True:
            flag = " ✓"
        elif ok is False:
            flag = " ✗"
        print(f"  {name:<28}  {val_str:>14}  {tgt_str:>10}{flag}")

    row("Min latency",  s["min"],  target="—")
    row("P50 latency",  s["p50"],  target="< 10 ms", ok=s["p50"] < 10.0)
    row("P95 latency",  s["p95"],  target="< 10 ms", ok=s["p95"] < 10.0)
    row("P99 latency",  s["p99"],  target="< 10 ms", ok=s["p99"] < 10.0)
    row("Max latency",  s["max"],  target="—")
    row("Mean latency", s["mean"], target="—")
    row("Throughput",   s["throughput"], unit="act/s", target="—")
    print("─" * W)
    verdict = "PASS — <10ms claim verified ✓" if claim_ok else "FAIL — P99 exceeded 10ms ✗"
    print(f"  {verdict}")
    print("═" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    db_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="autonomica_load_")
    os.close(db_fd)

    try:
        storage = SQLiteStorage(db_path)
        await storage.initialize()

        config = AutonomicaConfig(
            soft_gate_timeout_seconds=0.001,   # don't block on gates
            hard_gate_timeout_seconds=0.001,
            fail_policy="open",
            database_url=f"sqlite:///{db_path}",
        )
        escalation = _SilentEscalation()
        gov = Autonomica(config=config, storage=storage, escalation=escalation)

        action_mix = "Mix: 70% reads · 20% writes · 10% financial  |  SQLite storage"

        print(f"\nRunning Test 1 — Sequential ({N_ACTIONS:,} calls) ...")
        seq = await test_sequential(gov)
        _print_table("Test 1 — Sequential", seq, action_mix)

        # Fresh instance so profiles/caches don't carry over
        gov2 = Autonomica(config=config, storage=storage, escalation=escalation)

        print(f"\nRunning Test 2 — Concurrent "
              f"({N_ACTIONS:,} calls · {BATCH_SIZE} at a time · {N_AGENTS} agents) ...")
        con = await test_concurrent(gov2)
        _print_table(
            f"Test 2 — Concurrent (batch={BATCH_SIZE}, agents={N_AGENTS})",
            con, action_mix,
        )

        print()
        print(f"  Sequential P99 : {seq['p99']:.3f} ms  |  "
              f"Concurrent P99 : {con['p99']:.3f} ms")
        print(f"  Sequential tput: {seq['throughput']:.0f} act/s  |  "
              f"Concurrent tput: {con['throughput']:.0f} act/s")
        print()

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
