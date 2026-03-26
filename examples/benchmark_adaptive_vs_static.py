#!/usr/bin/env python3
"""
Benchmark: Adaptive governance vs. static thresholds.

Simulates 100 actions with 2 incidents and 3 human overrides, run twice:
  1. Adaptive — Autonomica adjusts thresholds and trust based on outcomes.
  2. Static   — thresholds frozen at defaults; trust never changes.

Usage:
    python3 examples/benchmark_adaptive_vs_static.py
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from autonomica.adapter import AdaptationEngine
from autonomica.config import AutonomicaConfig
from autonomica.escalation.base import BaseEscalation
from autonomica.interceptor import Autonomica
from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceDecision,
    GovernanceMode,
    RiskScore,
)


# ---------------------------------------------------------------------------
# Silent escalation — no stdout noise, gates time-out instantly
# ---------------------------------------------------------------------------

class _SilentEscalation(BaseEscalation):
    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> None:
        pass

    async def wait_for_response(self, action_id: str, timeout: float) -> Optional[bool]:
        # None → SOFT_GATE auto-proceeds; HARD_GATE auto-blocks
        return None


# ---------------------------------------------------------------------------
# Static adapter — tracks per-tool call counts (for novelty scoring) but
# never adjusts thresholds or the trust EMA.
# ---------------------------------------------------------------------------

class _StaticAdapter(AdaptationEngine):
    def __init__(self) -> None:
        pass  # skip parent __init__; no config needed

    def update_after_action(
        self,
        action: AgentAction,
        decision: GovernanceDecision,
        profile: AgentProfile,
    ) -> None:
        # Track call counts so novelty correctly decreases over repeated calls,
        # but do NOT touch thresholds or trust score.
        count = profile.per_tool_trust.get(action.tool_name, 0)
        profile.per_tool_trust[action.tool_name] = count + 1

    def update_after_override(self, *args, **kwargs) -> None:  # type: ignore[override]
        pass  # feedback ignored

    def update_after_incident(self, *args, **kwargs) -> None:  # type: ignore[override]
        pass  # incidents ignored


# ---------------------------------------------------------------------------
# Action sequence (85 reads + 10 emails + 5 financials = 100)
# ---------------------------------------------------------------------------

# Place emails and payments at fixed indices; everything else is a read.
_EMAIL_IDX:   frozenset[int] = frozenset({10, 20, 35, 45, 49, 59, 65, 70, 75, 80})
_PAYMENT_IDX: frozenset[int] = frozenset({15, 25, 55, 73, 90})

# Inject a bad outcome here (record_outcome → tighten thresholds in adaptive)
_INCIDENT_IDX: frozenset[int] = frozenset({49, 59})

# Call record_human_override here (adaptive learns gate was too strict)
_OVERRIDE_IDX: frozenset[int] = frozenset({70, 75, 80})


def _make_action(idx: int) -> AgentAction:
    if idx in _EMAIL_IDX:
        return AgentAction(
            agent_id="bench",
            agent_name="Benchmark Agent",
            tool_name="send_email",
            tool_input={"to": "ops@corp.com", "subject": "weekly status"},
            action_type=ActionType.COMMUNICATE,
        )
    if idx in _PAYMENT_IDX:
        return AgentAction(
            agent_id="bench",
            agent_name="Benchmark Agent",
            tool_name="process_payment",
            tool_input={"amount": 50_000, "vendor": "supplier@corp.com"},
            action_type=ActionType.FINANCIAL,
        )
    return AgentAction(
        agent_id="bench",
        agent_name="Benchmark Agent",
        tool_name="db_read",
        tool_input={"query": f"SELECT id FROM logs WHERE id = {idx}"},
        action_type=ActionType.READ,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    actions: int = 0
    escalations: int = 0


@dataclass
class RunStats:
    label: str
    escalations: int = 0           # SOFT_GATE or above
    blocked: int = 0               # not approved
    incidents_caught: int = 0      # escalations in post-incident window
    false_alarms_gated: int = 0    # override targets that were actually gated
    trust_end: float = 50.0
    log_alert_max_end: float = 35.0
    phases: dict[str, PhaseStats] = field(default_factory=dict)


_PHASES: list[tuple[str, range]] = [
    ("routine   (0-49)",   range(0, 50)),
    ("incidents (50-69)",  range(50, 70)),
    ("overrides (70-84)",  range(70, 85)),
    ("recovery  (85-99)",  range(85, 100)),
]


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

async def run_simulation(adaptive: bool) -> RunStats:
    config = AutonomicaConfig(
        adaptation_rate=1.0,
        min_actions_before_adaptation=5,
        soft_gate_timeout_seconds=0.001,
        hard_gate_timeout_seconds=0.001,
        fail_policy="open",
    )
    adapter = None if adaptive else _StaticAdapter()
    gov = Autonomica(config=config, escalation=_SilentEscalation(), adapter=adapter)

    stats = RunStats(
        label="Adaptive" if adaptive else "Static  ",
        phases={name: PhaseStats() for name, _ in _PHASES},
    )

    for i in range(100):
        action = _make_action(i)
        decision = await gov.evaluate_action(action)
        escalated = decision.mode >= GovernanceMode.SOFT_GATE

        if escalated:
            stats.escalations += 1
        if not decision.approved:
            stats.blocked += 1

        for name, r in _PHASES:
            if i in r:
                stats.phases[name].actions += 1
                if escalated:
                    stats.phases[name].escalations += 1
                break

        # Inject a bad outcome at positions 49 and 59
        if i in _INCIDENT_IDX:
            gov.record_outcome(decision.action_id, success=False)
            if escalated:
                # The system caught something risky before it happened
                stats.incidents_caught += 1

        # Inject a human override at positions 70, 75, 80
        if i in _OVERRIDE_IDX:
            if escalated:
                stats.false_alarms_gated += 1
            gov.record_human_override(decision.action_id, approved=True)

    if profile := gov.get_agent_profile("bench"):
        stats.trust_end = round(profile.trust_score, 1)
        stats.log_alert_max_end = round(profile.mode_thresholds["log_alert_max"], 2)

    return stats


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def _print_results(a: RunStats, s: RunStats) -> None:
    W = 66
    fmt = "  {:<38}  {:>10}  {:>10}"

    def row(label: str, av, sv, *, prefer: str = "lower") -> None:
        a_str, s_str = str(av), str(sv)
        if isinstance(av, (int, float)) and isinstance(sv, (int, float)) and av != sv:
            winner = "a" if (av < sv) == (prefer == "lower") else "s"
            if winner == "a":
                a_str = f"{av} ✓"
            else:
                s_str = f"{sv} ✓"
        print(fmt.format(label, a_str, s_str))

    print()
    print("═" * W)
    print("  AUTONOMICA — Adaptive vs. Static Governance Benchmark")
    print("─" * W)
    print("  Scenario : 100 actions · 2 incidents · 3 human overrides")
    print("  Actions  : 85 reads · 10 emails · 5 financial ($50K each)")
    print("─" * W)
    print(fmt.format("Metric", "Adaptive", "Static"))
    print("  " + "-" * 38 + "  " + "-" * 10 + "  " + "-" * 10)
    row("Total human interruptions",       a.escalations,         s.escalations)
    row("Actions blocked",                 a.blocked,             s.blocked)
    row("Incidents caught pre-execution",  a.incidents_caught,    s.incidents_caught,
        prefer="higher")
    row("False alarms escalated for review", a.false_alarms_gated, s.false_alarms_gated,
        prefer="higher")
    row("Agent trust score (end)",         a.trust_end,           s.trust_end,
        prefer="higher")
    row("log_alert_max threshold (end)",   a.log_alert_max_end,   s.log_alert_max_end,
        prefer="higher")
    print("─" * W)

    phfmt = "  {:<26}  {:>7}  {:>10}  {:>10}"
    print(f"\n  Escalations by phase:")
    print(phfmt.format("Phase", "Actions", "Adaptive", "Static"))
    print("  " + "-" * 26 + "  " + "-" * 7 + "  " + "-" * 10 + "  " + "-" * 10)
    for name in a.phases:
        ap, sp = a.phases[name], s.phases[name]
        diff = ap.escalations - sp.escalations
        suffix = f" (+{diff})" if diff > 0 else (f" ({diff})" if diff < 0 else "")
        print(phfmt.format(name, ap.actions, f"{ap.escalations}{suffix}", sp.escalations))

    print()
    print("  What this shows:")
    diff = s.escalations - a.escalations
    if diff > 0:
        print(f"  • Adaptive reduced human interruptions by {diff} ({diff/s.escalations:.0%})")
        print(f"    — it learned the agent is reliable and widened its thresholds.")
    if a.false_alarms_gated > s.false_alarms_gated:
        extra = a.false_alarms_gated - s.false_alarms_gated
        print(f"  • Adaptive caught {extra} extra false-alarm action(s) for human review;")
        print(f"    static passed them silently — no gate, no oversight opportunity.")
    if a.trust_end > s.trust_end:
        print(f"  • Trust grew from 50.0 → {a.trust_end} in adaptive (static stays frozen at {s.trust_end}).")
    print("═" * W)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print("Running adaptive simulation ...")
    adaptive_stats = await run_simulation(adaptive=True)
    print("Running static simulation  ...")
    static_stats = await run_simulation(adaptive=False)
    _print_results(adaptive_stats, static_stats)


if __name__ == "__main__":
    asyncio.run(main())
