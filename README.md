# 🧠 Autonomica — Runtime adaptive governance for AI agents

[![CI](https://github.com/ai-singh07/autonomica/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-singh07/autonomica/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-lightgrey)](#)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](#installation)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

> *Like the autonomic nervous system — your agents breathe freely when safe, tighten up when risky.*

---

## The Problem

AI agents can send emails, move money, and delete records — all without asking. The industry treats agent governance as binary: either the agent runs free, or a human rubber-stamps every action. **Neither is acceptable at scale.**

---

## Quickstart

```python
from autonomica import govern, GovernanceBlocked

@govern(agent_id="finance-bot", action_type="financial")
def process_payment(amount: float, recipient: str) -> str:
    return f"Paid ${amount} to {recipient}"

# Call normally — governance runs transparently in < 1 ms
result = process_payment(500.0, "vendor@corp.com")

# High-risk actions raise a structured exception
try:
    process_payment(1_000_000.0, "unknown@external.com")
except GovernanceBlocked as e:
    print(e.decision.risk_score.explanation)
```

```bash
pip install autonomica
python examples/real_agent_demo.py   # works without an API key
```

### LangChain integration

```python
from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools

gov = Autonomica()
tools = wrap_langchain_tools(tools, gov, agent_id="invoice-agent")
# Every tool call now flows through governance. That's it.
```

---

## Five Governance Modes

Every action lands in the *least restrictive* mode its risk warrants. Thresholds are static by default — predictable and auditable. Enable `adaptation_enabled=True` to let them drift based on real override history.

| Mode | Risk | Behaviour |
|------|:----:|-----------|
| 🟢 **FULL_AUTO** | 0–15 | Proceed silently. Log only. Zero latency overhead. |
| 🔵 **LOG_AND_ALERT** | 16–35 | Proceed immediately + async notification to your team. |
| 🟡 **SOFT_GATE** | 36–60 | Pause up to 60 s. Auto-proceeds unless a human vetoes. |
| 🔴 **HARD_GATE** | 61–85 | Full stop. Blocked until a human explicitly approves. |
| ⛔ **QUARANTINE** | 86–100 | Fully blocked. Requires audit review before any retry. |

---

## Benchmarks

Real measurements on Apple M-series, SQLite storage, Python 3.12:

| Metric | Result |
|--------|--------|
| P50 latency (sequential) | 0.057 ms |
| P99 latency (sequential) | **0.124 ms** |
| P99 latency (1 000 concurrent) | 0.123 ms |
| Throughput (sequential) | ~17 000 actions/sec |
| Human interruptions — static governance | 5 per 100 actions |
| Human interruptions — adaptive governance | **1 per 100 actions (80% fewer)** |

Run it yourself: `python examples/load_test.py` · `python examples/benchmark_adaptive_vs_static.py`

---

## Why Bio-Inspired?

The human brain runs two systems in parallel. **System 1** handles 99% of decisions instantly — breathing, walking, reading familiar text. Interrupting it for every action would cause paralysis. **System 2** kicks in only for genuinely high-stakes moments. Autonomica works the same way: routine agent actions flow through in < 1 ms with zero human friction; high-risk actions pause for review. The **vagal tone** metric tells you how well-calibrated this balance is — too tight means alert fatigue, too loose means incidents.

---

## Installation

```bash
pip install autonomica                  # PyPI — coming soon
# or from source:
git clone https://github.com/ai-singh07/autonomica
cd autonomica && pip install -e ".[dev]"
```

**Requirements:** Python 3.11+
**Core deps:** `pydantic >= 2.0` · `fastapi` · `uvicorn` · `httpx` · `langchain-core`

---

## Configuration

### Static mode (default)

Predictable, auditable, enterprise-safe. Thresholds never change without explicit config updates.

```python
from autonomica import Autonomica, AutonomicaConfig, SQLiteStorage
from autonomica.escalation.slack import SlackEscalation

gov = Autonomica(
    config=AutonomicaConfig(
        soft_gate_timeout_seconds=30,
        hard_gate_timeout_seconds=120,
        fail_policy="open",          # "open" | "closed" | "adaptive"
        tool_overrides={
            "process_payment": {     # always high-stakes, regardless of amount
                "financial_magnitude": 90,
                "reversibility": 80,
            },
            "write_tutorial": {      # zero financial/PII risk by design
                "data_sensitivity": 0,
                "financial_magnitude": 0,
            },
        },
    ),
    storage=SQLiteStorage("sqlite:///autonomica.db"),
    escalation=SlackEscalation("https://hooks.slack.com/services/YOUR/WEBHOOK/URL"),
)
```

### Adaptive mode

Agents earn trust over time. Thresholds tighten after incidents, widen after false alarms. Recommended after your deployment has a baseline of human override history.

```python
gov = Autonomica(
    config=AutonomicaConfig(
        adaptation_enabled=True,           # off by default
        adaptation_rate=0.3,
        min_actions_before_adaptation=20,
        default_trust_score=40.0,
    ),
)
```

Valid override signal names: `financial_magnitude` · `data_sensitivity` · `reversibility` · `agent_track_record` · `novelty` · `cascade_risk`. Values must be in **[0, 100]**.

---

## Override API

```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET  /api/agents` | All agents with trust score and vagal tone |
| `GET  /api/agents/{id}` | Profile + adaptive threshold detail |
| `GET  /api/metrics/overview` | Mode distribution, escalation rate, avg score |
| `POST /api/governance/override` | Approve or reject a pending gate |
| `GET  /api/audit/export?fmt=csv` | Compliance export (JSONL / JSON / CSV) |

---

## Examples

| File | What it shows |
|------|---------------|
| [`examples/quickstart.py`](examples/quickstart.py) | LangChain agent + `wrap_langchain_tools` |
| [`examples/real_agent_demo.py`](examples/real_agent_demo.py) | `@govern` decorator, 4 tool types, LLM fallback |
| [`examples/benchmark_adaptive_vs_static.py`](examples/benchmark_adaptive_vs_static.py) | Adaptive vs static human interruption comparison |
| [`examples/load_test.py`](examples/load_test.py) | P50/P95/P99 latency at 1 000 concurrent calls |

---

## Architecture

```
┌──────────────────────────────────────────┐
│              Your AI Agent               │
└──────────────────┬───────────────────────┘
                   │  every tool call
                   ▼
┌──────────────────────────────────────────┐
│               Autonomica                 │
│                                          │
│  ┌────────────┐  ┌──────────┐  ┌──────┐  │
│  │ Risk Scorer│─▶│ Governor │─▶│Adapt │  │
│  │ 6 signals  │  │ 5 modes  │  │ EMA  │  │
│  │ < 1 ms     │  │ enforce  │  │trust │  │
│  └────────────┘  └──────────┘  └──────┘  │
│                       │                  │
│               ┌───────┴──────┐           │
│               │  Escalation  │ Slack/CLI  │
│               └──────────────┘           │
└──────────────────┬───────────────────────┘
                   │  approved / blocked
                   ▼
┌──────────────────────────────────────────┐
│              Your Tools                  │
└──────────────────────────────────────────┘
```

---

## Roadmap

- [x] 6-signal heuristic risk scorer (< 1 ms, zero LLM calls)
- [x] 5-mode governance engine with adaptive thresholds
- [x] EMA trust score + vagal tone calibration
- [x] `@govern` universal decorator (sync + async)
- [x] LangChain integration (`wrap_langchain_tools`)
- [x] Slack escalation with colour-coded risk breakdowns
- [x] SQLite persistence + async audit log
- [x] FastAPI dashboard + override API
- [x] Per-tool risk overrides + argument-aware SQL scoring
- [x] Fail policy (open / closed / adaptive)
- [ ] PostgreSQL storage backend
- [ ] CrewAI integration
- [ ] AutoGen integration
- [ ] Slack interactive approve/reject buttons
- [ ] React dashboard frontend
- [ ] OpenTelemetry tracing support
- [ ] Interactive governance demo notebook
- [ ] ML-based risk scoring (optional upgrade path)
- [ ] Multi-tenancy + API key authentication

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are labelled [`good first issue`](https://github.com/ai-singh07/autonomica/issues?q=label%3A%22good+first+issue%22) on GitHub.

```bash
git clone https://github.com/ai-singh07/autonomica
cd autonomica && pip install -e ".[dev]"
pytest                             # 436 tests, < 2 s
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*The right governance model is not binary. It's graduated, earned, and adaptive — just like trust between humans.*
