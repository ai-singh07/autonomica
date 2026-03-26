# 🧠 Autonomica — Runtime adaptive governance for AI agents

[![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-lightgrey)](#)

> *Like the autonomic nervous system — your agents breathe freely when safe, tighten up when risky.*

---

## The Problem

AI agents can send emails, move money, delete records, and call downstream APIs — all without asking. The industry treats agent governance as binary: either the agent runs completely free, or a human rubber-stamps every single action. **Neither is acceptable at scale.**

---

## Quickstart

```python
from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools

gov = Autonomica()
tools = wrap_langchain_tools(tools, gov, agent_id="invoice-agent")
# Every tool call now flows through governance. That's it.
```

```bash
git clone https://github.com/hsbhatia1993-blip/autonomica
cd autonomica && pip install -e ".[dev]"
python examples/quickstart.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your AI Agent                        │
└──────────────────────────┬──────────────────────────────────┘
                           │  every tool call
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        Autonomica                           │
│                                                             │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   │
│   │ Risk Scorer │──▶│  Governor    │──▶│   Adapter     │   │
│   │ 6 signals   │   │ 5 modes      │   │ EMA trust     │   │
│   │ < 10 ms     │   │ enforce/gate │   │ threshold     │   │
│   └─────────────┘   └──────────────┘   │ drift         │   │
│                             │           └───────────────┘   │
│                    ┌────────┴────────┐                      │
│                    │  Escalation     │  Slack / Console      │
│                    └─────────────────┘                      │
└──────────────────────────┬──────────────────────────────────┘
                           │  approved / blocked
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Your Tools                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Five Governance Modes

Autonomica never makes a binary allow/block decision. Every action lands in the *least restrictive* mode the risk warrants — and thresholds shift automatically over time based on your team's feedback.

| Mode | Risk Score | Behaviour |
|------|:----------:|-----------|
| 🟢 **FULL_AUTO** | 0 – 15 | Proceed silently. Log only. Zero latency overhead. |
| 🔵 **LOG_AND_ALERT** | 16 – 35 | Proceed immediately + async notification to your team. |
| 🟡 **SOFT_GATE** | 36 – 60 | Pause up to 60 s. Auto-proceeds unless a human vetoes. |
| 🔴 **HARD_GATE** | 61 – 85 | Full stop. Blocked until a human explicitly approves. |
| ⛔ **QUARANTINE** | 86 – 100 | Fully blocked. Requires audit review before any retry. |

Thresholds are **per-agent and adaptive** — they drift based on your team's real override history, not static config.

---

## Why Bio-Inspired?

The human brain runs two systems in parallel:

**System 1** (fast, automatic) handles 99% of decisions instantly — breathing, walking, reading familiar words. Interrupting it for every action would be paralysis.

**System 2** (slow, deliberate) kicks in only for genuinely novel or high-stakes situations — a large financial decision, an unfamiliar context, a detected anomaly.

Autonomica works the same way. Routine agent actions flow through in < 10 ms with zero human friction (System 1). High-risk or novel actions pause for human review (System 2). The **vagal tone** metric tells you how well-calibrated this balance is for each agent:

| Vagal Tone | Meaning | Action |
|:----------:|---------|--------|
| **High (80–100)** | Governance is perfectly calibrated. | No action needed. |
| **Low — too tight** | Alert fatigue. Humans approving too much. | Thresholds auto-widen. |
| **Low — too loose** | Incidents happening. Agent not ready. | Thresholds auto-tighten. |

Trust uses an **exponential moving average** — one bad day can't tank a reliable agent, and one good streak can't launder a bad one.

---

## Installation

```bash
# From source (PyPI release coming soon)
git clone https://github.com/hsbhatia1993-blip/autonomica
cd autonomica
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+

**Core dependencies:** `pydantic >= 2.0` · `fastapi` · `uvicorn` · `httpx` · `langchain-core`

---

## Configuration

```python
from autonomica import Autonomica, AutonomicaConfig, SQLiteStorage
from autonomica.escalation.slack import SlackEscalation

gov = Autonomica(
    config=AutonomicaConfig(
        soft_gate_timeout_seconds=30,      # how long soft gates wait for a human
        hard_gate_timeout_seconds=120,     # how long hard gates wait before blocking
        default_trust_score=40.0,          # start new agents more conservatively
        adaptation_rate=0.3,               # how fast thresholds drift (0.1–1.0)
        min_actions_before_adaptation=20,  # minimum data points before adapting
    ),
    storage=SQLiteStorage("sqlite:///autonomica.db"),
    escalation=SlackEscalation("https://hooks.slack.com/services/YOUR/WEBHOOK/URL"),
)
```

### Override API

```bash
# Start the dashboard API
uvicorn api.main:app --reload --port 8000
# Docs → http://localhost:8000/docs
```

```bash
# Human approves a pending HARD_GATE action
curl -X POST http://localhost:8000/api/governance/override \
  -H "Content-Type: application/json" \
  -d '{"action_id": "abc-123", "approved": true, "reason": "Verified with finance team"}'
```

| Endpoint | Description |
|----------|-------------|
| `GET  /api/agents` | All agents with trust score and vagal tone |
| `GET  /api/agents/{id}` | Profile + adaptive threshold detail |
| `GET  /api/metrics/overview` | Mode distribution, escalation rate, avg score |
| `GET  /api/metrics/vagal-tone` | Calibration quality per agent |
| `POST /api/governance/override` | Approve or reject a pending gate |
| `GET  /api/audit/export?fmt=csv` | Compliance export (JSONL / JSON / CSV) |

---

## Roadmap

- [x] 6-signal heuristic risk scorer (< 10 ms, no LLM calls)
- [x] 5-mode governance engine with adaptive thresholds
- [x] EMA trust score + vagal tone calibration metric
- [x] LangChain integration (`wrap_langchain_tools` — 1 line)
- [x] Slack escalation with colour-coded risk breakdowns
- [x] SQLite persistence + async audit log
- [x] FastAPI dashboard with override API
- [ ] PostgreSQL storage backend
- [ ] Slack interactive buttons (approve/reject without leaving Slack)
- [ ] React dashboard frontend
- [ ] CrewAI and AutoGen integrations
- [ ] ML-based risk scoring (optional upgrade path from heuristics)
- [ ] Multi-tenancy and API key authentication

---

## Contributing

We'd love your help. Open an issue to discuss what you'd like to change, then submit a PR against `main`. Please include tests — the suite runs in < 1 s.

```bash
pytest                              # run all 287 tests
pytest tests/test_adapter.py -v    # scenario tests
pytest tests/test_api.py -v        # API endpoint tests
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*The right governance model is not binary. It's graduated, earned, and adaptive — just like trust between humans.*
