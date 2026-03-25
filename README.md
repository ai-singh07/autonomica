# Autonomica

**Runtime adaptive governance for AI agents.**
*Like the autonomic nervous system — your agents breathe freely when safe, tighten up when risky.*

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](#)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-lightgrey)](#)

---

## The problem

AI agents can send emails, move money, delete records, and call downstream APIs — all without asking. Today the industry treats agent governance as binary: either the agent runs free, or a human rubber-stamps every action. Neither is acceptable at scale.

## The solution

Autonomica sits between your agent and its tools. It scores every action in **< 10 ms**, maps the risk to one of **five graduated governance modes**, and — crucially — **learns over time**. An agent that consistently makes good decisions earns more autonomy. One that makes errors gets tighter guardrails automatically.

---

## 3-line integration

```python
from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools

gov = Autonomica()
tools = wrap_langchain_tools(tools, gov, agent_id="invoice-agent")
# That's it. Every tool call now flows through governance.
```

Run the quickstart demo:

```bash
python examples/quickstart.py
```

---

## Five governance modes

Autonomica never makes a binary allow/block decision. It chooses the *least restrictive* mode consistent with the risk:

| Mode | Score | Behaviour |
|------|-------|-----------|
| **FULL_AUTO** | 0–15 | Routine. Log only. Zero latency overhead. |
| **LOG_AND_ALERT** | 16–35 | Proceed + async notification to your team. |
| **SOFT_GATE** | 36–60 | Pause up to 60 s. Auto-proceed unless a human vetoes. |
| **HARD_GATE** | 61–85 | Stop. Blocked until a human explicitly approves. |
| **QUARANTINE** | 86–100 | Fully blocked. Full review required. |

Thresholds are **per-agent and adaptive** — they shift based on your team's override history.

---

## How risk is scored

Six heuristic signals, weighted composite, **no LLM calls** (< 10 ms guaranteed):

| Signal | Weight | Example |
|--------|--------|---------|
| Financial magnitude | 25% | $500K transfer → 80 |
| Data sensitivity | 20% | Health records detected → 80 |
| Reversibility | 20% | DELETE action → 80, READ → 0 |
| Agent track record | 15% | New agent (trust=50) → 50 |
| Novelty | 10% | First time using this tool → 70 |
| Cascade risk | 10% | Feeds 5 downstream agents → 75 |

---

## The autonomic nervous system analogy

Your body doesn't consciously decide to breathe or regulate blood pressure — the **autonomic nervous system** handles routine physiology automatically, escalating to conscious attention only when truly needed.

Autonomica works the same way:

- **Parasympathetic (rest & digest)** — Low-risk, familiar actions flow through without friction. Your agent barely notices governance exists.
- **Sympathetic (fight or flight)** — High-stakes or novel actions pause for human review, exactly like your body snapping to attention for a genuine threat.

The **vagal tone** metric (0–100) measures how well-calibrated governance is for each agent:

- **High vagal tone** → zero false alarms AND zero incidents. Governance is perfectly calibrated.
- **Low vagal tone (too tight)** → alert fatigue. Human operators are approving escalations too often. Thresholds widen automatically.
- **Low vagal tone (too loose)** → incidents happening. Thresholds tighten automatically.

---

## Adaptive thresholds

The key innovation: thresholds are **not static rules**. They drift based on feedback:

```
Human approves a SOFT_GATE action  → system was too strict  → soft_gate_max += 0.5
Human rejects a HARD_GATE action   → system was right       → hard_gate_max -= 1.0
Agent causes an incident           → all thresholds tighten, trust penalised
Agent runs 50 clean actions        → full_auto_max widens   (earned autonomy)
```

Trust score uses an **exponential moving average** so one bad day can't tank a reliable agent, and one good streak can't launder a bad one.

---

## Installation

```bash
# From source (until PyPI release)
git clone https://github.com/your-org/autonomica
cd autonomica
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+

**Core dependencies:** `pydantic >= 2.0`, `fastapi`, `uvicorn`, `httpx`, `langchain-core`

---

## Dashboard API

```bash
uvicorn api.main:app --reload --port 8000
# → Interactive docs at http://localhost:8000/docs
```

| Endpoint | Description |
|----------|-------------|
| `GET  /api/agents` | All agents with trust score and vagal tone |
| `GET  /api/agents/{id}` | Profile detail + adaptive thresholds |
| `GET  /api/agents/{id}/actions` | Action history for one agent |
| `GET  /api/actions` | Paginated governance decision log |
| `GET  /api/metrics/overview` | Total actions, mode distribution, escalation rate |
| `GET  /api/metrics/vagal-tone` | Calibration quality per agent |
| `GET  /api/metrics/adaptation` | Threshold drift from defaults |
| `POST /api/governance/override` | Human approve or reject a pending gate |
| `GET  /api/governance/config` | Active configuration |
| `GET  /api/audit/export?fmt=csv` | Compliance export (JSONL / JSON / CSV) |
| `GET  /health` | Health check |

---

## Escalation channels

```python
from autonomica.escalation.slack import SlackEscalation

gov = Autonomica(
    escalation=SlackEscalation("https://hooks.slack.com/services/T.../B.../...")
)
```

| Channel | Description |
|---------|-------------|
| `ConsoleEscalation` | Prints to stdout (default, zero config) |
| `SlackEscalation` | Rich colour-coded Slack messages with full risk breakdown |

Slack messages include agent name, tool, risk score breakdown, action ID, and instructions for the reviewer to approve or reject via the API.

---

## Persistent storage

```python
from autonomica import Autonomica, SQLiteStorage

gov = Autonomica(storage=SQLiteStorage("sqlite:///autonomica.db"))
# Agent profiles and decisions survive restarts.
```

---

## Configuration

```python
from autonomica import Autonomica, AutonomicaConfig

gov = Autonomica(config=AutonomicaConfig(
    soft_gate_timeout_seconds=30,          # how long soft gates wait
    hard_gate_timeout_seconds=120,         # how long hard gates wait
    default_trust_score=40.0,             # start new agents more restricted
    adaptation_rate=0.3,                  # slower threshold drift
    min_actions_before_adaptation=20,     # more data before adapting
))
```

---

## Testing

```bash
pytest                                  # 242 tests, ~0.3 s
pytest tests/test_adapter.py -v        # scenario tests (new employee, mistake, false alarm)
pytest tests/test_api.py -v            # API endpoint tests
pytest tests/test_scorer.py -v         # risk scorer unit tests
pytest tests/test_governor.py -v       # governance mode engine tests
```

---

## Project structure

```
autonomica/
├── interceptor.py        # Main Autonomica class — wraps tools
├── scorer.py             # Risk scorer — 6 signals, < 10 ms
├── governor.py           # Governance mode engine
├── adapter.py            # Adaptation engine — learns from feedback
├── config.py             # AutonomicaConfig (all tuneable parameters)
├── audit.py              # Structured JSONL audit logging
├── models.py             # Pydantic data models
├── integrations/
│   └── langchain.py      # GovernedTool, wrap_langchain_tools
├── escalation/
│   ├── console.py        # Default: print to stdout
│   └── slack.py          # Slack incoming-webhook escalation
└── storage/
    ├── base.py           # Abstract storage interface
    └── sqlite.py         # SQLite backend (asyncio.to_thread)
api/
├── main.py               # FastAPI app + CORS
├── dependencies.py       # Shared Autonomica instance (DI)
└── routes/
    ├── agents.py         # GET /api/agents, /api/agents/{id}
    ├── actions.py        # GET /api/actions, /api/actions/{id}
    ├── metrics.py        # GET /api/metrics/overview, /vagal-tone, /adaptation
    ├── governance.py     # POST /api/governance/override
    └── audit.py          # GET /api/audit/export
examples/
└── quickstart.py         # 3-line demo
```

---

## Roadmap

- [ ] PostgreSQL storage backend
- [ ] Slack interactive buttons (approve/reject without leaving Slack)
- [ ] React dashboard frontend
- [ ] CrewAI and AutoGen integrations
- [ ] ML-based risk scoring (optional upgrade path from heuristics)
- [ ] Multi-tenancy + API key auth

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built on the insight that the right governance model is not binary. It's graduated, earned, and adaptive — just like trust between humans.*
