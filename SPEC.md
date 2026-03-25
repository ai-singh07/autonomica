# Autonomica — Technical Specification v1.0
## Runtime Adaptive Governance for AI Agents

> Hand this document to Claude Code as context. It contains everything needed to build the MVP.

---

## 1. WHAT IS AUTONOMICA?

Autonomica is a runtime governance layer that controls how much autonomy AI agents get — adapting in real-time like the human autonomic nervous system.

It sits as a proxy/middleware between an agent framework (LangChain, CrewAI, etc.) and the tools/APIs agents interact with. It intercepts every agent action, scores its risk in real-time, and applies one of five graduated governance modes — from full autonomy to full quarantine.

**The key innovation:** Thresholds between governance modes are NOT static rules. They self-tune over time based on human feedback (overrides, approvals, incidents). An agent that consistently makes good decisions earns more autonomy. One that makes errors gets tighter guardrails.

---

## 2. SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────┐
│                    AGENT FRAMEWORK                   │
│              (LangChain / CrewAI / etc.)             │
└──────────────────────┬──────────────────────────────┘
                       │ Agent wants to call a tool
                       ▼
┌─────────────────────────────────────────────────────┐
│              AUTONOMICA SDK (Python)                  │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Action     │  │    Risk      │  │  Governance  │ │
│  │ Interceptor  │──│   Scorer     │──│    Mode      │ │
│  │              │  │  (Amygdala)  │  │   Engine     │ │
│  └─────────────┘  └──────────────┘  └──────┬──────┘ │
│                                             │        │
│  ┌─────────────┐  ┌──────────────┐         │        │
│  │ Adaptation   │  │  Audit Log   │◄────────┘        │
│  │   Engine     │  │  (Postgres)  │                  │
│  └─────────────┘  └──────────────┘                   │
└──────────────────────┬──────────────────────────────┘
                       │ Approved action passes through
                       ▼
┌─────────────────────────────────────────────────────┐
│                 EXTERNAL TOOLS/APIs                   │
│          (Databases, Email, APIs, etc.)               │
└─────────────────────────────────────────────────────┘
```

---

## 3. PROJECT STRUCTURE

```
autonomica/
├── README.md
├── pyproject.toml                 # Package config (use Poetry or hatch)
├── LICENSE                        # Apache 2.0
│
├── autonomica/                    # Main package
│   ├── __init__.py               # Public API exports
│   ├── interceptor.py            # Action Interceptor - wraps tool calls
│   ├── scorer.py                 # Risk Scorer - evaluates each action
│   ├── governor.py               # Governance Mode Engine - decides mode
│   ├── adapter.py                # Adaptation Engine - learns from feedback
│   ├── models.py                 # Pydantic models for all data structures
│   ├── config.py                 # Configuration management
│   ├── audit.py                  # Audit logging
│   │
│   ├── integrations/             # Framework-specific integrations
│   │   ├── __init__.py
│   │   ├── langchain.py          # LangChain tool wrapper
│   │   └── base.py               # Base integration interface
│   │
│   ├── escalation/               # Escalation channels
│   │   ├── __init__.py
│   │   ├── slack.py              # Slack webhook notifications
│   │   ├── console.py            # Console/terminal notifications
│   │   └── base.py               # Base escalation interface
│   │
│   └── storage/                  # Storage backends
│       ├── __init__.py
│       ├── sqlite.py             # SQLite for local dev
│       ├── postgres.py           # Postgres for production
│       └── base.py               # Base storage interface
│
├── api/                           # FastAPI dashboard backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── routes/
│   │   ├── agents.py             # Agent CRUD + status
│   │   ├── actions.py            # Action log + search
│   │   ├── governance.py         # Governance config
│   │   └── metrics.py            # Metrics + vagal tone
│   └── dependencies.py
│
├── dashboard/                     # React frontend (optional in MVP)
│   └── ...
│
├── tests/
│   ├── test_interceptor.py
│   ├── test_scorer.py
│   ├── test_governor.py
│   ├── test_adapter.py
│   ├── test_integration_langchain.py
│   └── conftest.py               # Shared fixtures
│
├── examples/
│   ├── quickstart.py             # 10-line getting started
│   ├── langchain_agent.py        # Full LangChain example
│   └── custom_scorer.py          # Custom risk scoring rules
│
└── docs/
    ├── getting-started.md
    ├── architecture.md
    └── configuration.md
```

---

## 4. CORE DATA MODELS

```python
# autonomica/models.py

from pydantic import BaseModel, Field
from enum import IntEnum
from datetime import datetime
from typing import Optional, Any

class GovernanceMode(IntEnum):
    """Five graduated governance modes - like the autonomic nervous system."""
    FULL_AUTO = 0          # Score 0-15: Routine. Log only.
    LOG_AND_ALERT = 1      # Score 16-35: Proceed + async notification.
    SOFT_GATE = 2          # Score 36-60: Pause 30-60s. Auto-proceed unless human intervenes.
    HARD_GATE = 3          # Score 61-85: Stop. Human must explicitly approve.
    QUARANTINE = 4         # Score 86-100: Blocked. Full review required.

class ActionType(str, Enum):
    """Categories of agent actions by reversibility."""
    READ = "read"              # Fully reversible (database reads, API gets)
    WRITE = "write"            # Partially reversible (database writes, file creation)
    COMMUNICATE = "communicate" # Irreversible (emails, messages, API posts)
    DELETE = "delete"          # Catastrophic (database deletes, file deletion)
    FINANCIAL = "financial"    # High-stakes (transactions, payments, contracts)

class AgentAction(BaseModel):
    """Represents a single action an agent wants to take."""
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str                    # Unique identifier for the agent
    agent_name: str                  # Human-readable agent name
    tool_name: str                   # Name of the tool being called
    tool_input: dict[str, Any]       # Arguments to the tool
    action_type: ActionType          # Category of action
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = {}    # Additional context

class RiskScore(BaseModel):
    """Output of the risk scorer."""
    composite_score: float           # 0-100 final score
    financial_magnitude: float       # 0-100
    data_sensitivity: float          # 0-100
    reversibility: float             # 0-100 (higher = less reversible = more risky)
    agent_track_record: float        # 0-100 (higher = worse track record = more risky)
    novelty: float                   # 0-100 (higher = more novel = more risky)
    cascade_risk: float              # 0-100
    explanation: str                 # Human-readable explanation of the score

class GovernanceDecision(BaseModel):
    """The governance engine's decision for an action."""
    action_id: str
    risk_score: RiskScore
    mode: GovernanceMode
    approved: bool                   # Whether the action was ultimately approved
    escalated_to: Optional[str]      # Channel/person it was escalated to
    human_override: Optional[bool]   # Did a human override the decision?
    decision_time_ms: float          # How long the decision took
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentProfile(BaseModel):
    """Per-agent autonomy profile — the learned trust state."""
    agent_id: str
    agent_name: str
    total_actions: int = 0
    approved_actions: int = 0
    escalated_actions: int = 0
    incidents: int = 0              # Actions that caused problems
    false_escalations: int = 0      # Escalations where human approved anyway
    trust_score: float = 50.0       # 0-100, starts at 50 (neutral)
    vagal_tone: float = 50.0        # System health metric
    mode_thresholds: dict[str, float] = {
        "full_auto_max": 15.0,
        "log_alert_max": 35.0,
        "soft_gate_max": 60.0,
        "hard_gate_max": 85.0,
        # Above 85 = quarantine
    }
    per_tool_trust: dict[str, float] = {}  # Per-tool trust scores
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## 5. CORE COMPONENTS — IMPLEMENTATION DETAILS

### 5.1 Action Interceptor (`interceptor.py`)

The interceptor wraps agent tool calls. For LangChain, this means wrapping `BaseTool.run()` and `BaseTool.arun()`.

```python
# Key interface:
class Autonomica:
    """Main entry point. Wraps an agent framework with governance."""

    def __init__(
        self,
        config: AutonomicaConfig = None,
        storage: BaseStorage = None,
        escalation: BaseEscalation = None,
    ):
        self.scorer = RiskScorer(config)
        self.governor = GovernanceEngine(config)
        self.adapter = AdaptationEngine(config, storage)
        self.storage = storage or SQLiteStorage()
        self.escalation = escalation or ConsoleEscalation()

    def wrap_tool(self, tool, agent_id: str, agent_name: str = None) -> WrappedTool:
        """Wrap a single tool with governance."""
        # Returns a new tool that intercepts calls through the governance pipeline

    def wrap_tools(self, tools: list, agent_id: str, agent_name: str = None) -> list:
        """Wrap multiple tools at once."""

    def wrap_langchain_agent(self, agent, agent_id: str) -> agent:
        """Convenience: wrap all tools in a LangChain agent."""

    async def evaluate_action(self, action: AgentAction) -> GovernanceDecision:
        """Core pipeline: score → govern → log → adapt."""

    def record_outcome(self, action_id: str, success: bool, notes: str = ""):
        """Record whether an approved action succeeded or failed."""

    def record_human_override(self, action_id: str, approved: bool, reason: str = ""):
        """Record a human's override decision for an escalated action."""
```

**Usage should be dead simple — 3 lines:**
```python
from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools

gov = Autonomica()
tools = wrap_langchain_tools(tools, gov, agent_id="invoice-agent")
# That's it. All tool calls now go through governance.
```

### 5.2 Risk Scorer (`scorer.py`)

MVP uses heuristic scoring (no ML). Each signal produces a 0-100 sub-score. The composite is a weighted average.

```python
class RiskScorer:
    """Evaluates the risk of an agent action. The 'amygdala' of the system."""

    DEFAULT_WEIGHTS = {
        "financial_magnitude": 0.25,
        "data_sensitivity": 0.20,
        "reversibility": 0.20,
        "agent_track_record": 0.15,
        "novelty": 0.10,
        "cascade_risk": 0.10,
    }

    def score(self, action: AgentAction, agent_profile: AgentProfile) -> RiskScore:
        """Score an action. Must complete in <10ms."""
        ...
```

**Scoring heuristics for MVP:**

| Signal | How to score (MVP) |
|--------|-------------------|
| Financial magnitude | Parse tool_input for amount/value fields. $0-100 = 0, $100-10K = 20, $10K-100K = 50, $100K+ = 80, $1M+ = 100 |
| Data sensitivity | Keyword scan of tool_input for PII patterns (email, SSN, phone, credit card). If PII detected → 60. If financial data → 70. If health data → 80. Configurable via data_sensitivity_rules in config. |
| Reversibility | Map ActionType: READ=0, WRITE=30, COMMUNICATE=60, DELETE=80, FINANCIAL=70 |
| Agent track record | Derived from AgentProfile: `100 - agent_profile.trust_score`. A trusted agent (trust=90) gets a low risk score (10) on this signal. |
| Novelty | Check if this agent has called this tool before. First time = 70. Called < 10 times = 40. Called 10+ times = 10. Check per_tool_trust in profile. |
| Cascade risk | MVP: static config per agent. Default = 20. Configurable: if this agent feeds data to N other agents, cascade_risk = min(N * 15, 100). |

### 5.3 Governance Mode Engine (`governor.py`)

```python
class GovernanceEngine:
    """Maps risk scores to governance modes. The 'autonomic switch'."""

    def decide(self, risk_score: RiskScore, agent_profile: AgentProfile) -> GovernanceMode:
        """
        Map composite score to a mode using the agent's personal thresholds.
        These thresholds are different per agent and adapt over time.
        """
        score = risk_score.composite_score
        thresholds = agent_profile.mode_thresholds

        if score <= thresholds["full_auto_max"]:
            return GovernanceMode.FULL_AUTO
        elif score <= thresholds["log_alert_max"]:
            return GovernanceMode.LOG_AND_ALERT
        elif score <= thresholds["soft_gate_max"]:
            return GovernanceMode.SOFT_GATE
        elif score <= thresholds["hard_gate_max"]:
            return GovernanceMode.HARD_GATE
        else:
            return GovernanceMode.QUARANTINE

    async def enforce(self, mode: GovernanceMode, action: AgentAction, escalation: BaseEscalation) -> bool:
        """
        Enforce the governance mode. Returns True if action should proceed.

        - FULL_AUTO: return True immediately, log async
        - LOG_AND_ALERT: return True, send async notification
        - SOFT_GATE: wait for `soft_gate_timeout` seconds. Proceed unless human rejects.
        - HARD_GATE: wait for human approval. Timeout = reject.
        - QUARANTINE: return False immediately. Log incident.
        """
        ...
```

### 5.4 Adaptation Engine (`adapter.py`)

```python
class AdaptationEngine:
    """Learns from governance decisions over time. The 'vagal tone' calibrator."""

    def update_after_decision(self, decision: GovernanceDecision, agent_profile: AgentProfile):
        """Called after every governance decision to update the agent's profile."""

        # Update total counts
        agent_profile.total_actions += 1

        if decision.mode == GovernanceMode.FULL_AUTO:
            agent_profile.approved_actions += 1

        if decision.human_override is True:
            # Human approved something we escalated → we were too strict
            agent_profile.false_escalations += 1
            self._widen_thresholds(agent_profile, decision, amount=0.5)

        if decision.human_override is False:
            # Human rejected something → we were right to escalate (or too lenient)
            self._tighten_thresholds(agent_profile, decision, amount=1.0)

        # Update trust score (exponential moving average)
        # Good outcomes increase trust, bad outcomes decrease it
        # ...

        # Update vagal tone (measure of calibration quality)
        # High vagal tone = low false escalation rate AND low incident rate
        # ...

    def _widen_thresholds(self, profile, decision, amount):
        """Increase autonomy: move threshold boundaries up."""
        # The mode that was triggered gets its upper bound increased
        # This means more actions fall into lower (more autonomous) modes
        ...

    def _tighten_thresholds(self, profile, decision, amount):
        """Decrease autonomy: move threshold boundaries down."""
        # The mode that was triggered gets its upper bound decreased
        # This means more actions fall into higher (more restrictive) modes
        ...

    def calculate_vagal_tone(self, profile: AgentProfile) -> float:
        """
        Vagal tone = how well-calibrated is this agent's governance?

        Perfect vagal tone (100): zero incidents AND zero false escalations.
        Low vagal tone: either too many incidents (too loose) or too many
        unnecessary escalations (too tight).

        Formula: 100 - (incident_rate * 60) - (false_escalation_rate * 40)
        """
        ...
```

---

## 6. CONFIGURATION

```python
# autonomica/config.py

class AutonomicaConfig(BaseModel):
    """Global configuration for Autonomica."""

    # Governance
    soft_gate_timeout_seconds: int = 60      # How long soft gate waits
    hard_gate_timeout_seconds: int = 300     # How long hard gate waits before rejecting
    default_trust_score: float = 50.0        # New agents start here (0-100)

    # Scoring weights (must sum to 1.0)
    scoring_weights: dict[str, float] = RiskScorer.DEFAULT_WEIGHTS

    # Financial thresholds (currency-agnostic amounts)
    financial_thresholds: dict[str, float] = {
        "low": 100,
        "medium": 10_000,
        "high": 100_000,
        "critical": 1_000_000,
    }

    # Data sensitivity keywords
    pii_patterns: list[str] = ["email", "ssn", "phone", "credit_card", "address", "dob"]
    financial_data_patterns: list[str] = ["account_number", "routing", "balance", "salary"]
    health_data_patterns: list[str] = ["diagnosis", "medication", "patient", "medical"]

    # Adaptation
    adaptation_rate: float = 0.5             # How fast thresholds adapt (0=never, 1=instant)
    min_actions_before_adaptation: int = 10  # Don't adapt until this many actions

    # Storage
    storage_backend: str = "sqlite"          # "sqlite" or "postgres"
    database_url: str = "sqlite:///autonomica.db"

    # Escalation
    escalation_backend: str = "console"      # "console", "slack", "webhook"
    slack_webhook_url: Optional[str] = None
```

---

## 7. LANGCHAIN INTEGRATION

```python
# autonomica/integrations/langchain.py

from langchain.tools import BaseTool
from autonomica import Autonomica
from autonomica.models import AgentAction, ActionType

class GovernedTool(BaseTool):
    """A LangChain tool wrapped with Autonomica governance."""

    original_tool: BaseTool
    autonomica: Autonomica
    agent_id: str

    def _run(self, *args, **kwargs):
        action = AgentAction(
            agent_id=self.agent_id,
            agent_name=self.agent_id,
            tool_name=self.original_tool.name,
            tool_input=kwargs or {"input": args[0] if args else ""},
            action_type=self._infer_action_type(),
        )

        decision = self.autonomica.evaluate_action_sync(action)

        if not decision.approved:
            return f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). Risk score: {decision.risk_score.composite_score:.1f}. Reason: {decision.risk_score.explanation}"

        result = self.original_tool._run(*args, **kwargs)
        self.autonomica.record_outcome(action.action_id, success=True)
        return result

    async def _arun(self, *args, **kwargs):
        # Async version of the same
        ...

    def _infer_action_type(self) -> ActionType:
        """Infer action type from tool name/description."""
        name = self.original_tool.name.lower()
        desc = (self.original_tool.description or "").lower()

        if any(w in name or w in desc for w in ["delete", "remove", "drop"]):
            return ActionType.DELETE
        if any(w in name or w in desc for w in ["send", "email", "message", "post", "notify"]):
            return ActionType.COMMUNICATE
        if any(w in name or w in desc for w in ["pay", "transfer", "invoice", "charge"]):
            return ActionType.FINANCIAL
        if any(w in name or w in desc for w in ["write", "update", "create", "insert", "set"]):
            return ActionType.WRITE
        return ActionType.READ


def wrap_langchain_tools(tools: list[BaseTool], autonomica: Autonomica, agent_id: str) -> list[GovernedTool]:
    """Convenience function to wrap a list of LangChain tools."""
    return [
        GovernedTool(
            name=tool.name,
            description=tool.description,
            original_tool=tool,
            autonomica=autonomica,
            agent_id=agent_id,
        )
        for tool in tools
    ]
```

---

## 8. EXAMPLE: QUICKSTART

```python
# examples/quickstart.py
"""
Autonomica Quickstart — Add governance to any LangChain agent in 3 lines.
"""

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from autonomica import Autonomica
from autonomica.integrations.langchain import wrap_langchain_tools

# 1. Define your tools as normal
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}: {subject}"

@tool
def read_database(query: str) -> str:
    """Read data from the database."""
    return f"Results for: {query}"

@tool
def process_payment(amount: float, recipient: str) -> str:
    """Process a payment to a recipient."""
    return f"Payment of ${amount} sent to {recipient}"

# 2. Wrap tools with Autonomica (THIS IS THE MAGIC)
gov = Autonomica()
tools = wrap_langchain_tools(
    [send_email, read_database, process_payment],
    gov,
    agent_id="finance-assistant"
)

# 3. Create your agent as normal — governance is automatic
llm = ChatOpenAI(model="gpt-4o-mini")
# ... create agent with governed tools ...
# Every tool call now goes through risk scoring and graduated governance.
```

---

## 9. API ENDPOINTS (FastAPI Dashboard Backend)

```
GET  /api/agents                    → List all agent profiles
GET  /api/agents/{agent_id}         → Get agent profile + vagal tone
GET  /api/agents/{agent_id}/actions → Get action history for agent

GET  /api/actions                   → List all recent actions (paginated)
GET  /api/actions/{action_id}       → Get action detail + governance decision

GET  /api/metrics/overview          → Dashboard metrics (total actions, modes distribution, escalation rate)
GET  /api/metrics/vagal-tone        → Vagal tone across all agents
GET  /api/metrics/adaptation        → Threshold changes over time

POST /api/governance/override       → Human override for a pending action
PUT  /api/governance/config         → Update governance configuration
GET  /api/governance/config         → Get current configuration

GET  /api/audit/export              → Export audit log (CSV/JSON) for compliance
```

---

## 10. TESTING STRATEGY

**Unit tests (priority 1):**
- Risk scorer produces expected scores for known inputs
- Governance engine maps scores to correct modes
- Adaptation engine correctly widens/tightens thresholds
- LangChain wrapper correctly intercepts and passes through

**Integration tests (priority 2):**
- Full pipeline: action → score → govern → log → adapt
- Soft gate timeout behavior (action proceeds after timeout)
- Hard gate timeout behavior (action rejected after timeout)
- Escalation notifications actually fire

**Scenario tests (priority 3):**
- New agent: everything escalated → gradually earns autonomy
- Trusted agent: routine tasks auto-approved, novel tasks escalated
- Incident: agent makes mistake → thresholds tighten → recovers over time
- Financial escalation: small amounts auto-approved, large amounts gated

---

## 11. MVP MILESTONES

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Core models + scorer | `models.py`, `scorer.py` with tests passing |
| 3-4 | Governor + interceptor | `governor.py`, `interceptor.py`, LangChain integration |
| 5-6 | Adaptation + storage | `adapter.py`, SQLite storage, audit logging |
| 7-8 | API + basic dashboard | FastAPI endpoints, metrics, Slack escalation |
| 9-10 | Quickstart + docs | Examples, README, getting-started guide |
| 11-12 | Design partner pilot | Deploy with 2-3 real users, collect metrics |

---

## 12. KEY DESIGN PRINCIPLES

1. **< 10ms latency** — Governance must not slow down agents noticeably. Score with heuristics, not LLM calls.
2. **Fail open, not closed** — If Autonomica itself crashes, agent actions should proceed (with logging). Never be the bottleneck.
3. **3-line integration** — If it takes more than 3 lines to add governance, adoption will fail.
4. **Graduated, never binary** — Five modes, not two. The body doesn't just have "alive" and "dead."
5. **Earned trust** — New agents start restricted. Trust is earned through demonstrated reliability, per-tool.
6. **Observable** — Every decision must be explainable. "This was blocked because: financial magnitude 85, agent has only done this 3 times before."

---

*This spec is your source of truth. Build from it. Test against it. Ship with it.*
