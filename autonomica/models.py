from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class GovernanceMode(IntEnum):
    """Five graduated governance modes — like the autonomic nervous system."""
    FULL_AUTO = 0       # Score 0-15: Routine. Log only.
    LOG_AND_ALERT = 1   # Score 16-35: Proceed + async notification.
    SOFT_GATE = 2       # Score 36-60: Pause 30-60s. Auto-proceed unless human intervenes.
    HARD_GATE = 3       # Score 61-85: Stop. Human must explicitly approve.
    QUARANTINE = 4      # Score 86-100: Blocked. Full review required.


class ActionType(str, Enum):
    """Categories of agent actions by reversibility."""
    READ = "read"                # Fully reversible (database reads, API gets)
    WRITE = "write"              # Partially reversible (database writes, file creation)
    COMMUNICATE = "communicate"  # Irreversible (emails, messages, API posts)
    DELETE = "delete"            # Catastrophic (database deletes, file deletion)
    FINANCIAL = "financial"      # High-stakes (transactions, payments, contracts)


class AgentAction(BaseModel):
    """Represents a single action an agent wants to take."""
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    agent_name: str
    tool_name: str
    tool_input: dict[str, Any]
    action_type: ActionType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    approved: bool
    escalated_to: Optional[str] = None
    human_override: Optional[bool] = None
    decision_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Set when the decision was produced by the fail-policy handler rather than
    # the normal scoring pipeline (e.g. scorer crash, storage down).
    pipeline_error: Optional[str] = None


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
    mode_thresholds: dict[str, float] = Field(default_factory=lambda: {
        "full_auto_max": 15.0,
        "log_alert_max": 35.0,
        "soft_gate_max": 60.0,
        "hard_gate_max": 85.0,
        # Above 85 = quarantine
    })
    per_tool_trust: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
