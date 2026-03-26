"""AutonomicaConfig — central configuration model (Section 6 of spec).

Pass an instance to Autonomica() to override any default.  All fields have
sensible defaults so the zero-config path still works::

    gov = Autonomica()                        # all defaults
    gov = Autonomica(config=AutonomicaConfig(soft_gate_timeout_seconds=10))
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# The six signal names recognised by RiskScorer.  Used to validate tool_overrides.
_VALID_SIGNALS: frozenset[str] = frozenset({
    "financial_magnitude",
    "data_sensitivity",
    "reversibility",
    "agent_track_record",
    "novelty",
    "cascade_risk",
})


class AutonomicaConfig(BaseModel):
    """Global configuration for Autonomica.  All fields are optional."""

    # ── Governance timeouts ────────────────────────────────────────────────
    soft_gate_timeout_seconds: float = 60.0
    hard_gate_timeout_seconds: float = 300.0
    default_trust_score: float = 50.0

    # ── Scoring weights (must sum to 1.0) ─────────────────────────────────
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "financial_magnitude": 0.25,
            "data_sensitivity": 0.20,
            "reversibility": 0.20,
            "agent_track_record": 0.15,
            "novelty": 0.10,
            "cascade_risk": 0.10,
        }
    )

    # ── Financial thresholds (currency-agnostic amounts) ──────────────────
    financial_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "low": 100.0,
            "medium": 10_000.0,
            "high": 100_000.0,
            "critical": 1_000_000.0,
        }
    )

    # ── Data-sensitivity keyword lists ────────────────────────────────────
    pii_patterns: list[str] = Field(
        default_factory=lambda: [
            "email", "ssn", "phone", "credit_card", "address", "dob",
        ]
    )
    financial_data_patterns: list[str] = Field(
        default_factory=lambda: [
            "account_number", "routing", "balance", "salary",
        ]
    )
    health_data_patterns: list[str] = Field(
        default_factory=lambda: [
            "diagnosis", "medication", "patient", "medical",
        ]
    )

    # ── Adaptation ────────────────────────────────────────────────────────
    # When False (default) thresholds never change — governance is fully
    # predictable and auditable.  Set True to enable dampened adaptive
    # thresholds that drift based on agent behaviour over time.
    adaptation_enabled: bool = False

    # How fast thresholds and trust score adapt when adaptation_enabled=True.
    # 0.1 = very slow drift, 1.0 = fast response.
    adaptation_rate: float = 0.5
    # Don't adapt until this many actions have been observed (avoids early noise).
    min_actions_before_adaptation: int = 10  # min observations before thresholds adapt

    # ── Storage ───────────────────────────────────────────────────────────
    storage_backend: str = "sqlite"
    database_url: str = "sqlite:///autonomica.db"

    # ── Escalation ────────────────────────────────────────────────────────
    escalation_backend: str = "console"
    slack_webhook_url: Optional[str] = None

    # ── Audit ─────────────────────────────────────────────────────────────
    # Path to a JSONL audit log file.  None = log to Python logging only.
    audit_log_file: Optional[str] = None

    # ── Fail policy ───────────────────────────────────────────────────────
    # What to do when the governance pipeline itself throws an unexpected error
    # (scorer crash, storage down, etc.).
    #
    # "open"     — always proceed (approved=True, LOG_AND_ALERT).
    #              Use in high-availability read-heavy workloads where false
    #              blocks are more costly than missed governance.
    #
    # "closed"   — always block (approved=False, QUARANTINE).
    #              Use in highly regulated environments where any governance
    #              gap is unacceptable.
    #
    # "adaptive" — (default) risk-sensitive:
    #              READ actions → fail open (LOG_AND_ALERT, approved=True)
    #              WRITE / DELETE / COMMUNICATE / FINANCIAL → fail to HARD_GATE
    #              (block until a human explicitly approves or the gate times out).
    fail_policy: Literal["open", "closed", "adaptive"] = "adaptive"

    # ── SQL-aware scoring ─────────────────────────────────────────────────
    # Table names whose presence in any tool_input string bumps data_sensitivity
    # by +30.  Matched as whole words (case-insensitive) so "super_users" does
    # not trigger a match on "users".
    sensitive_tables: list[str] = Field(
        default_factory=lambda: [
            "users", "payments", "accounts", "credentials", "medical_records",
        ]
    )

    # ── Per-tool signal overrides ─────────────────────────────────────────
    # Pin one or more risk signal scores for a named tool, bypassing the
    # heuristic scorer for those signals.  All other signals still score
    # normally.  Values must be in [0, 100].
    #
    # Example::
    #
    #   tool_overrides={
    #       "write_tutorial": {"data_sensitivity": 0, "financial_magnitude": 0},
    #       "process_payment": {"financial_magnitude": 90, "reversibility": 80},
    #   }
    tool_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("tool_overrides")
    @classmethod
    def _validate_tool_overrides(
        cls, v: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        for tool_name, signals in v.items():
            for signal, value in signals.items():
                if signal not in _VALID_SIGNALS:
                    raise ValueError(
                        f"Unknown signal {signal!r} in tool_overrides[{tool_name!r}]. "
                        f"Valid signals: {sorted(_VALID_SIGNALS)}"
                    )
                if not (0.0 <= float(value) <= 100.0):
                    raise ValueError(
                        f"Signal value {value!r} is out of range [0, 100] "
                        f"in tool_overrides[{tool_name!r}][{signal!r}]"
                    )
        return v
