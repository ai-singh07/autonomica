"""AutonomicaConfig — central configuration model (Section 6 of spec).

Pass an instance to Autonomica() to override any default.  All fields have
sensible defaults so the zero-config path still works::

    gov = Autonomica()                        # all defaults
    gov = Autonomica(config=AutonomicaConfig(soft_gate_timeout_seconds=10))
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


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
    # How fast thresholds and trust score adapt.  0 = never adapt, 1 = instant.
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
