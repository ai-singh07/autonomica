"""Structured audit logging for every governance decision.

Every event is written as a JSON object on a single line (JSONL format) so
the log can be streamed, grepped, and fed into log-aggregation systems
(Datadog, Splunk, etc.) without parsing overhead.

Events
------
``governance_decision``  Fired after every call to evaluate_action().
``human_override``       Fired when record_human_override() is called.
``incident``             Fired when record_outcome(success=False) is called.

Usage::

    # Default: logs to Python logging at INFO level (no file I/O)
    audit = AuditLogger()

    # With a JSONL file:
    audit = AuditLogger(log_file="/var/log/autonomica/audit.jsonl")

    # Export for compliance:
    n = audit.export("/tmp/export.jsonl")          # JSONL
    n = audit.export("/tmp/export.json", fmt="json")  # pretty JSON array
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from autonomica.models import AgentAction, AgentProfile, GovernanceDecision

_log = logging.getLogger("autonomica.audit")


class AuditLogger:
    """Write governance events to a JSONL file and/or the Python logging system."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        """
        Args:
            log_file: Path to a JSONL file.  Directory is created automatically.
                      Pass None to log to the Python logger only.
        """
        self._log_file = log_file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # ── Event emitters ────────────────────────────────────────────────────────

    def log_decision(
        self,
        action: AgentAction,
        decision: GovernanceDecision,
        profile: AgentProfile,
    ) -> None:
        """Emit a ``governance_decision`` event."""
        entry: dict[str, Any] = {
            "event": "governance_decision",
            "timestamp": _now(),
            "action_id": decision.action_id,
            "agent_id": action.agent_id,
            "agent_name": action.agent_name,
            "tool_name": action.tool_name,
            "action_type": action.action_type.value,
            "composite_score": round(decision.risk_score.composite_score, 2),
            "governance_mode": decision.mode.name,
            "approved": decision.approved,
            "decision_time_ms": round(decision.decision_time_ms, 2),
            "trust_score": round(profile.trust_score, 2),
            "vagal_tone": round(profile.vagal_tone, 2),
            "risk_breakdown": {
                "financial_magnitude": decision.risk_score.financial_magnitude,
                "data_sensitivity": decision.risk_score.data_sensitivity,
                "reversibility": decision.risk_score.reversibility,
                "agent_track_record": decision.risk_score.agent_track_record,
                "novelty": decision.risk_score.novelty,
                "cascade_risk": decision.risk_score.cascade_risk,
            },
        }
        self._write(entry)

    def log_override(
        self,
        action_id: str,
        agent_id: str,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Emit a ``human_override`` event."""
        entry: dict[str, Any] = {
            "event": "human_override",
            "timestamp": _now(),
            "action_id": action_id,
            "agent_id": agent_id,
            "approved": approved,
            "reason": reason,
        }
        self._write(entry)

    def log_incident(
        self,
        action_id: str,
        agent_id: str,
        notes: str = "",
    ) -> None:
        """Emit an ``incident`` event."""
        entry: dict[str, Any] = {
            "event": "incident",
            "timestamp": _now(),
            "action_id": action_id,
            "agent_id": agent_id,
            "notes": notes,
        }
        self._write(entry)

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self, output_path: str, fmt: str = "jsonl") -> int:
        """Export the audit log to *output_path*.

        Args:
            output_path: Destination file.
            fmt: ``"jsonl"`` (default) or ``"json"`` (pretty JSON array).

        Returns:
            Number of events exported.

        Raises:
            ValueError: If no log_file was configured.
        """
        if not self._log_file:
            raise ValueError(
                "AuditLogger was created without a log_file; nothing to export."
            )
        src = Path(self._log_file)
        if not src.exists():
            return 0

        entries = [
            json.loads(line)
            for line in src.read_text().splitlines()
            if line.strip()
        ]

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            dest.write_text(json.dumps(entries, indent=2))
        else:
            dest.write_text("\n".join(json.dumps(e) for e in entries))

        return len(entries)

    def read_entries(self) -> list[dict[str, Any]]:
        """Return all logged entries as a list of dicts (for testing / API)."""
        if not self._log_file:
            return []
        src = Path(self._log_file)
        if not src.exists():
            return []
        return [
            json.loads(line)
            for line in src.read_text().splitlines()
            if line.strip()
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, separators=(",", ":"))
        _log.info(line)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
