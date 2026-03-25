"""Governance Mode Engine — the 'autonomic switch'.

Maps risk scores to governance modes and enforces the appropriate
response: pass-through, notify, gate, or block.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from autonomica.escalation.base import BaseEscalation
from autonomica.models import AgentAction, AgentProfile, GovernanceMode, RiskScore


class GovernanceEngine:
    """Maps risk scores to governance modes and enforces them."""

    def __init__(self, config: Any = None) -> None:
        """
        Args:
            config: Optional AutonomicaConfig. Reads soft_gate_timeout_seconds
                    and hard_gate_timeout_seconds (defaults: 60s / 300s).
        """
        if config is not None:
            self._soft_gate_timeout = float(
                getattr(config, "soft_gate_timeout_seconds", 60)
            )
            self._hard_gate_timeout = float(
                getattr(config, "hard_gate_timeout_seconds", 300)
            )
        else:
            self._soft_gate_timeout = 60.0
            self._hard_gate_timeout = 300.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self, risk_score: RiskScore, agent_profile: AgentProfile
    ) -> GovernanceMode:
        """
        Map a composite risk score to a governance mode using the agent's
        personal (adaptive) thresholds.
        """
        score = risk_score.composite_score
        t = agent_profile.mode_thresholds

        if score <= t["full_auto_max"]:
            return GovernanceMode.FULL_AUTO
        elif score <= t["log_alert_max"]:
            return GovernanceMode.LOG_AND_ALERT
        elif score <= t["soft_gate_max"]:
            return GovernanceMode.SOFT_GATE
        elif score <= t["hard_gate_max"]:
            return GovernanceMode.HARD_GATE
        else:
            return GovernanceMode.QUARANTINE

    async def enforce(
        self,
        mode: GovernanceMode,
        action: AgentAction,
        escalation: BaseEscalation,
    ) -> bool:
        """
        Enforce the governance mode. Returns True if the action should proceed.

        FULL_AUTO   → True immediately; notification scheduled as background task.
        LOG_AND_ALERT → True immediately; notification scheduled as background task.
        SOFT_GATE   → Notify, then wait up to soft_gate_timeout. Proceed unless
                      the human explicitly vetoes (False). Timeout → proceed.
        HARD_GATE   → Notify, then wait up to hard_gate_timeout for explicit
                      approval. Timeout or rejection → block.
        QUARANTINE  → Notify and block immediately.
        """
        if mode == GovernanceMode.FULL_AUTO:
            # Fire-and-forget: don't block the agent on logging.
            asyncio.create_task(escalation.notify(action, mode))
            return True

        if mode == GovernanceMode.LOG_AND_ALERT:
            # Fire-and-forget: proceed immediately, notification is async.
            asyncio.create_task(escalation.notify(action, mode))
            return True

        if mode == GovernanceMode.SOFT_GATE:
            await escalation.notify(action, mode)
            response = await escalation.wait_for_response(
                action.action_id, self._soft_gate_timeout
            )
            # Proceed unless human explicitly vetoed (False).
            # Timeout (None) and approval (True) both allow the action.
            return response is not False

        if mode == GovernanceMode.HARD_GATE:
            await escalation.notify(action, mode)
            response = await escalation.wait_for_response(
                action.action_id, self._hard_gate_timeout
            )
            # Block unless human explicitly approved (True).
            # Timeout (None) and rejection (False) both block.
            return response is True

        if mode == GovernanceMode.QUARANTINE:
            await escalation.notify(action, mode)
            return False

        # Unreachable with a valid GovernanceMode, but fail-safe.
        return False  # pragma: no cover
