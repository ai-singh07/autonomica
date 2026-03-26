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
        risk_score: Optional[RiskScore] = None,
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
        # Provide a no-op RiskScore if not supplied (backwards compatibility).
        if risk_score is None:
            risk_score = RiskScore(
                composite_score=0.0,
                financial_magnitude=0.0,
                data_sensitivity=0.0,
                reversibility=0.0,
                agent_track_record=50.0,
                novelty=0.0,
                cascade_risk=0.0,
                explanation="n/a",
            )

        if mode == GovernanceMode.FULL_AUTO:
            asyncio.create_task(escalation.notify(action, mode, risk_score))
            return True

        if mode == GovernanceMode.LOG_AND_ALERT:
            asyncio.create_task(escalation.notify(action, mode, risk_score))
            return True

        if mode == GovernanceMode.SOFT_GATE:
            await escalation.notify(action, mode, risk_score)
            response = await escalation.wait_for_response(
                action.action_id, self._soft_gate_timeout
            )
            return response is not False

        if mode == GovernanceMode.HARD_GATE:
            await escalation.notify(action, mode, risk_score)
            response = await escalation.wait_for_response(
                action.action_id, self._hard_gate_timeout
            )
            return response is True

        if mode == GovernanceMode.QUARANTINE:
            await escalation.notify(action, mode, risk_score)
            return False

        return False  # pragma: no cover
