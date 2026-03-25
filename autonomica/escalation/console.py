"""Console escalation — prints governance events to stdout.

Used as the default escalation backend in local development.
`wait_for_response` returns None immediately (no interactive terminal input).
"""
from __future__ import annotations

from typing import Optional

from autonomica.escalation.base import BaseEscalation
from autonomica.models import AgentAction, GovernanceMode

_MODE_LABELS = {
    GovernanceMode.FULL_AUTO: "FULL_AUTO",
    GovernanceMode.LOG_AND_ALERT: "LOG_AND_ALERT",
    GovernanceMode.SOFT_GATE: "SOFT_GATE",
    GovernanceMode.HARD_GATE: "HARD_GATE",
    GovernanceMode.QUARANTINE: "QUARANTINE",
}


class ConsoleEscalation(BaseEscalation):
    """Writes governance events to stdout. Non-interactive."""

    async def notify(self, action: AgentAction, mode: GovernanceMode) -> None:
        label = _MODE_LABELS.get(mode, mode.name)
        print(
            f"[AUTONOMICA] [{label}] agent={action.agent_id!r} "
            f"tool={action.tool_name!r} action_id={action.action_id}"
        )

    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        # Non-interactive: no human at the terminal, always time out.
        return None
