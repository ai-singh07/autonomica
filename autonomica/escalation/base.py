"""Abstract base class for escalation channels."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from autonomica.models import AgentAction, GovernanceMode


class BaseEscalation(ABC):
    """Interface all escalation backends must implement."""

    @abstractmethod
    async def notify(self, action: AgentAction, mode: GovernanceMode) -> None:
        """
        Send a notification that an action has been intercepted.
        Called for every mode except FULL_AUTO (which logs async).
        Should be fast — do not block the governance pipeline.
        """

    @abstractmethod
    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        """
        Wait up to `timeout` seconds for a human response to a pending action.

        Returns:
            True  — human explicitly approved the action
            False — human explicitly vetoed/rejected the action
            None  — timeout elapsed with no response
        """
