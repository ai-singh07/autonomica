"""Abstract storage interface for agent profiles and governance decisions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from autonomica.models import AgentProfile, GovernanceDecision


class BaseStorage(ABC):
    """Interface all storage backends must implement.

    Every method is async so implementations can use async DB drivers without
    wrapping.  Sync backends should run their blocking calls in a thread
    (e.g. ``asyncio.to_thread``).
    """

    @abstractmethod
    async def save_profile(self, profile: AgentProfile) -> None:
        """Upsert an agent profile."""

    @abstractmethod
    async def load_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Return the agent profile, or None if not found."""

    @abstractmethod
    async def save_decision(
        self,
        decision: GovernanceDecision,
        agent_id: str = "unknown",
    ) -> None:
        """Persist a governance decision (insert or replace)."""

    @abstractmethod
    async def load_decision(self, action_id: str) -> Optional[GovernanceDecision]:
        """Return a governance decision by action_id, or None."""

    @abstractmethod
    async def list_profiles(self) -> list[AgentProfile]:
        """Return all agent profiles."""

    @abstractmethod
    async def list_decisions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[GovernanceDecision]:
        """Return recent decisions, optionally filtered by agent_id."""

    @abstractmethod
    async def close(self) -> None:
        """Release any persistent resources (connections, file handles)."""
