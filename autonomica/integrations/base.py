"""Abstract base interface for framework-specific integrations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseIntegration(ABC):
    """Implemented once per agent framework (LangChain, CrewAI, etc.)."""

    @abstractmethod
    def wrap_tool(self, tool: Any, agent_id: str, agent_name: str | None = None) -> Any:
        """Wrap a single tool with governance. Returns a governed equivalent."""

    @abstractmethod
    def wrap_tools(
        self, tools: list[Any], agent_id: str, agent_name: str | None = None
    ) -> list[Any]:
        """Wrap a list of tools with governance."""
