"""CrewAI integration — wrap CrewAI BaseTool with Autonomica governance."""
from __future__ import annotations

import inspect
from typing import Any, Optional

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:  # type: ignore
        name: str
        description: str
        def _run(self, *args, **kwargs) -> Any: pass
        async def _arun(self, *args, **kwargs) -> Any: pass

from autonomica.models import ActionType, AgentAction


class GovernedCrewAITool(BaseTool):
    """A CrewAI BaseTool wrapped with Autonomica governance."""
    original_tool: Any
    autonomica: Any
    agent_id: str
    agent_name: str = ""

    def _run(self, **kwargs: Any) -> Any:
        action = self._build_action(kwargs)
        decision = self.autonomica.evaluate_action_sync(action)
        if not decision.approved:
            return (
                f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). "
                f"Risk score: {decision.risk_score.composite_score:.1f}. "
                f"Reason: {decision.risk_score.explanation}"
            )
        try:
            result = self.original_tool._run(**kwargs)
        except Exception as exc:
            self.autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
            raise
        self.autonomica.record_outcome(action.action_id, success=True)
        return result

    async def _arun(self, **kwargs: Any) -> Any:
        action = self._build_action(kwargs)
        decision = await self.autonomica.evaluate_action(action)
        if not decision.approved:
            return (
                f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). "
                f"Risk score: {decision.risk_score.composite_score:.1f}. "
                f"Reason: {decision.risk_score.explanation}"
            )
        try:
            if hasattr(self.original_tool, "_arun") and inspect.iscoroutinefunction(self.original_tool._arun):
                result = await self.original_tool._arun(**kwargs)
            else:
                result = self.original_tool._run(**kwargs)
        except Exception as exc:
            self.autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
            raise
        self.autonomica.record_outcome(action.action_id, success=True)
        return result

    def _infer_action_type(self) -> ActionType:
        name = self.original_tool.name.lower()
        desc = (self.original_tool.description or "").lower()
        if any(w in name or w in desc for w in ["delete", "remove", "drop"]):
            return ActionType.DELETE
        if any(w in name or w in desc for w in ["send", "email", "message", "post", "notify"]):
            return ActionType.COMMUNICATE
        if any(w in name or w in desc for w in ["pay", "transfer", "invoice", "charge"]):
            return ActionType.FINANCIAL
        if any(w in name or w in desc for w in ["write", "update", "create", "insert", "set"]):
            return ActionType.WRITE
        return ActionType.READ

    def _build_action(self, kwargs: dict) -> AgentAction:
        return AgentAction(
            agent_id=self.agent_id,
            agent_name=self.agent_name or self.agent_id,
            tool_name=self.original_tool.name,
            tool_input=kwargs,
            action_type=self._infer_action_type(),
        )


def wrap_crewai_tools(
    tools: list[BaseTool],
    autonomica: Any,
    agent_id: str,
    agent_name: str = "",
) -> list[GovernedCrewAITool]:
    governed = []
    for tool in tools:
        governed.append(
            GovernedCrewAITool(
                name=tool.name,
                description=tool.description,
                original_tool=tool,
                autonomica=autonomica,
                agent_id=agent_id,
                agent_name=agent_name or agent_id,
            )
        )
    return governed
