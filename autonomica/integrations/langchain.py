"""LangChain integration — GovernedTool wraps BaseTool with Autonomica governance."""
from __future__ import annotations

import inspect
import json
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from autonomica.models import ActionType, AgentAction


class GovernedTool(BaseTool):
    """A LangChain BaseTool wrapped with Autonomica governance.

    Every call to _run / _arun flows through the full governance pipeline:
    score → decide mode → enforce → log → update profile.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Use Any to avoid Pydantic resolving forward references at class
    # definition time (langchain BaseTool stores original_tool as a field).
    original_tool: Any   # BaseTool
    autonomica: Any      # Autonomica — injected at wrap time
    agent_id: str
    agent_name: str = ""

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _run(
        self,
        *args: Any,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        action = self._build_action(args, kwargs)
        decision = self.autonomica.evaluate_action_sync(action)

        if not decision.approved:
            return (
                f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). "
                f"Risk score: {decision.risk_score.composite_score:.1f}. "
                f"Reason: {decision.risk_score.explanation}"
            )

        try:
            result = self._call_original_run(args, kwargs, run_manager)
        except Exception as exc:
            self.autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
            raise

        self.autonomica.record_outcome(action.action_id, success=True)
        return result

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        action = self._build_action(args, kwargs)
        decision = await self.autonomica.evaluate_action(action)

        if not decision.approved:
            return (
                f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). "
                f"Risk score: {decision.risk_score.composite_score:.1f}. "
                f"Reason: {decision.risk_score.explanation}"
            )

        try:
            result = await self._call_original_arun(args, kwargs, run_manager)
        except Exception as exc:
            self.autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
            raise

        self.autonomica.record_outcome(action.action_id, success=True)
        return result

    # ------------------------------------------------------------------
    # Action type inference (from spec §7)
    # ------------------------------------------------------------------

    def _infer_action_type(self) -> ActionType:
        """Infer ActionType from tool name and description keywords."""
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_args(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Normalise LangChain 1.x tool-call inputs into proper kwargs.

        Handles two non-standard calling conventions LangChain 1.x may use:
        1. Single JSON string positional arg: _run('{"amount": 500000, ...}')
        2. Positional list in kwargs:          _run(args=[500000, "vendor@..."])
        In both cases we map values to the original tool's _run parameter names.
        """
        # Case 1: single JSON string
        if len(args) == 1 and not kwargs and isinstance(args[0], str):
            try:
                parsed = json.loads(args[0])
                if isinstance(parsed, dict):
                    return (), parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Case 2: LangChain passes positional list as {'args': [...]}
        if not args and set(kwargs.keys()) == {"args"} and isinstance(kwargs["args"], list):
            positional = kwargs["args"]
            try:
                params = [
                    p for p in inspect.signature(self.original_tool._run).parameters.values()
                    if p.name not in ("self", "run_manager", "kwargs")
                    and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ]
                if len(positional) <= len(params):
                    return (), dict(zip([p.name for p in params], positional))
            except (ValueError, TypeError):
                pass

        return args, kwargs

    def _build_action(self, args: tuple, kwargs: dict) -> AgentAction:
        args, kwargs = self._parse_args(args, kwargs)
        tool_input = kwargs if kwargs else {"input": args[0] if args else ""}
        return AgentAction(
            agent_id=self.agent_id,
            agent_name=self.agent_name or self.agent_id,
            tool_name=self.original_tool.name,
            tool_input=tool_input,
            action_type=self._infer_action_type(),
        )

    def _call_original_run(
        self, args: tuple, kwargs: dict, run_manager: Any
    ) -> Any:
        args, kwargs = self._parse_args(args, kwargs)
        extra = {"run_manager": run_manager} if run_manager is not None else {}
        try:
            return self.original_tool._run(*args, **extra, **kwargs)
        except TypeError:
            return self.original_tool._run(*args, **kwargs)

    async def _call_original_arun(
        self, args: tuple, kwargs: dict, run_manager: Any
    ) -> Any:
        args, kwargs = self._parse_args(args, kwargs)
        extra = {"run_manager": run_manager} if run_manager is not None else {}
        try:
            return await self.original_tool._arun(*args, **extra, **kwargs)
        except TypeError:
            return await self.original_tool._arun(*args, **kwargs)


# ---------------------------------------------------------------------------
# Convenience function — the 3-line integration path
# ---------------------------------------------------------------------------

def wrap_langchain_tools(
    tools: list[BaseTool],
    autonomica: Any,  # Autonomica
    agent_id: str,
    agent_name: str = "",
) -> list[GovernedTool]:
    """Wrap a list of LangChain tools with Autonomica governance.

    Example::

        gov = Autonomica()
        tools = wrap_langchain_tools(tools, gov, agent_id="my-agent")
    """
    governed = []
    for tool in tools:
        # Forward the original tool's input schema so LangChain 1.x passes
        # proper named kwargs rather than a positional list or JSON string.
        schema = getattr(tool, "args_schema", None)
        if schema is None:
            try:
                schema = tool.get_input_schema()
            except Exception:
                schema = None

        governed.append(GovernedTool(
            name=tool.name,
            description=tool.description or "",
            original_tool=tool,
            autonomica=autonomica,
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
            args_schema=schema,
        ))
    return governed
