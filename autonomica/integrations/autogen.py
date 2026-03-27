"""AutoGen integration — wrap AutoGen tool callables with Autonomica governance."""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from autonomica.models import ActionType, AgentAction


def wrap_autogen_callable(
    f: Callable,
    autonomica: Any,
    agent_id: str,
    name: str | None = None,
    description: str | None = None,
    agent_name: str = "",
) -> Callable:
    tool_name = name or f.__name__
    tool_desc = description or f.__doc__ or ""

    def _infer_action_type() -> ActionType:
        n = tool_name.lower()
        d = tool_desc.lower()
        if any(w in n or w in d for w in ["delete", "remove", "drop"]):
            return ActionType.DELETE
        if any(w in n or w in d for w in ["send", "email", "message", "post", "notify"]):
            return ActionType.COMMUNICATE
        if any(w in n or w in d for w in ["pay", "transfer", "invoice", "charge"]):
            return ActionType.FINANCIAL
        if any(w in n or w in d for w in ["write", "update", "create", "insert", "set"]):
            return ActionType.WRITE
        return ActionType.READ

    action_type = _infer_action_type()

    if inspect.iscoroutinefunction(f):
        @functools.wraps(f)
        async def wrapped_async(*args, **kwargs):
            action = AgentAction(
                agent_id=agent_id,
                agent_name=agent_name or agent_id,
                tool_name=tool_name,
                tool_input=kwargs,
                action_type=action_type,
            )
            decision = await autonomica.evaluate_action(action)
            if not decision.approved:
                return f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). Risk score: {decision.risk_score.composite_score:.1f}. Reason: {decision.risk_score.explanation}"
            try:
                result = await f(*args, **kwargs)
            except Exception as exc:
                autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
                raise
            autonomica.record_outcome(action.action_id, success=True)
            return result
        return wrapped_async
    else:
        @functools.wraps(f)
        def wrapped_sync(*args, **kwargs):
            action = AgentAction(
                agent_id=agent_id,
                agent_name=agent_name or agent_id,
                tool_name=tool_name,
                tool_input=kwargs,
                action_type=action_type,
            )
            decision = autonomica.evaluate_action_sync(action)
            if not decision.approved:
                return f"[AUTONOMICA] Action blocked (Mode: {decision.mode.name}). Risk score: {decision.risk_score.composite_score:.1f}. Reason: {decision.risk_score.explanation}"
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                autonomica.record_outcome(action.action_id, success=False, notes=str(exc))
                raise
            autonomica.record_outcome(action.action_id, success=True)
            return result
        return wrapped_sync


def govern_autogen_agent(
    agent: Any,
    autonomica: Any,
    agent_id: str,
    agent_name: str = "",
) -> None:
    if not hasattr(agent, "function_map"):
        return
    for name, func in agent.function_map.items():
        agent.function_map[name] = wrap_autogen_callable(
            func, autonomica, agent_id, name=name, agent_name=agent_name
        )
