"""Universal ``@govern`` decorator — drop-in governance for any Python function.

Usage::

    from autonomica import govern

    @govern(agent_id="finance-bot", action_type="financial")
    def process_payment(amount: float, recipient: str) -> str:
        return f"Paid ${amount} to {recipient}"

    @govern(agent_id="research-bot")          # action_type inferred from name
    async def search_database(query: str) -> str:
        return f"Results for: {query}"

Works with **sync and async** functions.  Goes through the full pipeline:
score → decide mode → enforce → audit log → adapt.

If governance blocks the action a ``GovernanceBlocked`` exception is raised
containing the ``GovernanceDecision`` so callers can inspect the reason.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from autonomica.models import ActionType, AgentAction, GovernanceDecision


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class GovernanceBlocked(Exception):
    """Raised when governance denies a function call.

    Attributes:
        decision: The full ``GovernanceDecision`` explaining why the call
                  was blocked (mode, risk_score, etc.).
        function_name: The name of the blocked function.
    """

    def __init__(self, decision: GovernanceDecision, function_name: str) -> None:
        self.decision = decision
        self.function_name = function_name
        super().__init__(
            f"Governance blocked '{function_name}' "
            f"(mode={decision.mode.name}, "
            f"score={decision.risk_score.composite_score:.1f}, "
            f"action_id={decision.action_id})"
        )


# ---------------------------------------------------------------------------
# Action-type inference from function name
# ---------------------------------------------------------------------------

_READ_KEYWORDS = frozenset({
    "read", "get", "fetch", "search", "query", "list", "find",
    "select", "load", "retrieve", "lookup", "scan", "show", "view",
})
_WRITE_KEYWORDS = frozenset({
    "write", "create", "update", "insert", "save", "store", "put",
    "set", "upload", "add", "edit", "modify", "patch", "upsert",
})
_COMMUNICATE_KEYWORDS = frozenset({
    "send", "email", "notify", "post", "publish", "message",
    "alert", "broadcast", "push", "forward", "reply",
})
_DELETE_KEYWORDS = frozenset({
    "delete", "remove", "drop", "purge", "clear", "destroy",
    "archive", "truncate", "wipe",
})
_FINANCIAL_KEYWORDS = frozenset({
    "pay", "payment", "transfer", "charge", "invoice", "bill",
    "refund", "withdraw", "deposit", "purchase", "buy", "sell",
    "transaction", "fund", "debit", "credit",
})

_ACTION_TYPE_STR_MAP: dict[str, ActionType] = {
    "read":       ActionType.READ,
    "write":      ActionType.WRITE,
    "communicate": ActionType.COMMUNICATE,
    "delete":     ActionType.DELETE,
    "financial":  ActionType.FINANCIAL,
}


def _infer_action_type(func_name: str) -> ActionType:
    """Infer ActionType from function name by keyword matching.

    Splits the name on underscores and checks each word against priority-
    ordered keyword sets.  Falls back to READ (safest / least restrictive).
    """
    words = set(func_name.lower().split("_"))
    if words & _DELETE_KEYWORDS:
        return ActionType.DELETE
    if words & _FINANCIAL_KEYWORDS:
        return ActionType.FINANCIAL
    if words & _COMMUNICATE_KEYWORDS:
        return ActionType.COMMUNICATE
    if words & _WRITE_KEYWORDS:
        return ActionType.WRITE
    if words & _READ_KEYWORDS:
        return ActionType.READ
    return ActionType.READ   # safe default


def _parse_action_type(value: str | ActionType | None, func_name: str) -> ActionType:
    """Resolve the caller-supplied action_type (string, enum, or None)."""
    if value is None:
        return _infer_action_type(func_name)
    if isinstance(value, ActionType):
        return value
    lowered = value.lower().replace("-", "_")
    if lowered not in _ACTION_TYPE_STR_MAP:
        raise ValueError(
            f"Unknown action_type {value!r}. "
            f"Valid values: {sorted(_ACTION_TYPE_STR_MAP)}"
        )
    return _ACTION_TYPE_STR_MAP[lowered]


# ---------------------------------------------------------------------------
# Module-level default Autonomica instance (created lazily)
# ---------------------------------------------------------------------------

_default_instance: Any = None   # autonomica.Autonomica | None


def _get_default_instance() -> Any:
    global _default_instance
    if _default_instance is None:
        from autonomica.interceptor import Autonomica  # avoid circular at module level
        _default_instance = Autonomica()
    return _default_instance


# ---------------------------------------------------------------------------
# Governance helpers
# ---------------------------------------------------------------------------

def _build_action(
    func: Callable,
    args: tuple,
    kwargs: dict,
    agent_id: str,
    agent_name: str,
    action_type: ActionType,
) -> AgentAction:
    """Bind call arguments to parameter names to populate tool_input."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        tool_input: dict[str, Any] = dict(bound.arguments)
    except TypeError:
        # If binding fails fall back to positional/keyword representation
        tool_input = {f"arg_{i}": v for i, v in enumerate(args)}
        tool_input.update(kwargs)

    return AgentAction(
        agent_id=agent_id,
        agent_name=agent_name,
        tool_name=func.__name__,
        tool_input=tool_input,
        action_type=action_type,
    )


async def _evaluate_async(gov: Any, action: AgentAction) -> GovernanceDecision:
    return await gov.evaluate_action(action)


def _evaluate_sync(gov: Any, action: AgentAction) -> GovernanceDecision:
    """Run evaluate_action from a synchronous call site.

    Delegates to Autonomica.evaluate_action_sync which already handles
    both 'running loop' and 'no loop' contexts correctly.
    """
    return gov.evaluate_action_sync(action)


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

def govern(
    agent_id: str,
    *,
    action_type: str | ActionType | None = None,
    agent_name: str | None = None,
    autonomica: Any = None,
) -> Callable:
    """Governance decorator for any Python function.

    Args:
        agent_id:    Identifier for the agent making the call.  Required.
        action_type: Override the action type (``"read"``, ``"write"``,
                     ``"communicate"``, ``"delete"``, ``"financial"``).
                     Inferred from the function name if omitted.
        agent_name:  Human-readable agent label.  Defaults to ``agent_id``.
        autonomica:  An existing ``Autonomica`` instance to use.  A shared
                     module-level instance is created on first use if omitted.

    Returns:
        A decorator that wraps the target function with governance.

    Raises:
        GovernanceBlocked: When the governance engine denies the action.
    """
    def decorator(func: Callable) -> Callable:
        resolved_action_type = _parse_action_type(action_type, func.__name__)
        resolved_agent_name  = agent_name or agent_id

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                gov = autonomica if autonomica is not None else _get_default_instance()
                action = _build_action(
                    func, args, kwargs,
                    agent_id, resolved_agent_name, resolved_action_type,
                )
                decision = await _evaluate_async(gov, action)
                if not decision.approved:
                    raise GovernanceBlocked(decision, func.__name__)
                return await func(*args, **kwargs)

            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                gov = autonomica if autonomica is not None else _get_default_instance()
                action = _build_action(
                    func, args, kwargs,
                    agent_id, resolved_agent_name, resolved_action_type,
                )
                decision = _evaluate_sync(gov, action)
                if not decision.approved:
                    raise GovernanceBlocked(decision, func.__name__)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
