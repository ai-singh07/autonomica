"""Tests for the @govern decorator (autonomica/decorator.py)."""
from __future__ import annotations

import asyncio

import pytest

from autonomica import (
    Autonomica,
    AutonomicaConfig,
    GovernanceBlocked,
    GovernanceMode,
    govern,
)
from autonomica.decorator import _infer_action_type
from autonomica.escalation.base import BaseEscalation
from autonomica.models import ActionType, AgentAction, GovernanceMode, RiskScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SilentEscalation(BaseEscalation):
    async def notify(self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore) -> None:
        pass
    async def wait_for_response(self, action_id: str, timeout: float):
        return None   # SOFT_GATE auto-proceeds; HARD_GATE auto-blocks


def _open_gov() -> Autonomica:
    """Autonomica that auto-approves everything (short timeouts, fail-open)."""
    return Autonomica(
        config=AutonomicaConfig(
            soft_gate_timeout_seconds=0.001,
            hard_gate_timeout_seconds=0.001,
            fail_policy="open",
        ),
        escalation=_SilentEscalation(),
    )


def _blocking_gov(agent_id: str = "bot") -> Autonomica:
    """Autonomica that blocks everything by zeroing all mode thresholds.

    Any composite score > 0 → QUARANTINE → approved=False.
    The cascade_risk default (20) guarantees every action scores above 0.
    """
    gov = Autonomica(
        config=AutonomicaConfig(
            soft_gate_timeout_seconds=0.001,
            hard_gate_timeout_seconds=0.001,
        ),
        escalation=_SilentEscalation(),
    )
    # Pre-create the profile and zero all thresholds so every action →
    # QUARANTINE regardless of its composite score.
    profile = gov._get_or_create_profile(agent_id, agent_id)
    for key in profile.mode_thresholds:
        profile.mode_thresholds[key] = 0.0
    return gov


# ---------------------------------------------------------------------------
# Action-type inference
# ---------------------------------------------------------------------------

class TestActionTypeInference:
    def test_read_keywords(self):
        assert _infer_action_type("search_database") == ActionType.READ
        assert _infer_action_type("get_user") == ActionType.READ
        assert _infer_action_type("fetch_records") == ActionType.READ
        assert _infer_action_type("list_orders") == ActionType.READ
        assert _infer_action_type("find_report") == ActionType.READ

    def test_write_keywords(self):
        assert _infer_action_type("create_record") == ActionType.WRITE
        assert _infer_action_type("update_config") == ActionType.WRITE
        assert _infer_action_type("save_document") == ActionType.WRITE
        assert _infer_action_type("insert_row") == ActionType.WRITE

    def test_communicate_keywords(self):
        assert _infer_action_type("send_email") == ActionType.COMMUNICATE
        assert _infer_action_type("notify_user") == ActionType.COMMUNICATE
        assert _infer_action_type("post_message") == ActionType.COMMUNICATE

    def test_delete_keywords(self):
        assert _infer_action_type("delete_record") == ActionType.DELETE
        assert _infer_action_type("remove_user") == ActionType.DELETE
        assert _infer_action_type("purge_logs") == ActionType.DELETE

    def test_financial_keywords(self):
        assert _infer_action_type("process_payment") == ActionType.FINANCIAL
        assert _infer_action_type("transfer_funds") == ActionType.FINANCIAL
        assert _infer_action_type("charge_customer") == ActionType.FINANCIAL
        assert _infer_action_type("refund_order") == ActionType.FINANCIAL

    def test_unknown_falls_back_to_read(self):
        assert _infer_action_type("do_something") == ActionType.READ
        assert _infer_action_type("run_job") == ActionType.READ

    def test_delete_beats_read(self):
        # "delete" should win over any read-like word in the name
        assert _infer_action_type("delete_and_get") == ActionType.DELETE

    def test_financial_beats_write(self):
        assert _infer_action_type("save_payment") == ActionType.FINANCIAL


# ---------------------------------------------------------------------------
# Sync functions
# ---------------------------------------------------------------------------

class TestSyncDecorator:
    def test_basic_sync_function_executes(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type="read", autonomica=gov)
        def search(query: str) -> str:
            return f"results:{query}"

        result = search("hello")
        assert result == "results:hello"

    def test_sync_preserves_return_value(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type="write", autonomica=gov)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(3, 4) == 7

    def test_sync_passes_kwargs(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type="read", autonomica=gov)
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}"

        assert greet(name="World", greeting="Hi") == "Hi, World"

    def test_sync_preserves_function_metadata(self):
        gov = _open_gov()

        @govern(agent_id="bot", autonomica=gov)
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_sync_blocked_raises_governance_blocked(self):
        gov = _blocking_gov()

        @govern(agent_id="bot", action_type="financial", autonomica=gov)
        def process_payment(amount: float) -> str:
            return "paid"

        with pytest.raises(GovernanceBlocked) as exc_info:
            process_payment(1000.0)

        err = exc_info.value
        assert err.function_name == "process_payment"
        assert err.decision is not None
        assert not err.decision.approved

    def test_governance_blocked_has_decision_details(self):
        gov = _blocking_gov()

        @govern(agent_id="bot", action_type="delete", autonomica=gov)
        def delete_records():
            return "deleted"

        with pytest.raises(GovernanceBlocked) as exc_info:
            delete_records()

        decision = exc_info.value.decision
        assert decision.action_id
        assert decision.risk_score is not None

    def test_sync_infers_action_type_from_name(self):
        """If action_type is omitted, it should be inferred correctly."""
        gov = _open_gov()
        captured: list[ActionType] = []

        @govern(agent_id="bot", autonomica=gov)
        def fetch_data(key: str) -> str:
            return key

        # Patch the scorer to capture what action type was used
        original_evaluate = gov.evaluate_action_sync

        def capturing_evaluate(action: AgentAction):
            captured.append(action.action_type)
            return original_evaluate(action)

        gov.evaluate_action_sync = capturing_evaluate
        fetch_data("test")

        assert captured[0] == ActionType.READ

    def test_sync_explicit_action_type_overrides_inference(self):
        gov = _open_gov()
        captured: list[ActionType] = []

        @govern(agent_id="bot", action_type="financial", autonomica=gov)
        def get_balance() -> float:  # "get" would infer READ
            return 100.0

        original = gov.evaluate_action_sync
        def cap(action):
            captured.append(action.action_type)
            return original(action)
        gov.evaluate_action_sync = cap

        get_balance()
        assert captured[0] == ActionType.FINANCIAL


# ---------------------------------------------------------------------------
# Async functions
# ---------------------------------------------------------------------------

class TestAsyncDecorator:
    async def test_basic_async_function_executes(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type="read", autonomica=gov)
        async def search(query: str) -> str:
            return f"results:{query}"

        result = await search("hello")
        assert result == "results:hello"

    async def test_async_preserves_return_value(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type="write", autonomica=gov)
        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        assert await add(5, 6) == 11

    async def test_async_preserves_function_metadata(self):
        gov = _open_gov()

        @govern(agent_id="bot", autonomica=gov)
        async def my_async_fn():
            """Async docstring."""

        assert my_async_fn.__name__ == "my_async_fn"
        assert my_async_fn.__doc__ == "Async docstring."

    async def test_async_blocked_raises_governance_blocked(self):
        gov = _blocking_gov()

        @govern(agent_id="bot", action_type="financial", autonomica=gov)
        async def wire_transfer(amount: float) -> str:
            return "wired"

        with pytest.raises(GovernanceBlocked) as exc_info:
            await wire_transfer(50_000.0)

        err = exc_info.value
        assert err.function_name == "wire_transfer"
        assert not err.decision.approved

    async def test_async_infers_action_type_from_name(self):
        gov = _open_gov()
        captured: list[ActionType] = []

        @govern(agent_id="bot", autonomica=gov)
        async def delete_user(user_id: str) -> None:
            pass

        original = gov.evaluate_action
        async def cap(action):
            captured.append(action.action_type)
            return await original(action)
        gov.evaluate_action = cap

        await delete_user("u-1")
        assert captured[0] == ActionType.DELETE


# ---------------------------------------------------------------------------
# Custom Autonomica instance
# ---------------------------------------------------------------------------

class TestCustomInstance:
    def test_custom_instance_is_used(self):
        """Decorator should use the passed autonomica instance, not the default."""
        gov = _open_gov()
        call_log: list[str] = []

        original = gov.evaluate_action_sync
        def logged(action):
            call_log.append(action.agent_id)
            return original(action)
        gov.evaluate_action_sync = logged

        @govern(agent_id="custom-agent", autonomica=gov)
        def noop() -> None:
            pass

        noop()
        assert "custom-agent" in call_log

    async def test_custom_instance_used_in_async(self):
        gov = _open_gov()
        call_log: list[str] = []

        original = gov.evaluate_action
        async def logged(action):
            call_log.append(action.agent_id)
            return await original(action)
        gov.evaluate_action = logged

        @govern(agent_id="async-custom", autonomica=gov)
        async def noop() -> None:
            pass

        await noop()
        assert "async-custom" in call_log

    def test_two_decorators_use_separate_instances(self):
        gov_a = _open_gov()
        gov_b = _open_gov()
        calls_a: list = []
        calls_b: list = []

        orig_a = gov_a.evaluate_action_sync
        orig_b = gov_b.evaluate_action_sync
        gov_a.evaluate_action_sync = lambda a: (calls_a.append(1), orig_a(a))[1]
        gov_b.evaluate_action_sync = lambda a: (calls_b.append(1), orig_b(a))[1]

        @govern(agent_id="a", autonomica=gov_a)
        def fn_a():
            pass

        @govern(agent_id="b", autonomica=gov_b)
        def fn_b():
            pass

        fn_a()
        fn_b()
        fn_b()

        assert len(calls_a) == 1
        assert len(calls_b) == 2


# ---------------------------------------------------------------------------
# Tool input capture
# ---------------------------------------------------------------------------

class TestToolInputCapture:
    def test_positional_args_captured(self):
        gov = _open_gov()
        captured: list[dict] = []

        original = gov.evaluate_action_sync
        def cap(action):
            captured.append(dict(action.tool_input))
            return original(action)
        gov.evaluate_action_sync = cap

        @govern(agent_id="bot", autonomica=gov)
        def fn(amount: float, recipient: str) -> None:
            pass

        fn(99.9, "alice")
        assert captured[0]["amount"] == 99.9
        assert captured[0]["recipient"] == "alice"

    def test_keyword_args_captured(self):
        gov = _open_gov()
        captured: list[dict] = []

        original = gov.evaluate_action_sync
        def cap(action):
            captured.append(dict(action.tool_input))
            return original(action)
        gov.evaluate_action_sync = cap

        @govern(agent_id="bot", autonomica=gov)
        def fn(x: int, y: int = 10) -> None:
            pass

        fn(x=5)
        assert captured[0]["x"] == 5
        assert captured[0]["y"] == 10  # default captured


# ---------------------------------------------------------------------------
# Invalid action_type string
# ---------------------------------------------------------------------------

class TestInvalidActionType:
    def test_unknown_action_type_raises_at_decoration_time(self):
        with pytest.raises(ValueError, match="Unknown action_type"):
            @govern(agent_id="bot", action_type="explode")
            def fn():
                pass

    def test_enum_value_accepted_directly(self):
        gov = _open_gov()

        @govern(agent_id="bot", action_type=ActionType.FINANCIAL, autonomica=gov)
        def fn() -> str:
            return "ok"

        assert fn() == "ok"
