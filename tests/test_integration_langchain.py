"""Integration tests — LangChain tool wrapping and full governance pipeline.

Covers:
- Low-risk read passes through automatically (FULL_AUTO)
- First-call read by new agent gets LOG_AND_ALERT but still proceeds
- Financial action gets escalated (SOFT_GATE)
- High-risk action hits QUARANTINE and is blocked
- Action type inference for all keyword categories
- wrap_langchain_tools convenience function
- Decision and profile logging
- record_outcome and record_human_override
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from autonomica import Autonomica, GovernanceMode
from autonomica.escalation.base import BaseEscalation
from autonomica.integrations.langchain import GovernedTool, wrap_langchain_tools
from autonomica.models import ActionType, AgentAction, AgentProfile

# ---------------------------------------------------------------------------
# Mock LangChain tools
# ---------------------------------------------------------------------------

class _MockBaseTool(BaseTool):
    """Helper that accepts arbitrary kwargs so run_manager forwarding is safe."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    call_count: int = 0

    def _run(self, *args: Any, **kwargs: Any) -> str:
        self.call_count += 1
        return f"{self.name}:ok"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        self.call_count += 1
        return f"{self.name}:ok:async"


def make_tool(name: str, description: str = "") -> _MockBaseTool:
    return _MockBaseTool(name=name, description=description or f"Tool {name}")


# ---------------------------------------------------------------------------
# Mock escalation (records notifications, returns pre-set response instantly)
# ---------------------------------------------------------------------------

class MockEscalation(BaseEscalation):
    def __init__(self, response: Optional[bool] = None) -> None:
        self.notifications: list[tuple[AgentAction, GovernanceMode]] = []
        self._response = response

    async def notify(self, action: AgentAction, mode: GovernanceMode) -> None:
        self.notifications.append((action, mode))

    async def wait_for_response(self, action_id: str, timeout: float) -> Optional[bool]:
        return self._response


class FastConfig:
    """Near-zero gate timeouts so tests don't sleep."""
    soft_gate_timeout_seconds = 0.02
    hard_gate_timeout_seconds = 0.02


def make_gov(escalation: BaseEscalation | None = None) -> Autonomica:
    return Autonomica(config=FastConfig(), escalation=escalation or MockEscalation())


# ---------------------------------------------------------------------------
# Helpers for seeding agent profiles
# ---------------------------------------------------------------------------

def seed_trusted_profile(gov: Autonomica, agent_id: str = "agent") -> AgentProfile:
    """Trust=90, 20 uses of every tool → FULL_AUTO territory for read actions."""
    profile = gov._get_or_create_profile(agent_id, agent_id)
    profile.trust_score = 90.0
    profile.per_tool_trust = {
        "read_database": 20,
        "search": 20,
        "get_user": 20,
    }
    return profile


# ===========================================================================
# Scenario 1 — trusted agent, familiar read tool → FULL_AUTO (passes through)
# ===========================================================================

class TestLowRiskReadPassThrough:
    """Trusted agent + familiar tool → score 4.5 → FULL_AUTO → action executes."""

    async def test_arun_passes_through(self):
        esc = MockEscalation()
        gov = make_gov(esc)
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database", "Read data from the database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        result = await governed._arun(query="SELECT 1")

        assert result == "read_database:ok:async"
        assert tool.call_count == 1

    async def test_decision_is_full_auto(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun(query="SELECT 1")

        # Last recorded decision
        decisions = list(gov._decisions.values())
        assert len(decisions) == 1
        assert decisions[0].mode == GovernanceMode.FULL_AUTO
        assert decisions[0].approved is True

    async def test_profile_counters_updated(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun()

        profile = gov.get_agent_profile("agent")
        assert profile.total_actions == 1
        assert profile.approved_actions == 1
        assert profile.escalated_actions == 0

    def test_sync_run_passes_through(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        result = governed._run(query="SELECT 1")

        assert result == "read_database:ok"
        assert tool.call_count == 1


# ===========================================================================
# Scenario 2 — new agent, first-time read → LOG_AND_ALERT (still passes)
# ===========================================================================

class TestNewAgentReadLogAndAlert:
    """New default agent + first tool use → score 16.5 → LOG_AND_ALERT → proceeds."""

    async def test_proceeds_despite_escalation(self):
        gov = make_gov()
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="fresh-agent")[0]

        result = await governed._arun(query="SELECT 1")

        # Tool still executed
        assert result == "read_database:ok:async"

    async def test_mode_is_log_and_alert(self):
        gov = make_gov()
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="fresh-agent")[0]

        await governed._arun()

        decision = list(gov._decisions.values())[0]
        assert decision.mode == GovernanceMode.LOG_AND_ALERT
        assert decision.approved is True

    async def test_escalated_actions_not_incremented_for_log_and_alert(self):
        """LOG_AND_ALERT is not counted as an 'escalation' — it's just alerting."""
        gov = make_gov()
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="fresh-agent")[0]

        await governed._arun()

        profile = gov.get_agent_profile("fresh-agent")
        # LOG_AND_ALERT < SOFT_GATE, so escalated_actions stays 0
        assert profile.escalated_actions == 0


# ===========================================================================
# Scenario 3 — large financial action → SOFT_GATE (proceeds on timeout)
# ===========================================================================

class TestHighRiskFinancialEscalation:
    """New agent + $500K payment → score 50.5 → SOFT_GATE → proceeds on timeout."""

    async def test_mode_is_soft_gate(self):
        esc = MockEscalation(response=None)  # simulate timeout → proceed
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment or charge")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance-agent")[0]

        await governed._arun(amount=500_000.0, recipient="vendor@corp.com")

        decision = list(gov._decisions.values())[0]
        assert decision.mode == GovernanceMode.SOFT_GATE

    async def test_soft_gate_timeout_proceeds(self):
        esc = MockEscalation(response=None)
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance-agent")[0]

        result = await governed._arun(amount=500_000.0)

        # Soft gate: timeout → proceed
        assert result == "process_payment:ok:async"
        assert tool.call_count == 1

    async def test_soft_gate_veto_blocks(self):
        esc = MockEscalation(response=False)  # human vetoes
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance-agent")[0]

        result = await governed._arun(amount=500_000.0)

        assert "[AUTONOMICA] Action blocked" in result
        assert tool.call_count == 0

    async def test_escalated_actions_incremented(self):
        esc = MockEscalation(response=None)
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance-agent")[0]

        await governed._arun(amount=500_000.0)

        profile = gov.get_agent_profile("finance-agent")
        assert profile.escalated_actions == 1

    async def test_notification_sent(self):
        esc = MockEscalation(response=None)
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance-agent")[0]

        await governed._arun(amount=500_000.0)

        assert len(esc.notifications) == 1
        _, notified_mode = esc.notifications[0]
        assert notified_mode == GovernanceMode.SOFT_GATE


# ===========================================================================
# Scenario 4 — quarantine: zero-trust agent, delete, health data, many downstream
# ===========================================================================

class TestQuarantineBlocked:
    """score 87.5 → QUARANTINE → blocked immediately, tool never called."""

    def _make_quarantine_gov(self) -> Autonomica:
        """
        Build a gov instance where our worst-case action (score 79.5) lands in
        QUARANTINE.  We lower hard_gate_max to 75 so anything above 75 is
        quarantined — exercising the per-agent adaptive threshold feature.

        Worst-case inputs (trust=10, DELETE, health data, $2M):
          financial=100, sensitivity=80, reversibility=80,
          track_record=90, novelty=70, cascade=20  →  composite = 79.5
        With hard_gate_max=75 that puts 79.5 into QUARANTINE.
        """
        esc = MockEscalation()
        gov = make_gov(esc)
        profile = gov._get_or_create_profile("untrusted", "untrusted")
        profile.trust_score = 10.0
        profile.mode_thresholds["hard_gate_max"] = 75.0  # tightened threshold
        return gov

    async def test_blocked_returns_autonomica_message(self):
        gov = self._make_quarantine_gov()
        tool = make_tool("delete_records", "Delete and drop database records")
        governed = wrap_langchain_tools([tool], gov, agent_id="untrusted")[0]

        result = await governed._arun(
            table="patients",
            diagnosis="cancer",
            amount=2_000_000.0,
        )

        assert "[AUTONOMICA] Action blocked" in result
        assert "QUARANTINE" in result

    async def test_original_tool_never_called(self):
        gov = self._make_quarantine_gov()
        tool = make_tool("delete_records", "Delete and drop database records")
        governed = wrap_langchain_tools([tool], gov, agent_id="untrusted")[0]

        await governed._arun(
            table="patients",
            diagnosis="cancer",
            amount=2_000_000.0,
        )

        assert tool.call_count == 0

    async def test_decision_approved_false(self):
        gov = self._make_quarantine_gov()
        tool = make_tool("delete_records", "Delete and drop database records")
        governed = wrap_langchain_tools([tool], gov, agent_id="untrusted")[0]

        await governed._arun(table="patients", diagnosis="cancer", amount=2_000_000.0)

        decision = list(gov._decisions.values())[0]
        assert decision.mode == GovernanceMode.QUARANTINE
        assert decision.approved is False


# ===========================================================================
# Action type inference
# ===========================================================================

class TestActionTypeInference:
    """_infer_action_type() classifies tools by name/description keywords."""

    def _make_governed(self, name: str, description: str = "") -> GovernedTool:
        gov = make_gov()
        tool = make_tool(name, description)
        return wrap_langchain_tools([tool], gov, agent_id="a")[0]

    @pytest.mark.parametrize("name,desc,expected", [
        # DELETE keywords
        ("delete_user",     "",                         ActionType.DELETE),
        ("remove_file",     "",                         ActionType.DELETE),
        ("drop_table",      "",                         ActionType.DELETE),
        ("cleanup",         "drop all inactive records",ActionType.DELETE),
        # COMMUNICATE keywords
        ("send_email",      "",                         ActionType.COMMUNICATE),
        ("notify_slack",    "",                         ActionType.COMMUNICATE),
        ("post_message",    "",                         ActionType.COMMUNICATE),
        ("alert",           "send a message to user",  ActionType.COMMUNICATE),
        # FINANCIAL keywords
        ("pay_invoice",     "",                         ActionType.FINANCIAL),
        ("wire_transfer",   "",                         ActionType.FINANCIAL),
        ("process_charge",  "",                         ActionType.FINANCIAL),
        ("billing",         "invoice the customer",    ActionType.FINANCIAL),
        # WRITE keywords
        ("write_record",    "",                         ActionType.WRITE),
        ("update_user",     "",                         ActionType.WRITE),
        ("create_order",    "",                         ActionType.WRITE),
        ("insert_row",      "",                         ActionType.WRITE),
        ("set_config",      "",                         ActionType.WRITE),
        # READ fallback
        ("get_user",        "",                         ActionType.READ),
        ("search_products", "",                         ActionType.READ),
        ("fetch_report",    "",                         ActionType.READ),
        ("query_db",        "",                         ActionType.READ),
    ])
    def test_infer(self, name, desc, expected):
        governed = self._make_governed(name, desc)
        assert governed._infer_action_type() == expected

    def test_delete_beats_communicate(self):
        """delete keyword takes priority over send."""
        governed = self._make_governed("delete_and_send_notification")
        assert governed._infer_action_type() == ActionType.DELETE

    def test_communicate_beats_financial_in_desc(self):
        """First matching priority wins: DELETE > COMMUNICATE > FINANCIAL > WRITE > READ."""
        governed = self._make_governed("notify_payment", "send invoice charge")
        # 'send' matches COMMUNICATE before 'invoice' matches FINANCIAL
        assert governed._infer_action_type() == ActionType.COMMUNICATE


# ===========================================================================
# wrap_langchain_tools convenience function
# ===========================================================================

class TestWrapLangchainTools:
    def test_returns_governed_tools(self):
        gov = make_gov()
        tools = [make_tool("tool_a"), make_tool("tool_b"), make_tool("tool_c")]
        wrapped = wrap_langchain_tools(tools, gov, agent_id="agent-1")
        assert len(wrapped) == 3
        assert all(isinstance(t, GovernedTool) for t in wrapped)

    def test_retains_name_and_description(self):
        gov = make_gov()
        tool = make_tool("my_tool", "Does something useful")
        (wrapped,) = wrap_langchain_tools([tool], gov, agent_id="a")
        assert wrapped.name == "my_tool"
        assert wrapped.description == "Does something useful"

    def test_agent_id_set_correctly(self):
        gov = make_gov()
        tools = [make_tool("t1"), make_tool("t2")]
        wrapped = wrap_langchain_tools(tools, gov, agent_id="invoice-agent")
        assert all(t.agent_id == "invoice-agent" for t in wrapped)

    def test_original_tool_preserved(self):
        gov = make_gov()
        original = make_tool("my_tool")
        (wrapped,) = wrap_langchain_tools([original], gov, agent_id="a")
        assert wrapped.original_tool is original

    def test_autonomica_instance_preserved(self):
        gov = make_gov()
        (wrapped,) = wrap_langchain_tools([make_tool("t")], gov, agent_id="a")
        assert wrapped.autonomica is gov

    def test_three_line_usage_compiles(self):
        """The 3-line integration from the spec must work end-to-end."""
        from autonomica import Autonomica
        from autonomica.integrations.langchain import wrap_langchain_tools as wlt

        tools = [make_tool("read_data"), make_tool("send_email")]
        gov = Autonomica()
        governed = wlt(tools, gov, agent_id="invoice-agent")

        assert len(governed) == 2
        assert all(isinstance(t, GovernedTool) for t in governed)


# ===========================================================================
# wrap_tool / wrap_tools on Autonomica instance
# ===========================================================================

class TestAutonomicaWrapMethods:
    def test_wrap_tool_returns_governed_tool(self):
        gov = make_gov()
        tool = make_tool("search")
        wrapped = gov.wrap_tool(tool, agent_id="a")
        assert isinstance(wrapped, GovernedTool)
        assert wrapped.agent_id == "a"

    def test_wrap_tools_wraps_all(self):
        gov = make_gov()
        tools = [make_tool(f"t{i}") for i in range(5)]
        wrapped = gov.wrap_tools(tools, agent_id="a")
        assert len(wrapped) == 5


# ===========================================================================
# record_outcome
# ===========================================================================

class TestRecordOutcome:
    async def test_success_does_not_increment_incidents(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun()
        action_id = list(gov._decisions.keys())[0]

        gov.record_outcome(action_id, success=True)
        assert gov.get_agent_profile("agent").incidents == 0

    async def test_failure_increments_incidents(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun()
        action_id = list(gov._decisions.keys())[0]

        gov.record_outcome(action_id, success=False, notes="DB timeout")
        assert gov.get_agent_profile("agent").incidents == 1

    def test_unknown_action_id_is_safe(self):
        gov = make_gov()
        gov.record_outcome("nonexistent-id", success=False)  # must not raise


# ===========================================================================
# record_human_override
# ===========================================================================

class TestRecordHumanOverride:
    async def test_override_increments_false_escalations_on_approve(self):
        esc = MockEscalation(response=None)
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance")[0]

        await governed._arun(amount=500_000.0)

        action_id = list(gov._decisions.keys())[0]
        gov.record_human_override(action_id, approved=True, reason="CFO approved")

        profile = gov.get_agent_profile("finance")
        assert profile.false_escalations == 1

    async def test_override_rejection_does_not_increment_false_escalations(self):
        esc = MockEscalation(response=None)
        gov = make_gov(esc)
        tool = make_tool("process_payment", "Process a payment")
        governed = wrap_langchain_tools([tool], gov, agent_id="finance")[0]

        await governed._arun(amount=500_000.0)

        action_id = list(gov._decisions.keys())[0]
        gov.record_human_override(action_id, approved=False, reason="Suspicious")

        profile = gov.get_agent_profile("finance")
        assert profile.false_escalations == 0

    def test_override_on_unknown_action_is_safe(self):
        gov = make_gov()
        gov.record_human_override("ghost-id", approved=True)  # must not raise


# ===========================================================================
# Decision logging
# ===========================================================================

class TestDecisionLogging:
    async def test_decision_stored_and_retrievable(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun(query="SELECT id FROM users")

        action_id = list(gov._decisions.keys())[0]
        decision = gov.get_decision(action_id)
        assert decision is not None
        assert decision.action_id == action_id

    async def test_decision_time_ms_recorded(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun()

        decision = list(gov._decisions.values())[0]
        assert decision.decision_time_ms >= 0.0

    async def test_risk_score_in_decision(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun()

        decision = list(gov._decisions.values())[0]
        assert decision.risk_score.composite_score == pytest.approx(4.5)

    async def test_multiple_actions_all_logged(self):
        gov = make_gov()
        seed_trusted_profile(gov, "agent")
        tool = make_tool("read_database")
        governed = wrap_langchain_tools([tool], gov, agent_id="agent")[0]

        await governed._arun(query="q1")
        await governed._arun(query="q2")
        await governed._arun(query="q3")

        assert len(gov._decisions) == 3
