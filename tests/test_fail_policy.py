"""
Tests for the fail-policy feature (§ interceptor — error handling).

Three policies:
  "open"     — always proceed (LOG_AND_ALERT, approved=True)
  "closed"   — always block  (QUARANTINE,    approved=False)
  "adaptive" — READ → fail open; WRITE/DELETE/COMMUNICATE/FINANCIAL → HARD_GATE

All tests simulate a pipeline crash by replacing scorer.score() with a function
that raises RuntimeError (same net effect as a storage crash mid-pipeline).
"""
from __future__ import annotations

import asyncio

import pytest

from autonomica import Autonomica, AutonomicaConfig
from autonomica.models import ActionType, AgentAction, GovernanceMode


# ── Helpers ────────────────────────────────────────────────────────────────────

def _action(action_type: ActionType, agent_id: str = "test-agent") -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        agent_name="Test Agent",
        tool_name="test_tool",
        tool_input={"key": "value"},
        action_type=action_type,
    )


def _crashing_scorer(exc_msg: str = "scorer crashed"):
    """Return a callable that raises RuntimeError — drop-in for scorer.score."""
    def _raise(action, profile):
        raise RuntimeError(exc_msg)
    return _raise


def _gov(fail_policy: str, *, hard_gate_timeout: float = 0.02) -> Autonomica:
    """Autonomica with a crashing scorer and short timeouts for fast tests."""
    config = AutonomicaConfig(
        fail_policy=fail_policy,
        soft_gate_timeout_seconds=0.02,
        hard_gate_timeout_seconds=hard_gate_timeout,
    )
    gov = Autonomica(config=config)
    gov.scorer.score = _crashing_scorer()
    return gov


# ── Fail open ──────────────────────────────────────────────────────────────────

class TestFailOpen:
    """fail_policy='open': every action type proceeds regardless of error."""

    @pytest.mark.parametrize("action_type", list(ActionType))
    async def test_always_approved(self, action_type):
        gov = _gov("open")
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.approved is True, f"Expected approved for {action_type}"

    @pytest.mark.parametrize("action_type", list(ActionType))
    async def test_mode_is_log_and_alert(self, action_type):
        gov = _gov("open")
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.mode == GovernanceMode.LOG_AND_ALERT

    async def test_pipeline_error_field_populated(self):
        gov = _gov("open")
        decision = await gov.evaluate_action(_action(ActionType.FINANCIAL))
        assert decision.pipeline_error is not None
        assert "scorer crashed" in decision.pipeline_error

    async def test_decision_stored_for_override(self):
        """Error-path decisions must be retrievable for feedback calls."""
        gov = _gov("open")
        action = _action(ActionType.WRITE)
        decision = await gov.evaluate_action(action)
        assert gov.get_decision(action.action_id) == decision

    async def test_profile_counters_incremented(self):
        gov = _gov("open")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        profile = gov.get_agent_profile("test-agent")
        assert profile.total_actions == 1
        assert profile.approved_actions == 1


# ── Fail closed ────────────────────────────────────────────────────────────────

class TestFailClosed:
    """fail_policy='closed': every action type is blocked regardless of error."""

    @pytest.mark.parametrize("action_type", list(ActionType))
    async def test_always_blocked(self, action_type):
        gov = _gov("closed")
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.approved is False, f"Expected blocked for {action_type}"

    @pytest.mark.parametrize("action_type", list(ActionType))
    async def test_mode_is_quarantine(self, action_type):
        gov = _gov("closed")
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.mode == GovernanceMode.QUARANTINE

    async def test_pipeline_error_field_populated(self):
        gov = _gov("closed")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.pipeline_error is not None
        assert "scorer crashed" in decision.pipeline_error

    async def test_decision_stored(self):
        gov = _gov("closed")
        action = _action(ActionType.DELETE)
        decision = await gov.evaluate_action(action)
        assert gov.get_decision(action.action_id) == decision

    async def test_profile_counters_incremented(self):
        gov = _gov("closed")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        profile = gov.get_agent_profile("test-agent")
        assert profile.total_actions == 1
        assert profile.approved_actions == 0  # blocked


# ── Fail adaptive (default) ────────────────────────────────────────────────────

class TestFailAdaptive:
    """fail_policy='adaptive': READ → open, everything else → HARD_GATE."""

    # ── READ actions: fail open ────────────────────────────────────────────────

    async def test_read_proceeds_on_error(self):
        gov = _gov("adaptive")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.approved is True

    async def test_read_mode_is_log_and_alert(self):
        gov = _gov("adaptive")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.mode == GovernanceMode.LOG_AND_ALERT

    # ── Non-READ actions: HARD_GATE ────────────────────────────────────────────

    @pytest.mark.parametrize("action_type", [
        ActionType.WRITE,
        ActionType.DELETE,
        ActionType.COMMUNICATE,
        ActionType.FINANCIAL,
    ])
    async def test_non_read_mode_is_hard_gate(self, action_type):
        gov = _gov("adaptive")
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.mode == GovernanceMode.HARD_GATE

    @pytest.mark.parametrize("action_type", [
        ActionType.WRITE,
        ActionType.DELETE,
        ActionType.COMMUNICATE,
        ActionType.FINANCIAL,
    ])
    async def test_non_read_blocked_on_timeout(self, action_type):
        """HARD_GATE with no human response → approved=False (block on timeout)."""
        gov = _gov("adaptive", hard_gate_timeout=0.02)
        decision = await gov.evaluate_action(_action(action_type))
        assert decision.approved is False

    async def test_financial_pipeline_error_recorded(self):
        gov = _gov("adaptive")
        decision = await gov.evaluate_action(_action(ActionType.FINANCIAL))
        assert decision.pipeline_error is not None
        assert "scorer crashed" in decision.pipeline_error

    async def test_write_approved_when_human_overrides(self):
        """Human approves a HARD_GATE during the error-path gate — must unblock."""
        config = AutonomicaConfig(
            fail_policy="adaptive",
            hard_gate_timeout_seconds=5.0,  # plenty of time for the test override
        )
        gov = Autonomica(config=config)
        gov.scorer.score = _crashing_scorer()
        action = _action(ActionType.WRITE)

        # Start evaluation as a background task so we can race the override.
        task = asyncio.create_task(gov.evaluate_action(action))

        # Yield to the event loop so the error handler registers the pending
        # Future and starts waiting at the HARD_GATE.
        await asyncio.sleep(0.05)

        # Human approves — should resolve the gate immediately.
        gov.record_human_override(action.action_id, approved=True)

        decision = await task
        assert decision.approved is True
        assert decision.mode == GovernanceMode.HARD_GATE

    async def test_delete_rejected_when_human_vetoes(self):
        """Human rejects a HARD_GATE during the error-path gate — stays blocked."""
        config = AutonomicaConfig(
            fail_policy="adaptive",
            hard_gate_timeout_seconds=5.0,
        )
        gov = Autonomica(config=config)
        gov.scorer.score = _crashing_scorer()
        action = _action(ActionType.DELETE)

        task = asyncio.create_task(gov.evaluate_action(action))
        await asyncio.sleep(0.05)
        gov.record_human_override(action.action_id, approved=False)

        decision = await task
        assert decision.approved is False
        assert decision.mode == GovernanceMode.HARD_GATE

    async def test_default_policy_is_adaptive(self):
        """Autonomica() with no config must default to adaptive."""
        gov = Autonomica()  # no config → defaults
        gov.scorer.score = _crashing_scorer()
        # READ → should proceed (adaptive = fail-open for reads)
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.approved is True
        assert decision.mode == GovernanceMode.LOG_AND_ALERT

    async def test_profile_read_counters_correct(self):
        gov = _gov("adaptive")
        await gov.evaluate_action(_action(ActionType.READ))
        profile = gov.get_agent_profile("test-agent")
        assert profile.total_actions == 1
        assert profile.approved_actions == 1
        assert profile.escalated_actions == 0  # LOG_AND_ALERT is not escalated

    async def test_profile_write_counters_correct(self):
        gov = _gov("adaptive")
        await gov.evaluate_action(_action(ActionType.WRITE))
        profile = gov.get_agent_profile("test-agent")
        assert profile.total_actions == 1
        assert profile.approved_actions == 0   # blocked
        assert profile.escalated_actions == 1  # HARD_GATE counts as escalated


# ── Storage-crash simulation ───────────────────────────────────────────────────

class TestStorageCrash:
    """
    Simulate a crash during evaluate_action by injecting a failure into the
    scoring step — same observable effect as a storage crash mid-pipeline.

    All three policies are exercised; action types READ and WRITE are used as
    representative members of the 'fail-open' and 'fail-closed' branches.
    """

    async def test_open_policy_read_proceeds(self):
        gov = Autonomica(config=AutonomicaConfig(fail_policy="open"))
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.approved is True
        assert "storage crashed" in (decision.pipeline_error or "")

    async def test_open_policy_write_proceeds(self):
        gov = Autonomica(config=AutonomicaConfig(fail_policy="open"))
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.WRITE))
        assert decision.approved is True

    async def test_closed_policy_read_blocked(self):
        gov = Autonomica(config=AutonomicaConfig(fail_policy="closed"))
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.approved is False
        assert decision.mode == GovernanceMode.QUARANTINE

    async def test_closed_policy_financial_blocked(self):
        gov = Autonomica(config=AutonomicaConfig(fail_policy="closed"))
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.FINANCIAL))
        assert decision.approved is False
        assert decision.mode == GovernanceMode.QUARANTINE

    async def test_adaptive_read_proceeds_despite_crash(self):
        config = AutonomicaConfig(fail_policy="adaptive")
        gov = Autonomica(config=config)
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.READ))
        assert decision.approved is True
        assert decision.pipeline_error is not None
        assert "storage crashed" in decision.pipeline_error

    async def test_adaptive_write_hard_gates_despite_crash(self):
        config = AutonomicaConfig(
            fail_policy="adaptive",
            hard_gate_timeout_seconds=0.02,
        )
        gov = Autonomica(config=config)
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.WRITE))
        assert decision.approved is False
        assert decision.mode == GovernanceMode.HARD_GATE

    async def test_adaptive_financial_hard_gates_despite_crash(self):
        config = AutonomicaConfig(
            fail_policy="adaptive",
            hard_gate_timeout_seconds=0.02,
        )
        gov = Autonomica(config=config)
        gov.scorer.score = _crashing_scorer("storage crashed")
        decision = await gov.evaluate_action(_action(ActionType.FINANCIAL))
        assert decision.approved is False
        assert decision.mode == GovernanceMode.HARD_GATE

    async def test_decision_still_stored_after_crash(self):
        """Even a crashed pipeline must leave a retrievable decision."""
        gov = Autonomica(config=AutonomicaConfig(fail_policy="open"))
        gov.scorer.score = _crashing_scorer("storage crashed")
        action = _action(ActionType.READ)
        decision = await gov.evaluate_action(action)
        assert gov.get_decision(action.action_id) is not None
        assert gov.get_decision(action.action_id).pipeline_error is not None

    async def test_multiple_agents_isolated(self):
        """A crash for one agent must not corrupt another agent's profile."""
        config = AutonomicaConfig(fail_policy="adaptive")
        gov = Autonomica(config=config)

        # Agent A: crashes
        gov.scorer.score = _crashing_scorer()
        await gov.evaluate_action(_action(ActionType.READ, agent_id="agent-a"))

        # Agent B: normal scorer
        from autonomica.scorer import RiskScorer
        gov.scorer = RiskScorer()
        await gov.evaluate_action(_action(ActionType.READ, agent_id="agent-b"))

        profile_a = gov.get_agent_profile("agent-a")
        profile_b = gov.get_agent_profile("agent-b")
        assert profile_a is not None
        assert profile_b is not None
        assert profile_a.agent_id != profile_b.agent_id


# ── Config validation ──────────────────────────────────────────────────────────

class TestFailPolicyConfig:
    def test_default_is_adaptive(self):
        config = AutonomicaConfig()
        assert config.fail_policy == "adaptive"

    def test_open_accepted(self):
        config = AutonomicaConfig(fail_policy="open")
        assert config.fail_policy == "open"

    def test_closed_accepted(self):
        config = AutonomicaConfig(fail_policy="closed")
        assert config.fail_policy == "closed"

    def test_invalid_policy_rejected(self):
        import pytest
        with pytest.raises(Exception):  # pydantic ValidationError
            AutonomicaConfig(fail_policy="maybe")

    def test_autonomica_reads_policy_from_config(self):
        gov = Autonomica(config=AutonomicaConfig(fail_policy="open"))
        assert gov._fail_policy == "open"

    def test_autonomica_defaults_to_adaptive_without_config(self):
        gov = Autonomica()
        assert gov._fail_policy == "adaptive"
