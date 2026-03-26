"""Tests for autonomica/governor.py — GovernanceEngine."""
from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from autonomica.escalation.base import BaseEscalation
from autonomica.governor import GovernanceEngine
from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceMode,
    RiskScore,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class RecordingEscalation(BaseEscalation):
    """Captures notifications. wait_for_response returns a pre-set answer."""

    def __init__(
        self,
        response: Optional[bool] = None,   # None = simulate timeout
        response_delay: float = 0.0,        # seconds before responding
    ) -> None:
        self.notifications: list[tuple[AgentAction, GovernanceMode]] = []
        self._response = response
        self._response_delay = response_delay

    async def notify(self, action: AgentAction, mode: GovernanceMode, risk_score=None) -> None:
        self.notifications.append((action, mode))

    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        if self._response_delay > 0:
            await asyncio.sleep(self._response_delay)
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_profile(
    trust_score: float = 50.0,
    thresholds: dict | None = None,
) -> AgentProfile:
    p = AgentProfile(agent_id="test-agent", agent_name="Test Agent", trust_score=trust_score)
    if thresholds:
        p.mode_thresholds = thresholds
    return p


def make_action(action_type: ActionType = ActionType.READ) -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="test_tool",
        tool_input={},
        action_type=action_type,
    )


def make_risk_score(composite: float) -> RiskScore:
    return RiskScore(
        composite_score=composite,
        financial_magnitude=0.0,
        data_sensitivity=0.0,
        reversibility=0.0,
        agent_track_record=0.0,
        novelty=0.0,
        cascade_risk=0.0,
        explanation="test",
    )


@pytest.fixture
def engine() -> GovernanceEngine:
    return GovernanceEngine()


@pytest.fixture
def fast_engine() -> GovernanceEngine:
    """Engine with near-zero timeouts so gate tests don't wait."""
    class FastConfig:
        soft_gate_timeout_seconds = 0.05
        hard_gate_timeout_seconds = 0.05
    return GovernanceEngine(FastConfig())


# ---------------------------------------------------------------------------
# decide() — threshold mapping
# ---------------------------------------------------------------------------

class TestDecide:
    """decide() maps composite scores to modes using agent thresholds."""

    # Default thresholds: full_auto≤15, log_alert≤35, soft_gate≤60, hard_gate≤85

    @pytest.mark.parametrize("score,expected_mode", [
        (0.0,  GovernanceMode.FULL_AUTO),
        (7.5,  GovernanceMode.FULL_AUTO),
        (15.0, GovernanceMode.FULL_AUTO),      # exactly at boundary → FULL_AUTO
        (15.1, GovernanceMode.LOG_AND_ALERT),
        (25.0, GovernanceMode.LOG_AND_ALERT),
        (35.0, GovernanceMode.LOG_AND_ALERT),  # exactly at boundary → LOG_AND_ALERT
        (35.1, GovernanceMode.SOFT_GATE),
        (50.0, GovernanceMode.SOFT_GATE),
        (60.0, GovernanceMode.SOFT_GATE),      # exactly at boundary → SOFT_GATE
        (60.1, GovernanceMode.HARD_GATE),
        (75.0, GovernanceMode.HARD_GATE),
        (85.0, GovernanceMode.HARD_GATE),      # exactly at boundary → HARD_GATE
        (85.1, GovernanceMode.QUARANTINE),
        (100.0,GovernanceMode.QUARANTINE),
    ])
    def test_default_threshold_mapping(self, engine, score, expected_mode):
        profile = make_profile()
        risk_score = make_risk_score(score)
        assert engine.decide(risk_score, profile) == expected_mode

    def test_uses_agent_custom_thresholds(self, engine):
        """Adapted agent with shifted thresholds gets different mode for same score."""
        # Agent has earned more autonomy — FULL_AUTO boundary raised to 30
        profile = make_profile(thresholds={
            "full_auto_max": 30.0,
            "log_alert_max": 50.0,
            "soft_gate_max": 70.0,
            "hard_gate_max": 90.0,
        })
        # Score 25 would normally be LOG_AND_ALERT; with widened thresholds → FULL_AUTO
        assert engine.decide(make_risk_score(25.0), profile) == GovernanceMode.FULL_AUTO

    def test_tightened_thresholds_more_restrictive(self, engine):
        """Agent with tightened thresholds gets escalated for a low score."""
        profile = make_profile(thresholds={
            "full_auto_max": 5.0,
            "log_alert_max": 10.0,
            "soft_gate_max": 20.0,
            "hard_gate_max": 30.0,
        })
        # Score 12 would normally be FULL_AUTO; with tight thresholds → SOFT_GATE
        assert engine.decide(make_risk_score(12.0), profile) == GovernanceMode.SOFT_GATE

    def test_returns_governance_mode_type(self, engine):
        profile = make_profile()
        result = engine.decide(make_risk_score(50.0), profile)
        assert isinstance(result, GovernanceMode)

    def test_all_five_modes_reachable(self, engine):
        profile = make_profile()
        scores = [10.0, 25.0, 50.0, 70.0, 95.0]
        modes = {engine.decide(make_risk_score(s), profile) for s in scores}
        assert modes == set(GovernanceMode)


# ---------------------------------------------------------------------------
# enforce() — FULL_AUTO
# ---------------------------------------------------------------------------

class TestEnforceFullAuto:
    async def test_returns_true(self, engine):
        esc = RecordingEscalation()
        action = make_action()
        result = await engine.enforce(GovernanceMode.FULL_AUTO, action, esc)
        assert result is True

    async def test_notification_sent_async(self, engine):
        esc = RecordingEscalation()
        action = make_action()
        await engine.enforce(GovernanceMode.FULL_AUTO, action, esc)
        # create_task doesn't run until we yield to the event loop
        await asyncio.sleep(0)
        assert len(esc.notifications) == 1
        assert esc.notifications[0] == (action, GovernanceMode.FULL_AUTO)

    async def test_does_not_wait_for_response(self, engine):
        """FULL_AUTO should never call wait_for_response."""
        class StrictEscalation(RecordingEscalation):
            async def wait_for_response(self, action_id, timeout):
                raise AssertionError("wait_for_response must not be called for FULL_AUTO")

        esc = StrictEscalation()
        result = await engine.enforce(GovernanceMode.FULL_AUTO, make_action(), esc)
        assert result is True


# ---------------------------------------------------------------------------
# enforce() — LOG_AND_ALERT
# ---------------------------------------------------------------------------

class TestEnforceLogAndAlert:
    async def test_returns_true(self, engine):
        esc = RecordingEscalation()
        result = await engine.enforce(GovernanceMode.LOG_AND_ALERT, make_action(), esc)
        assert result is True

    async def test_notification_sent_async(self, engine):
        esc = RecordingEscalation()
        action = make_action()
        await engine.enforce(GovernanceMode.LOG_AND_ALERT, action, esc)
        await asyncio.sleep(0)
        assert len(esc.notifications) == 1
        assert esc.notifications[0] == (action, GovernanceMode.LOG_AND_ALERT)

    async def test_does_not_wait_for_response(self, engine):
        class StrictEscalation(RecordingEscalation):
            async def wait_for_response(self, action_id, timeout):
                raise AssertionError("wait_for_response must not be called for LOG_AND_ALERT")

        esc = StrictEscalation()
        result = await engine.enforce(GovernanceMode.LOG_AND_ALERT, make_action(), esc)
        assert result is True


# ---------------------------------------------------------------------------
# enforce() — SOFT_GATE
# ---------------------------------------------------------------------------

class TestEnforceSoftGate:
    async def test_timeout_proceeds(self, fast_engine):
        """No response (None / timeout) → action proceeds."""
        esc = RecordingEscalation(response=None)
        result = await fast_engine.enforce(GovernanceMode.SOFT_GATE, make_action(), esc)
        assert result is True

    async def test_explicit_approval_proceeds(self, fast_engine):
        """Human approves (True) → action proceeds."""
        esc = RecordingEscalation(response=True)
        result = await fast_engine.enforce(GovernanceMode.SOFT_GATE, make_action(), esc)
        assert result is True

    async def test_explicit_veto_blocks(self, fast_engine):
        """Human vetoes (False) → action blocked."""
        esc = RecordingEscalation(response=False)
        result = await fast_engine.enforce(GovernanceMode.SOFT_GATE, make_action(), esc)
        assert result is False

    async def test_notification_sent(self, fast_engine):
        esc = RecordingEscalation(response=None)
        action = make_action()
        await fast_engine.enforce(GovernanceMode.SOFT_GATE, action, esc)
        assert len(esc.notifications) == 1
        assert esc.notifications[0] == (action, GovernanceMode.SOFT_GATE)

    async def test_uses_soft_gate_timeout(self):
        """Engine passes its configured soft_gate_timeout to wait_for_response."""
        received_timeouts: list[float] = []

        class CapturingEscalation(BaseEscalation):
            async def notify(self, action, mode, risk_score=None): pass
            async def wait_for_response(self, action_id, timeout):
                received_timeouts.append(timeout)
                return None

        class Config:
            soft_gate_timeout_seconds = 42
            hard_gate_timeout_seconds = 300

        eng = GovernanceEngine(Config())
        await eng.enforce(GovernanceMode.SOFT_GATE, make_action(), CapturingEscalation())
        assert received_timeouts == [42.0]


# ---------------------------------------------------------------------------
# enforce() — HARD_GATE
# ---------------------------------------------------------------------------

class TestEnforceHardGate:
    async def test_timeout_blocks(self, fast_engine):
        """No response (None / timeout) → action blocked."""
        esc = RecordingEscalation(response=None)
        result = await fast_engine.enforce(GovernanceMode.HARD_GATE, make_action(), esc)
        assert result is False

    async def test_explicit_rejection_blocks(self, fast_engine):
        """Human rejects (False) → action blocked."""
        esc = RecordingEscalation(response=False)
        result = await fast_engine.enforce(GovernanceMode.HARD_GATE, make_action(), esc)
        assert result is False

    async def test_explicit_approval_proceeds(self, fast_engine):
        """Human approves (True) → action proceeds."""
        esc = RecordingEscalation(response=True)
        result = await fast_engine.enforce(GovernanceMode.HARD_GATE, make_action(), esc)
        assert result is True

    async def test_notification_sent(self, fast_engine):
        esc = RecordingEscalation(response=None)
        action = make_action()
        await fast_engine.enforce(GovernanceMode.HARD_GATE, action, esc)
        assert len(esc.notifications) == 1
        assert esc.notifications[0] == (action, GovernanceMode.HARD_GATE)

    async def test_uses_hard_gate_timeout(self):
        """Engine passes its configured hard_gate_timeout to wait_for_response."""
        received_timeouts: list[float] = []

        class CapturingEscalation(BaseEscalation):
            async def notify(self, action, mode, risk_score=None): pass
            async def wait_for_response(self, action_id, timeout):
                received_timeouts.append(timeout)
                return None

        class Config:
            soft_gate_timeout_seconds = 60
            hard_gate_timeout_seconds = 99

        eng = GovernanceEngine(Config())
        await eng.enforce(GovernanceMode.HARD_GATE, make_action(), CapturingEscalation())
        assert received_timeouts == [99.0]


# ---------------------------------------------------------------------------
# enforce() — QUARANTINE
# ---------------------------------------------------------------------------

class TestEnforceQuarantine:
    async def test_returns_false(self, engine):
        esc = RecordingEscalation()
        result = await engine.enforce(GovernanceMode.QUARANTINE, make_action(), esc)
        assert result is False

    async def test_notification_sent(self, engine):
        esc = RecordingEscalation()
        action = make_action()
        await engine.enforce(GovernanceMode.QUARANTINE, action, esc)
        assert len(esc.notifications) == 1
        assert esc.notifications[0] == (action, GovernanceMode.QUARANTINE)

    async def test_does_not_wait_for_response(self, engine):
        """QUARANTINE blocks immediately — never waits for a human."""
        class StrictEscalation(RecordingEscalation):
            async def wait_for_response(self, action_id, timeout):
                raise AssertionError("wait_for_response must not be called for QUARANTINE")

        esc = StrictEscalation()
        result = await engine.enforce(GovernanceMode.QUARANTINE, make_action(), esc)
        assert result is False


# ---------------------------------------------------------------------------
# Config injection
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_soft_gate_timeout(self):
        eng = GovernanceEngine()
        assert eng._soft_gate_timeout == 60.0

    def test_default_hard_gate_timeout(self):
        eng = GovernanceEngine()
        assert eng._hard_gate_timeout == 300.0

    def test_custom_timeouts_from_config(self):
        class Config:
            soft_gate_timeout_seconds = 15
            hard_gate_timeout_seconds = 120

        eng = GovernanceEngine(Config())
        assert eng._soft_gate_timeout == 15.0
        assert eng._hard_gate_timeout == 120.0

    def test_config_none_uses_defaults(self):
        eng = GovernanceEngine(config=None)
        assert eng._soft_gate_timeout == 60.0
        assert eng._hard_gate_timeout == 300.0


# ---------------------------------------------------------------------------
# Decide + enforce integration
# ---------------------------------------------------------------------------

class TestDecideEnforceIntegration:
    """decide() feeds directly into enforce() — verify end-to-end approval."""

    async def test_low_score_auto_approved(self, engine):
        profile = make_profile()
        risk_score = make_risk_score(10.0)
        mode = engine.decide(risk_score, profile)
        esc = RecordingEscalation()
        result = await engine.enforce(mode, make_action(), esc)
        assert mode == GovernanceMode.FULL_AUTO
        assert result is True

    async def test_high_score_quarantined(self, engine):
        profile = make_profile()
        risk_score = make_risk_score(90.0)
        mode = engine.decide(risk_score, profile)
        esc = RecordingEscalation()
        result = await engine.enforce(mode, make_action(), esc)
        assert mode == GovernanceMode.QUARANTINE
        assert result is False

    async def test_mid_score_hard_gate_approved_by_human(self, fast_engine):
        profile = make_profile()
        risk_score = make_risk_score(70.0)
        mode = fast_engine.decide(risk_score, profile)
        assert mode == GovernanceMode.HARD_GATE
        esc = RecordingEscalation(response=True)  # human approves
        result = await fast_engine.enforce(mode, make_action(), esc)
        assert result is True

    async def test_mid_score_hard_gate_timeout_blocked(self, fast_engine):
        profile = make_profile()
        risk_score = make_risk_score(70.0)
        mode = fast_engine.decide(risk_score, profile)
        esc = RecordingEscalation(response=None)  # no human response
        result = await fast_engine.enforce(mode, make_action(), esc)
        assert result is False
