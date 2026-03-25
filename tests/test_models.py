"""Unit tests for autonomica/models.py."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceDecision,
    GovernanceMode,
    RiskScore,
)


# ---------------------------------------------------------------------------
# GovernanceMode
# ---------------------------------------------------------------------------

class TestGovernanceMode:
    def test_values_are_ordered(self):
        assert GovernanceMode.FULL_AUTO < GovernanceMode.LOG_AND_ALERT
        assert GovernanceMode.LOG_AND_ALERT < GovernanceMode.SOFT_GATE
        assert GovernanceMode.SOFT_GATE < GovernanceMode.HARD_GATE
        assert GovernanceMode.HARD_GATE < GovernanceMode.QUARANTINE

    def test_integer_values(self):
        assert GovernanceMode.FULL_AUTO == 0
        assert GovernanceMode.LOG_AND_ALERT == 1
        assert GovernanceMode.SOFT_GATE == 2
        assert GovernanceMode.HARD_GATE == 3
        assert GovernanceMode.QUARANTINE == 4

    def test_is_int_enum(self):
        assert isinstance(GovernanceMode.FULL_AUTO, int)


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------

class TestActionType:
    def test_all_values(self):
        assert ActionType.READ == "read"
        assert ActionType.WRITE == "write"
        assert ActionType.COMMUNICATE == "communicate"
        assert ActionType.DELETE == "delete"
        assert ActionType.FINANCIAL == "financial"

    def test_is_str_enum(self):
        assert isinstance(ActionType.READ, str)

    def test_five_types(self):
        assert len(ActionType) == 5


# ---------------------------------------------------------------------------
# AgentAction
# ---------------------------------------------------------------------------

class TestAgentAction:
    def test_auto_generates_action_id(self, read_action):
        # Should be a valid UUID string
        UUID(read_action.action_id)  # raises if invalid

    def test_unique_ids(self, read_action, financial_action):
        assert read_action.action_id != financial_action.action_id

    def test_timestamp_is_utc_aware(self, read_action):
        assert read_action.timestamp.tzinfo is not None

    def test_metadata_defaults_to_empty_dict(self, read_action):
        assert read_action.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        a1 = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.READ,
        )
        a2 = AgentAction(
            agent_id="b", agent_name="B", tool_name="t",
            tool_input={}, action_type=ActionType.READ,
        )
        a1.metadata["key"] = "value"
        assert "key" not in a2.metadata

    def test_stores_tool_input(self, financial_action):
        assert financial_action.tool_input["amount"] == 50000.0

    def test_custom_action_id(self):
        action = AgentAction(
            action_id="custom-id",
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.READ,
        )
        assert action.action_id == "custom-id"

    def test_all_action_types_accepted(self):
        for action_type in ActionType:
            action = AgentAction(
                agent_id="a", agent_name="A", tool_name="t",
                tool_input={}, action_type=action_type,
            )
            assert action.action_type == action_type


# ---------------------------------------------------------------------------
# RiskScore
# ---------------------------------------------------------------------------

class TestRiskScore:
    def make_score(self, **overrides) -> RiskScore:
        defaults = dict(
            composite_score=25.0,
            financial_magnitude=10.0,
            data_sensitivity=20.0,
            reversibility=30.0,
            agent_track_record=50.0,
            novelty=40.0,
            cascade_risk=20.0,
            explanation="Test score",
        )
        defaults.update(overrides)
        return RiskScore(**defaults)

    def test_valid_construction(self):
        score = self.make_score()
        assert score.composite_score == 25.0

    def test_explanation_stored(self):
        score = self.make_score(explanation="High financial risk")
        assert score.explanation == "High financial risk"

    def test_all_sub_scores_stored(self):
        score = self.make_score(
            financial_magnitude=80.0,
            data_sensitivity=60.0,
            reversibility=70.0,
            agent_track_record=40.0,
            novelty=50.0,
            cascade_risk=30.0,
        )
        assert score.financial_magnitude == 80.0
        assert score.data_sensitivity == 60.0
        assert score.reversibility == 70.0
        assert score.agent_track_record == 40.0
        assert score.novelty == 50.0
        assert score.cascade_risk == 30.0

    def test_boundary_scores(self):
        # Scores at 0 and 100 are valid
        self.make_score(composite_score=0.0, financial_magnitude=0.0, data_sensitivity=0.0,
                        reversibility=0.0, agent_track_record=0.0, novelty=0.0, cascade_risk=0.0)
        self.make_score(composite_score=100.0, financial_magnitude=100.0, data_sensitivity=100.0,
                        reversibility=100.0, agent_track_record=100.0, novelty=100.0, cascade_risk=100.0)


# ---------------------------------------------------------------------------
# GovernanceDecision
# ---------------------------------------------------------------------------

class TestGovernanceDecision:
    def make_risk_score(self) -> RiskScore:
        return RiskScore(
            composite_score=25.0,
            financial_magnitude=10.0,
            data_sensitivity=20.0,
            reversibility=30.0,
            agent_track_record=50.0,
            novelty=40.0,
            cascade_risk=20.0,
            explanation="test",
        )

    def test_basic_approved_decision(self):
        decision = GovernanceDecision(
            action_id="abc-123",
            risk_score=self.make_risk_score(),
            mode=GovernanceMode.LOG_AND_ALERT,
            approved=True,
            decision_time_ms=2.5,
        )
        assert decision.approved is True
        assert decision.mode == GovernanceMode.LOG_AND_ALERT

    def test_optional_fields_default_to_none(self):
        decision = GovernanceDecision(
            action_id="abc-123",
            risk_score=self.make_risk_score(),
            mode=GovernanceMode.FULL_AUTO,
            approved=True,
            decision_time_ms=1.0,
        )
        assert decision.escalated_to is None
        assert decision.human_override is None

    def test_human_override_can_be_true_or_false(self):
        for val in (True, False):
            d = GovernanceDecision(
                action_id="x",
                risk_score=self.make_risk_score(),
                mode=GovernanceMode.HARD_GATE,
                approved=val,
                decision_time_ms=10.0,
                human_override=val,
            )
            assert d.human_override is val

    def test_timestamp_is_utc_aware(self):
        decision = GovernanceDecision(
            action_id="x",
            risk_score=self.make_risk_score(),
            mode=GovernanceMode.QUARANTINE,
            approved=False,
            decision_time_ms=0.5,
        )
        assert decision.timestamp.tzinfo is not None

    def test_escalated_to_stored(self):
        decision = GovernanceDecision(
            action_id="x",
            risk_score=self.make_risk_score(),
            mode=GovernanceMode.HARD_GATE,
            approved=False,
            decision_time_ms=5.0,
            escalated_to="slack:#ops-alerts",
        )
        assert decision.escalated_to == "slack:#ops-alerts"


# ---------------------------------------------------------------------------
# AgentProfile
# ---------------------------------------------------------------------------

class TestAgentProfile:
    def test_defaults(self, agent_profile):
        assert agent_profile.total_actions == 0
        assert agent_profile.approved_actions == 0
        assert agent_profile.escalated_actions == 0
        assert agent_profile.incidents == 0
        assert agent_profile.false_escalations == 0
        assert agent_profile.trust_score == 50.0
        assert agent_profile.vagal_tone == 50.0

    def test_default_thresholds(self, agent_profile):
        t = agent_profile.mode_thresholds
        assert t["full_auto_max"] == 15.0
        assert t["log_alert_max"] == 35.0
        assert t["soft_gate_max"] == 60.0
        assert t["hard_gate_max"] == 85.0

    def test_per_tool_trust_defaults_empty(self, agent_profile):
        assert agent_profile.per_tool_trust == {}

    def test_thresholds_are_independent_per_instance(self):
        p1 = AgentProfile(agent_id="a1", agent_name="A1")
        p2 = AgentProfile(agent_id="a2", agent_name="A2")
        p1.mode_thresholds["full_auto_max"] = 99.0
        assert p2.mode_thresholds["full_auto_max"] == 15.0

    def test_per_tool_trust_is_independent_per_instance(self):
        p1 = AgentProfile(agent_id="a1", agent_name="A1")
        p2 = AgentProfile(agent_id="a2", agent_name="A2")
        p1.per_tool_trust["send_email"] = 90.0
        assert "send_email" not in p2.per_tool_trust

    def test_timestamps_are_utc_aware(self, agent_profile):
        assert agent_profile.created_at.tzinfo is not None
        assert agent_profile.updated_at.tzinfo is not None

    def test_custom_trust_score(self):
        p = AgentProfile(agent_id="trusted", agent_name="Trusted", trust_score=90.0)
        assert p.trust_score == 90.0

    def test_thresholds_increase_with_mode(self, agent_profile):
        t = agent_profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]
