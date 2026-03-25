"""Comprehensive unit tests for autonomica/scorer.py."""

import pytest

from autonomica.models import ActionType, AgentAction, AgentProfile, RiskScore
from autonomica.scorer import RiskScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_profile(
    trust_score: float = 50.0,
    per_tool_trust: dict | None = None,
) -> AgentProfile:
    return AgentProfile(
        agent_id="test-agent",
        agent_name="Test Agent",
        trust_score=trust_score,
        per_tool_trust=per_tool_trust or {},
    )


def make_action(
    tool_name: str = "test_tool",
    tool_input: dict | None = None,
    action_type: ActionType = ActionType.READ,
    metadata: dict | None = None,
) -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name=tool_name,
        tool_input=tool_input or {},
        action_type=action_type,
        metadata=metadata or {},
    )


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


# ---------------------------------------------------------------------------
# Scenario 1: Low risk — database read, no PII, trusted agent, familiar tool
# ---------------------------------------------------------------------------

class TestLowRiskAction:
    """READ action by a highly trusted agent using a well-known tool."""

    def test_composite_is_low(self, scorer):
        profile = make_profile(trust_score=90.0, per_tool_trust={"read_database": 20})
        action = make_action(
            tool_name="read_database",
            tool_input={"query": "SELECT id, name FROM products LIMIT 10"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        # Expected:
        #   financial_magnitude=0, data_sensitivity=0, reversibility=0,
        #   track_record=10, novelty=10, cascade=20
        # composite = 0*.25 + 0*.20 + 0*.20 + 10*.15 + 10*.10 + 20*.10 = 4.5
        assert result.composite_score == pytest.approx(4.5)

    def test_individual_signals(self, scorer):
        profile = make_profile(trust_score=90.0, per_tool_trust={"read_database": 20})
        action = make_action(
            tool_name="read_database",
            tool_input={"query": "SELECT * FROM orders"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 0.0
        assert result.data_sensitivity == 0.0
        assert result.reversibility == 0.0
        assert result.agent_track_record == pytest.approx(10.0)  # 100-90
        assert result.novelty == 10.0  # 20 uses >= 10
        assert result.cascade_risk == 20.0

    def test_score_is_full_auto_territory(self, scorer):
        profile = make_profile(trust_score=90.0, per_tool_trust={"read_database": 20})
        action = make_action(
            tool_name="read_database",
            tool_input={"query": "SELECT 1"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        # Composite 4.5 <= 15 (FULL_AUTO threshold)
        assert result.composite_score <= 15.0


# ---------------------------------------------------------------------------
# Scenario 2: Medium risk — email send with PII in input
# ---------------------------------------------------------------------------

class TestMediumRiskAction:
    """COMMUNICATE action with PII data, neutral agent, few tool uses."""

    def test_composite_in_medium_range(self, scorer):
        profile = make_profile(trust_score=50.0, per_tool_trust={"send_email": 3})
        action = make_action(
            tool_name="send_email",
            tool_input={
                "to": "user@example.com",   # 'email' key not here but value contains email
                "email": "user@example.com", # explicit PII key
                "subject": "Your order",
                "body": "Hi there",
            },
            action_type=ActionType.COMMUNICATE,
        )
        result = scorer.score(action, profile)
        # financial=0, sensitivity=60 (pii 'email'), reversibility=60,
        # track_record=50, novelty=40 (<10), cascade=20
        # composite = 0*.25 + 60*.20 + 60*.20 + 50*.15 + 40*.10 + 20*.10
        #           = 0 + 12 + 12 + 7.5 + 4 + 2 = 37.5
        assert result.composite_score == pytest.approx(37.5)

    def test_pii_detection_via_key(self, scorer):
        profile = make_profile()
        action = make_action(
            tool_input={"email": "test@example.com"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 60.0

    def test_pii_detection_via_value(self, scorer):
        """PII keyword appearing as a value substring is also detected."""
        profile = make_profile()
        action = make_action(
            tool_input={"field": "phone"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 60.0

    def test_communicate_reversibility(self, scorer):
        profile = make_profile()
        action = make_action(action_type=ActionType.COMMUNICATE)
        result = scorer.score(action, profile)
        assert result.reversibility == 60.0

    def test_few_uses_novelty(self, scorer):
        """Tool called 5 times → novelty=40."""
        profile = make_profile(per_tool_trust={"my_tool": 5})
        action = make_action(tool_name="my_tool")
        result = scorer.score(action, profile)
        assert result.novelty == 40.0


# ---------------------------------------------------------------------------
# Scenario 3: High risk — large financial transaction, new agent
# ---------------------------------------------------------------------------

class TestHighRiskAction:
    """FINANCIAL action, large amount, new agent, first tool use."""

    def test_composite_is_high(self, scorer):
        profile = make_profile(trust_score=50.0)  # new agent, tool not in dict
        action = make_action(
            tool_name="process_payment",
            tool_input={"amount": 500_000.0, "recipient": "vendor@corp.com"},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)
        # financial=80 ($500K), data_sensitivity=0, reversibility=70,
        # track_record=50, novelty=70 (first time), cascade=20
        # composite = 80*.25 + 0*.20 + 70*.20 + 50*.15 + 70*.10 + 20*.10
        #           = 20 + 0 + 14 + 7.5 + 7 + 2 = 50.5
        assert result.composite_score == pytest.approx(50.5)

    def test_million_dollar_transaction(self, scorer):
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="wire_transfer",
            tool_input={"amount": 2_000_000.0},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 100.0

    def test_financial_data_sensitivity(self, scorer):
        """account_number key → financial data sensitivity → 70."""
        profile = make_profile()
        action = make_action(
            tool_input={"account_number": "123456789"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 70.0

    def test_health_data_sensitivity(self, scorer):
        """diagnosis key → health data → 80 (highest tier)."""
        profile = make_profile()
        action = make_action(
            tool_input={"diagnosis": "Type 2 diabetes", "medication": "metformin"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 80.0

    def test_health_beats_financial_data(self, scorer):
        """Health data (80) takes priority over financial data (70)."""
        profile = make_profile()
        action = make_action(
            tool_input={"patient": "John", "account_number": "999"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 80.0

    def test_delete_reversibility(self, scorer):
        profile = make_profile()
        action = make_action(action_type=ActionType.DELETE)
        result = scorer.score(action, profile)
        assert result.reversibility == 80.0

    def test_first_time_tool_novelty(self, scorer):
        """Tool not in per_tool_trust → first use → novelty=70."""
        profile = make_profile(per_tool_trust={})
        action = make_action(tool_name="never_used_tool")
        result = scorer.score(action, profile)
        assert result.novelty == 70.0

    def test_low_trust_agent_raises_track_record(self, scorer):
        """Untrusted agent (trust=10) → track_record=90."""
        profile = make_profile(trust_score=10.0)
        action = make_action()
        result = scorer.score(action, profile)
        assert result.agent_track_record == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Financial magnitude thresholds
# ---------------------------------------------------------------------------

class TestFinancialMagnitude:
    @pytest.mark.parametrize("amount,expected", [
        (0.0, 0.0),
        (50.0, 0.0),
        (99.99, 0.0),
        (100.0, 20.0),
        (5_000.0, 20.0),
        (9_999.99, 20.0),
        (10_000.0, 50.0),
        (75_000.0, 50.0),
        (100_000.0, 80.0),
        (500_000.0, 80.0),
        (999_999.99, 80.0),
        (1_000_000.0, 100.0),
        (5_000_000.0, 100.0),
    ])
    def test_threshold_brackets(self, scorer, amount, expected):
        profile = make_profile()
        action = make_action(tool_input={"amount": amount}, action_type=ActionType.FINANCIAL)
        result = scorer.score(action, profile)
        assert result.financial_magnitude == expected

    def test_uses_abs_value(self, scorer):
        """Negative amounts (credits/refunds) should use absolute value."""
        profile = make_profile()
        action = make_action(tool_input={"amount": -500_000.0}, action_type=ActionType.FINANCIAL)
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 80.0

    def test_scans_multiple_amount_fields(self, scorer):
        """When multiple amount-like fields present, uses the largest."""
        profile = make_profile()
        action = make_action(
            tool_input={"fee": 50.0, "total": 150_000.0},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 80.0  # driven by 'total'

    def test_non_numeric_amount_ignored(self, scorer):
        """String 'amount' values don't crash and don't contribute to score."""
        profile = make_profile()
        action = make_action(
            tool_input={"amount": "not-a-number"},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 0.0

    def test_no_amount_fields(self, scorer):
        """No recognized amount fields → financial_magnitude=0."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT 1", "limit": 10},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.financial_magnitude == 0.0


# ---------------------------------------------------------------------------
# Cascade risk
# ---------------------------------------------------------------------------

class TestCascadeRisk:
    def test_default_cascade_risk(self, scorer):
        profile = make_profile()
        action = make_action()
        result = scorer.score(action, profile)
        assert result.cascade_risk == 20.0

    def test_cascade_risk_from_metadata(self, scorer):
        """N=3 downstream agents → min(3*15, 100) = 45."""
        profile = make_profile()
        action = make_action(metadata={"cascade_downstream_agents": 3})
        result = scorer.score(action, profile)
        assert result.cascade_risk == 45.0

    def test_cascade_risk_capped_at_100(self, scorer):
        """N=10 → min(150, 100) = 100."""
        profile = make_profile()
        action = make_action(metadata={"cascade_downstream_agents": 10})
        result = scorer.score(action, profile)
        assert result.cascade_risk == 100.0

    def test_cascade_risk_zero_agents(self, scorer):
        profile = make_profile()
        action = make_action(metadata={"cascade_downstream_agents": 0})
        result = scorer.score(action, profile)
        assert result.cascade_risk == 0.0


# ---------------------------------------------------------------------------
# Reversibility by ActionType
# ---------------------------------------------------------------------------

class TestReversibility:
    @pytest.mark.parametrize("action_type,expected", [
        (ActionType.READ, 0.0),
        (ActionType.WRITE, 30.0),
        (ActionType.COMMUNICATE, 60.0),
        (ActionType.FINANCIAL, 70.0),
        (ActionType.DELETE, 80.0),
    ])
    def test_all_action_types(self, scorer, action_type, expected):
        profile = make_profile()
        action = make_action(action_type=action_type)
        result = scorer.score(action, profile)
        assert result.reversibility == expected


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

class TestNovelty:
    @pytest.mark.parametrize("call_count,expected", [
        (None, 70.0),   # not in dict
        (0, 40.0),      # 0 < 10 (stored as 0 meaning has been recorded but count 0)
        (1, 40.0),
        (9, 40.0),
        (10, 10.0),
        (50, 10.0),
        (100, 10.0),
    ])
    def test_novelty_tiers(self, scorer, call_count, expected):
        per_tool = {}
        if call_count is not None:
            per_tool["my_tool"] = call_count
        profile = make_profile(per_tool_trust=per_tool)
        action = make_action(tool_name="my_tool")
        result = scorer.score(action, profile)
        assert result.novelty == expected


# ---------------------------------------------------------------------------
# Agent track record
# ---------------------------------------------------------------------------

class TestTrackRecord:
    @pytest.mark.parametrize("trust,expected", [
        (0.0, 100.0),
        (50.0, 50.0),
        (90.0, 10.0),
        (100.0, 0.0),
    ])
    def test_track_record_is_inverse_trust(self, scorer, trust, expected):
        profile = make_profile(trust_score=trust)
        action = make_action()
        result = scorer.score(action, profile)
        assert result.agent_track_record == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tool_input(self, scorer):
        """Empty tool_input shouldn't crash."""
        profile = make_profile()
        action = make_action(tool_input={})
        result = scorer.score(action, profile)
        assert isinstance(result, RiskScore)
        assert 0.0 <= result.composite_score <= 100.0

    def test_composite_never_exceeds_100(self, scorer):
        """Worst-case inputs can't produce a score above 100."""
        profile = make_profile(trust_score=0.0)  # track_record=100
        action = make_action(
            tool_name="never_used",
            tool_input={
                "amount": 10_000_000.0,
                "diagnosis": "cancer",
                "account_number": "123",
            },
            action_type=ActionType.DELETE,
            metadata={"cascade_downstream_agents": 20},
        )
        result = scorer.score(action, profile)
        assert result.composite_score <= 100.0

    def test_composite_never_below_zero(self, scorer):
        """Best-case inputs can't produce a negative score."""
        profile = make_profile(trust_score=100.0)
        action = make_action(
            tool_name="read_db",
            tool_input={"query": "SELECT 1"},
            action_type=ActionType.READ,
            metadata={"cascade_downstream_agents": 0},
        )
        result = scorer.score(action, profile)
        assert result.composite_score >= 0.0

    def test_explanation_is_non_empty_string(self, scorer):
        profile = make_profile()
        action = make_action()
        result = scorer.score(action, profile)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

    def test_explanation_contains_composite_score(self, scorer):
        profile = make_profile()
        action = make_action()
        result = scorer.score(action, profile)
        assert "Composite score" in result.explanation

    def test_explanation_contains_all_signal_names(self, scorer):
        profile = make_profile()
        action = make_action()
        result = scorer.score(action, profile)
        for signal in ("financial_magnitude", "data_sensitivity", "reversibility",
                       "agent_track_record", "novelty", "cascade_risk"):
            assert signal in result.explanation

    def test_deeply_nested_input_scanned(self, scorer):
        """Nested dict values should be scanned for sensitivity keywords."""
        profile = make_profile()
        action = make_action(
            tool_input={"payload": {"field": "email"}},
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == 60.0

    def test_returns_risk_score_model(self, scorer):
        profile = make_profile()
        action = make_action()
        result = scorer.score(action, profile)
        assert isinstance(result, RiskScore)

    def test_custom_weights_sum_to_one_respected(self):
        """Scorer respects custom weights from config."""

        class MockConfig:
            scoring_weights = {
                "financial_magnitude": 1.0,
                "data_sensitivity": 0.0,
                "reversibility": 0.0,
                "agent_track_record": 0.0,
                "novelty": 0.0,
                "cascade_risk": 0.0,
            }
            pii_patterns = list(("email",))
            financial_data_patterns = list(("account_number",))
            health_data_patterns = list(("diagnosis",))
            financial_thresholds = {
                "low": 100.0,
                "medium": 10_000.0,
                "high": 100_000.0,
                "critical": 1_000_000.0,
            }

        scorer = RiskScorer(MockConfig())
        profile = make_profile(trust_score=0.0)
        action = make_action(
            tool_input={"amount": 500_000.0},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)
        # Only financial_magnitude contributes: 80 * 1.0 = 80
        assert result.composite_score == pytest.approx(80.0)
