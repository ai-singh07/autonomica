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
            # Use explicit column list (not SELECT *) and a non-sensitive table so
            # no SQL-aware bumps apply — this test is about the base signal values.
            tool_input={"query": "SELECT id, status FROM orders"},
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


# ---------------------------------------------------------------------------
# Per-tool signal overrides
# ---------------------------------------------------------------------------

class TestToolOverrides:
    """
    Verify that tool_overrides in config pins specific signal scores for a
    named tool while leaving all other signals and all other tools untouched.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _scorer_with_overrides(overrides: dict) -> RiskScorer:
        """Build a RiskScorer from an AutonomicaConfig with tool_overrides."""
        from autonomica.config import AutonomicaConfig
        config = AutonomicaConfig(tool_overrides=overrides)
        return RiskScorer(config)

    # ── Full override: both financial_magnitude and data_sensitivity pinned ──

    def test_full_override_replaces_heuristic_scores(self):
        """Overridden signals use pinned values regardless of tool_input content."""
        scorer = self._scorer_with_overrides({
            "write_tutorial": {
                "data_sensitivity": 0,
                "financial_magnitude": 0,
            }
        })
        # Input that would normally trigger high financial + PII scores
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="write_tutorial",
            tool_input={"amount": 5_000_000.0, "email": "user@example.com"},
            action_type=ActionType.WRITE,
        )
        result = scorer.score(action, profile)

        assert result.financial_magnitude == 0.0, (
            "financial_magnitude should be pinned to 0 by override"
        )
        assert result.data_sensitivity == 0.0, (
            "data_sensitivity should be pinned to 0 by override"
        )

    def test_full_override_composite_uses_pinned_values(self):
        """Composite score must be computed from the pinned values, not heuristics."""
        scorer = self._scorer_with_overrides({
            "write_tutorial": {"data_sensitivity": 0, "financial_magnitude": 0}
        })
        profile = make_profile(trust_score=50.0, per_tool_trust={"write_tutorial": 20})
        action = make_action(
            tool_name="write_tutorial",
            tool_input={"amount": 5_000_000.0, "email": "user@example.com"},
            action_type=ActionType.WRITE,
        )
        result = scorer.score(action, profile)

        # financial=0, data_sensitivity=0, reversibility=30 (WRITE),
        # track_record=50 (100-50), novelty=10 (>=10 uses), cascade=20
        # composite = 0*.25 + 0*.20 + 30*.20 + 50*.15 + 10*.10 + 20*.10
        #           = 0 + 0 + 6 + 7.5 + 1 + 2 = 16.5
        assert result.composite_score == pytest.approx(16.5)

    def test_payment_tool_override_raises_risk(self):
        """process_payment override pins financial_magnitude=90, reversibility=80."""
        scorer = self._scorer_with_overrides({
            "process_payment": {"financial_magnitude": 90, "reversibility": 80}
        })
        # Low-value input that would normally score 0 financial, low reversibility
        profile = make_profile(trust_score=90.0, per_tool_trust={"process_payment": 50})
        action = make_action(
            tool_name="process_payment",
            tool_input={"amount": 1.0},        # $1 — would normally score 0
            action_type=ActionType.FINANCIAL,  # heuristic reversibility=70
        )
        result = scorer.score(action, profile)

        assert result.financial_magnitude == 90.0
        assert result.reversibility == 80.0

        # Composite: 90*.25 + 0*.20 + 80*.20 + 10*.15 + 10*.10 + 20*.10
        #          = 22.5 + 0 + 16 + 1.5 + 1 + 2 = 43.0
        assert result.composite_score == pytest.approx(43.0)

    # ── Partial override: one signal pinned, others score normally ────────────

    def test_partial_override_pins_only_specified_signal(self):
        """Only the overridden signal is pinned; all others use heuristics."""
        scorer = self._scorer_with_overrides({
            "send_newsletter": {"data_sensitivity": 0}
        })
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="send_newsletter",
            tool_input={"email": "user@example.com", "amount": 500_000.0},
            action_type=ActionType.COMMUNICATE,
        )
        result = scorer.score(action, profile)

        # data_sensitivity pinned to 0 (would normally be 60 for 'email' PII)
        assert result.data_sensitivity == 0.0

        # financial_magnitude still scored heuristically: $500K → 80
        assert result.financial_magnitude == 80.0

        # reversibility still scored heuristically: COMMUNICATE → 60
        assert result.reversibility == 60.0

    def test_partial_override_composite_mixes_heuristic_and_pinned(self):
        """Composite uses pinned value for overridden signal, heuristic for rest."""
        scorer = self._scorer_with_overrides({
            "send_newsletter": {"data_sensitivity": 0}
        })
        profile = make_profile(trust_score=50.0, per_tool_trust={"send_newsletter": 20})
        action = make_action(
            tool_name="send_newsletter",
            tool_input={"email": "user@example.com"},
            action_type=ActionType.COMMUNICATE,
        )
        result = scorer.score(action, profile)

        # financial=0, data_sensitivity=0 (override), reversibility=60,
        # track_record=50, novelty=10 (>=10 uses), cascade=20
        # composite = 0*.25 + 0*.20 + 60*.20 + 50*.15 + 10*.10 + 20*.10
        #           = 0 + 0 + 12 + 7.5 + 1 + 2 = 22.5
        assert result.composite_score == pytest.approx(22.5)

    def test_novelty_override_does_not_affect_other_signals(self):
        """Pinning novelty=0 leaves financial_magnitude unchanged."""
        scorer = self._scorer_with_overrides({
            "familiar_tool": {"novelty": 0}
        })
        profile = make_profile(trust_score=50.0)  # tool not in per_tool_trust → heuristic=70
        action = make_action(
            tool_name="familiar_tool",
            tool_input={"amount": 200_000.0},    # heuristic → 80
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)

        assert result.novelty == 0.0
        assert result.financial_magnitude == 80.0  # heuristic still runs

    # ── Override doesn't bleed to other tools ────────────────────────────────

    def test_override_does_not_affect_other_tool(self):
        """An override for tool A must not change scores for tool B."""
        scorer = self._scorer_with_overrides({
            "write_tutorial": {"financial_magnitude": 0, "data_sensitivity": 0}
        })
        profile = make_profile(trust_score=50.0)
        # This action uses a DIFFERENT tool — overrides must not apply
        action = make_action(
            tool_name="process_payment",
            tool_input={"amount": 5_000_000.0, "email": "user@example.com"},
            action_type=ActionType.FINANCIAL,
        )
        result = scorer.score(action, profile)

        # Heuristic scores must apply normally for process_payment
        assert result.financial_magnitude == 100.0  # $5M → critical
        assert result.data_sensitivity == 60.0       # 'email' PII

    def test_no_override_at_all_behaves_identically_to_default(self):
        """A config with empty tool_overrides must produce identical scores."""
        from autonomica.config import AutonomicaConfig
        scorer_default = RiskScorer()
        scorer_empty_override = RiskScorer(AutonomicaConfig(tool_overrides={}))

        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="any_tool",
            tool_input={"amount": 50_000.0, "email": "x@y.com"},
            action_type=ActionType.WRITE,
        )

        r1 = scorer_default.score(action, profile)
        r2 = scorer_empty_override.score(action, profile)

        assert r1.composite_score == r2.composite_score
        assert r1.financial_magnitude == r2.financial_magnitude
        assert r1.data_sensitivity == r2.data_sensitivity

    # ── Explanation reflects overrides ────────────────────────────────────────

    def test_overridden_signals_tagged_in_explanation(self):
        """Each overridden signal must have '[override]' in the explanation."""
        scorer = self._scorer_with_overrides({
            "write_tutorial": {"data_sensitivity": 0, "financial_magnitude": 0}
        })
        profile = make_profile()
        action = make_action(tool_name="write_tutorial")
        result = scorer.score(action, profile)

        assert "[override]" in result.explanation
        # Both overridden signals should be marked
        lines = result.explanation.splitlines()
        ds_line = next(l for l in lines if "data_sensitivity" in l)
        fm_line = next(l for l in lines if "financial_magnitude" in l)
        assert "[override]" in ds_line
        assert "[override]" in fm_line

    def test_non_overridden_signals_not_tagged(self):
        """Signals not in the override dict must NOT have '[override]' in their line."""
        scorer = self._scorer_with_overrides({
            "write_tutorial": {"data_sensitivity": 0}
        })
        profile = make_profile()
        action = make_action(tool_name="write_tutorial")
        result = scorer.score(action, profile)

        lines = result.explanation.splitlines()
        fm_line = next(l for l in lines if "financial_magnitude" in l)
        assert "[override]" not in fm_line

    def test_explanation_has_no_override_tag_when_no_overrides(self):
        """Default scorer (no config) must never produce '[override]' tags."""
        scorer = RiskScorer()
        profile = make_profile()
        action = make_action(tool_name="any_tool")
        result = scorer.score(action, profile)
        assert "[override]" not in result.explanation

    # ── Config validation ─────────────────────────────────────────────────────

    def test_invalid_signal_name_rejected(self):
        """Unknown signal name in tool_overrides must raise a ValidationError."""
        from autonomica.config import AutonomicaConfig
        with pytest.raises(Exception):  # pydantic ValidationError
            AutonomicaConfig(tool_overrides={"my_tool": {"typo_signal": 50}})

    def test_out_of_range_value_rejected(self):
        """Signal value outside [0, 100] must raise a ValidationError."""
        from autonomica.config import AutonomicaConfig
        with pytest.raises(Exception):
            AutonomicaConfig(tool_overrides={"my_tool": {"novelty": 150}})

    def test_negative_value_rejected(self):
        """Negative signal value must raise a ValidationError."""
        from autonomica.config import AutonomicaConfig
        with pytest.raises(Exception):
            AutonomicaConfig(tool_overrides={"my_tool": {"data_sensitivity": -1}})

    def test_boundary_values_accepted(self):
        """Values exactly 0 and 100 are valid."""
        from autonomica.config import AutonomicaConfig
        config = AutonomicaConfig(tool_overrides={
            "my_tool": {"novelty": 0, "cascade_risk": 100}
        })
        assert config.tool_overrides["my_tool"]["novelty"] == 0
        assert config.tool_overrides["my_tool"]["cascade_risk"] == 100

    def test_multiple_tools_multiple_signals(self):
        """Multiple tools can each have their own independent overrides."""
        scorer = self._scorer_with_overrides({
            "read_logs": {"novelty": 0, "cascade_risk": 0},
            "send_wire": {"financial_magnitude": 100, "reversibility": 90},
        })
        profile = make_profile(trust_score=80.0, per_tool_trust={"read_logs": 0})

        # read_logs: novelty=0, cascade=0 (overridden); reversibility=0 (READ heuristic)
        action_read = make_action(
            tool_name="read_logs",
            tool_input={},
            action_type=ActionType.READ,
        )
        r_read = scorer.score(action_read, profile)
        assert r_read.novelty == 0.0
        assert r_read.cascade_risk == 0.0
        assert r_read.reversibility == 0.0   # READ heuristic, not overridden

        # send_wire: financial=100, reversibility=90 (overridden)
        action_wire = make_action(
            tool_name="send_wire",
            tool_input={"amount": 1.0},       # would normally score 0 financial
            action_type=ActionType.FINANCIAL,  # would normally score 70 reversibility
        )
        r_wire = scorer.score(action_wire, profile)
        assert r_wire.financial_magnitude == 100.0
        assert r_wire.reversibility == 90.0


# ---------------------------------------------------------------------------
# SQL-aware argument scoring
# ---------------------------------------------------------------------------

class TestSQLAwareDataSensitivity:
    """
    Verify the SQL-aware bumps on data_sensitivity:
      +20  SELECT * or DELETE without WHERE
      −10  LIMIT clause
      +30  sensitive table referenced
    All checks are case-insensitive and work on any string value in tool_input.
    """

    # ── SELECT * ─────────────────────────────────────────────────────────────

    def test_select_star_bumps_sensitivity_by_20(self, scorer):
        """SELECT * on a non-sensitive table → base 0 + 20 = 20."""
        profile = make_profile()
        action = make_action(tool_input={"query": "SELECT * FROM orders"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(20.0)

    def test_select_specific_columns_no_bump(self, scorer):
        """SELECT id, name (not *) must not trigger the +20 bump."""
        profile = make_profile()
        action = make_action(tool_input={"query": "SELECT id, name FROM orders"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_select_star_with_where_still_bumps(self, scorer):
        """SELECT * even with WHERE still bumps +20 (still broad column access)."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM orders WHERE status = 'paid'"}
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(20.0)

    # ── DELETE without WHERE ──────────────────────────────────────────────────

    def test_delete_without_where_bumps_by_20(self, scorer):
        """DELETE with no WHERE clause → base 0 + 20 = 20."""
        profile = make_profile()
        action = make_action(tool_input={"sql": "DELETE FROM sessions"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(20.0)

    def test_delete_with_where_no_bump(self, scorer):
        """DELETE … WHERE … is a targeted operation — no sensitivity bump."""
        profile = make_profile()
        action = make_action(
            tool_input={"sql": "DELETE FROM sessions WHERE id = 42"}
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_delete_without_where_and_select_star_only_one_bump(self, scorer):
        """SELECT * and DELETE without WHERE both present → +20 (not +40)."""
        profile = make_profile()
        action = make_action(
            tool_input={"script": "SELECT * FROM logs; DELETE FROM logs"}
        )
        result = scorer.score(action, profile)
        # broad=True triggers once → +20
        assert result.data_sensitivity == pytest.approx(20.0)

    # ── LIMIT clause ─────────────────────────────────────────────────────────

    def test_limit_alone_does_not_go_below_zero(self, scorer):
        """LIMIT on a non-sensitive, non-broad query → clamp to 0, not −10."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT id FROM orders LIMIT 5"}
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_select_star_plus_limit(self, scorer):
        """SELECT * (+20) with LIMIT (−10) → net +10."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM orders LIMIT 10"}
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(10.0)

    def test_limit_reduces_sensitive_table_bump(self, scorer):
        """Sensitive table (+30) with LIMIT (−10) → net +20."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT id FROM users LIMIT 5"}
        )
        result = scorer.score(action, profile)
        # base=0, +30 (users), −10 (LIMIT) = 20
        assert result.data_sensitivity == pytest.approx(20.0)

    # ── Sensitive tables ──────────────────────────────────────────────────────

    def test_sensitive_table_bumps_by_30(self, scorer):
        """Reference to 'users' table → +30."""
        profile = make_profile()
        action = make_action(tool_input={"query": "SELECT id FROM users"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(30.0)

    def test_all_default_sensitive_tables_trigger_bump(self, scorer):
        """Every table in the default sensitive_tables list must trigger +30.

        'medical_records' also contains the health-data keyword 'medical', so its
        base score is 80 and the total (80+30) clamps to 100.  All other tables
        yield base 0 + 30 = 30.
        """
        profile = make_profile()
        # (table_name, expected_data_sensitivity)
        expected = {
            "users": 30.0,
            "payments": 30.0,
            "accounts": 30.0,
            "credentials": 30.0,
            "medical_records": 100.0,  # health-data base 80 + 30 sensitive → clamp 100
        }
        for table, exp in expected.items():
            action = make_action(
                tool_input={"query": f"SELECT id FROM {table}"}
            )
            result = scorer.score(action, profile)
            assert result.data_sensitivity == pytest.approx(exp), (
                f"Expected {exp} for table '{table}', got {result.data_sensitivity}"
            )

    def test_multiple_sensitive_tables_only_one_bump(self, scorer):
        """Query joining two sensitive tables still triggers only +30, not +60."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM users JOIN payments ON users.id = payments.user_id"}
        )
        result = scorer.score(action, profile)
        # +20 (SELECT *) + +30 (first sensitive table found) = 50
        assert result.data_sensitivity == pytest.approx(50.0)

    def test_non_sensitive_table_no_bump(self, scorer):
        """A table not in sensitive_tables must not trigger the +30 bump."""
        profile = make_profile()
        action = make_action(tool_input={"query": "SELECT id FROM products"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_partial_word_does_not_match_sensitive_table(self, scorer):
        """'super_users' must not trigger the 'users' sensitive table bump."""
        profile = make_profile()
        action = make_action(tool_input={"query": "SELECT id FROM super_users"})
        result = scorer.score(action, profile)
        # 'users' appears but only as part of 'super_users' — word boundary blocks it
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_custom_sensitive_tables_config(self):
        """Custom sensitive_tables list overrides the defaults."""
        from autonomica.config import AutonomicaConfig
        config = AutonomicaConfig(sensitive_tables=["invoices", "contracts"])
        custom_scorer = RiskScorer(config)

        profile = make_profile()

        # "invoices" is in custom list → +30
        action_hit = make_action(tool_input={"query": "SELECT * FROM invoices"})
        result_hit = custom_scorer.score(action_hit, profile)
        # SELECT * (+20) + invoices (+30) = 50
        assert result_hit.data_sensitivity == pytest.approx(50.0)

        # "users" is NOT in custom list → no table bump
        action_miss = make_action(tool_input={"query": "SELECT id FROM users"})
        result_miss = custom_scorer.score(action_miss, profile)
        assert result_miss.data_sensitivity == pytest.approx(0.0)

    # ── Combined bumps ────────────────────────────────────────────────────────

    def test_select_star_sensitive_table_limit(self, scorer):
        """SELECT * (+20) + sensitive table (+30) + LIMIT (−10) = 40."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM users LIMIT 100"}
        )
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(40.0)

    def test_base_pii_plus_select_star_bump(self, scorer):
        """PII keyword (base 60) + SELECT * (+20) = 80, not exceeding 100."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM contacts WHERE email = 'x@y.com'"}
        )
        result = scorer.score(action, profile)
        # base=60 (PII 'email'), +20 (SELECT *) = 80
        assert result.data_sensitivity == pytest.approx(80.0)

    def test_health_base_plus_sensitive_table_capped_at_100(self, scorer):
        """health base (80) + sensitive table (+30) is clamped to 100."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT diagnosis FROM medical_records"}
        )
        result = scorer.score(action, profile)
        # base=80 (health 'diagnosis'), +30 (medical_records) → clamp to 100
        assert result.data_sensitivity == pytest.approx(100.0)

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_uppercase_sql_keywords_detected(self, scorer):
        """SQL keywords in any case must be detected (flattened text is lowercase)."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT * FROM USERS LIMIT 5"}
        )
        result = scorer.score(action, profile)
        # SELECT * (+20) + USERS (+30) + LIMIT (−10) = 40
        assert result.data_sensitivity == pytest.approx(40.0)

    def test_mixed_case_delete_without_where(self, scorer):
        profile = make_profile()
        action = make_action(tool_input={"sql": "Delete From Temp_Table"})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(20.0)

    # ── Non-SQL inputs are unaffected ─────────────────────────────────────────

    def test_non_sql_input_no_bump(self, scorer):
        """Tool inputs without SQL content must not receive any SQL bumps."""
        profile = make_profile()
        action = make_action(tool_input={"message": "hello world", "count": 42})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)

    def test_numeric_only_input_no_bump(self, scorer):
        profile = make_profile()
        action = make_action(tool_input={"amount": 500.0, "tax_rate": 0.08})
        result = scorer.score(action, profile)
        assert result.data_sensitivity == pytest.approx(0.0)


class TestSQLAwareReversibility:
    """
    DDL keywords (DROP, TRUNCATE, ALTER) override reversibility to 95
    regardless of ActionType.  Non-DDL queries use the existing action-type map.
    """

    @pytest.mark.parametrize("ddl_statement,action_type", [
        ("DROP TABLE audit_logs", ActionType.DELETE),
        ("TRUNCATE TABLE sessions", ActionType.WRITE),
        ("ALTER TABLE users ADD COLUMN mfa_enabled boolean", ActionType.WRITE),
        ("drop table if exists temp_results", ActionType.DELETE),   # lowercase
        ("truncate payments", ActionType.FINANCIAL),
        ("Alter Table accounts Rename Column old TO new", ActionType.WRITE),  # mixed case
    ])
    def test_ddl_sets_reversibility_to_95(self, scorer, ddl_statement, action_type):
        """Any DDL statement must override reversibility to 95."""
        profile = make_profile()
        action = make_action(
            tool_input={"sql": ddl_statement},
            action_type=action_type,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(95.0), (
            f"Expected 95 for DDL '{ddl_statement}', got {result.reversibility}"
        )

    def test_normal_select_uses_action_type_reversibility(self, scorer):
        """A plain SELECT query must use the READ heuristic (0), not DDL override."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT id FROM users WHERE id = 1"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(0.0)

    def test_insert_uses_write_reversibility(self, scorer):
        """INSERT (no DDL keywords) → WRITE action type → reversibility 30."""
        profile = make_profile()
        action = make_action(
            tool_input={"sql": "INSERT INTO orders (user_id) VALUES (1)"},
            action_type=ActionType.WRITE,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(30.0)

    def test_update_uses_write_reversibility(self, scorer):
        """UPDATE with no DDL keywords → WRITE → reversibility 30."""
        profile = make_profile()
        action = make_action(
            tool_input={"sql": "UPDATE orders SET status = 'shipped' WHERE id = 1"},
            action_type=ActionType.WRITE,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(30.0)

    def test_delete_dml_uses_delete_reversibility(self, scorer):
        """DELETE DML (not DDL) with WHERE → DELETE action type → reversibility 80."""
        profile = make_profile()
        action = make_action(
            tool_input={"sql": "DELETE FROM sessions WHERE expired = true"},
            action_type=ActionType.DELETE,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(80.0)

    def test_alter_in_column_name_not_a_ddl(self, scorer):
        """'alter_date' as a column reference must NOT trigger the DDL override."""
        profile = make_profile()
        action = make_action(
            tool_input={"query": "SELECT alter_date FROM schema_changes"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)
        # 'alter' appears as part of 'alter_date' — word boundary blocks DDL detection
        assert result.reversibility == pytest.approx(0.0)

    def test_drop_in_value_string_triggers_ddl(self, scorer):
        """The DDL check applies to any string value, not just SQL keys."""
        profile = make_profile()
        action = make_action(
            tool_input={"statement": "DROP TABLE old_backups"},
            action_type=ActionType.DELETE,
        )
        result = scorer.score(action, profile)
        assert result.reversibility == pytest.approx(95.0)


class TestSQLAwareComposite:
    """End-to-end composite score tests with realistic SQL tool inputs."""

    def test_high_risk_unbounded_delete_on_users(self, scorer):
        """
        DELETE FROM users (no WHERE) against a sensitive table:
          data_sensitivity = 0 + 20 (no-WHERE DELETE) + 30 (users) = 50
          reversibility    = 80 (DELETE action type, no DDL)
          Others score normally for a new agent with trust=50.
        """
        profile = make_profile(trust_score=50.0)  # track_record=50, novelty=70
        action = make_action(
            tool_name="run_query",
            tool_input={"query": "DELETE FROM users"},
            action_type=ActionType.DELETE,
        )
        result = scorer.score(action, profile)

        assert result.data_sensitivity == pytest.approx(50.0)
        assert result.reversibility == pytest.approx(80.0)
        # composite = 0*.25 + 50*.20 + 80*.20 + 50*.15 + 70*.10 + 20*.10
        #           = 0 + 10 + 16 + 7.5 + 7 + 2 = 42.5
        assert result.composite_score == pytest.approx(42.5)

    def test_ddl_drop_on_payments_is_very_high_risk(self, scorer):
        """
        DROP TABLE payments:
          data_sensitivity = 0 + 30 (payments sensitive) = 30  [no SELECT*/DELETE w/o WHERE]
          reversibility    = 95  (DDL override)
        """
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="admin_sql",
            tool_input={"sql": "DROP TABLE payments"},
            action_type=ActionType.DELETE,
        )
        result = scorer.score(action, profile)

        assert result.data_sensitivity == pytest.approx(30.0)
        assert result.reversibility == pytest.approx(95.0)
        # composite = 0*.25 + 30*.20 + 95*.20 + 50*.15 + 70*.10 + 20*.10
        #           = 0 + 6 + 19 + 7.5 + 7 + 2 = 41.5
        assert result.composite_score == pytest.approx(41.5)

    def test_low_risk_limited_read_on_nonsensitive_table(self, scorer):
        """
        SELECT id FROM products LIMIT 10:
          data_sensitivity = 0 (no PII, no sensitive table, LIMIT clamp=0)
          reversibility    = 0 (READ)
        """
        profile = make_profile(trust_score=90.0, per_tool_trust={"read_db": 20})
        action = make_action(
            tool_name="read_db",
            tool_input={"query": "SELECT id FROM products LIMIT 10"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)

        assert result.data_sensitivity == pytest.approx(0.0)
        assert result.reversibility == pytest.approx(0.0)
        # composite = 0*.25 + 0*.20 + 0*.20 + 10*.15 + 10*.10 + 20*.10 = 4.5
        assert result.composite_score == pytest.approx(4.5)

    def test_select_star_from_credentials_worst_case(self, scorer):
        """SELECT * FROM credentials is a critical breach pattern."""
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="run_query",
            tool_input={"query": "SELECT * FROM credentials"},
            action_type=ActionType.READ,
        )
        result = scorer.score(action, profile)

        # base=0, +20 (SELECT *), +30 (credentials) → 50; clamp stays 50
        assert result.data_sensitivity == pytest.approx(50.0)
        assert result.reversibility == pytest.approx(0.0)   # READ, no DDL

    def test_alter_table_on_accounts(self, scorer):
        """
        ALTER TABLE accounts … :
          data_sensitivity = 0 + 30 (accounts) = 30
          reversibility    = 95 (DDL ALTER)
        """
        profile = make_profile(trust_score=50.0)
        action = make_action(
            tool_name="schema_migration",
            tool_input={"sql": "ALTER TABLE accounts ADD COLUMN frozen boolean DEFAULT false"},
            action_type=ActionType.WRITE,
        )
        result = scorer.score(action, profile)

        assert result.data_sensitivity == pytest.approx(30.0)
        assert result.reversibility == pytest.approx(95.0)
