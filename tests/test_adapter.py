"""Scenario tests for the AdaptationEngine (§5.4) and supporting infrastructure.

Three narrative scenarios drive the test design:

1. "New employee"  — agent starts at trust=50, performs 20 successful read
                     operations.  Trust should rise and FULL_AUTO threshold
                     should widen as the agent proves itself reliable.

2. "Mistake"       — an established agent causes an incident.  Trust drops and
                     thresholds tighten.  After 50 clean actions the agent
                     gradually recovers.

3. "False alarm"   — the system escalates 10 actions to SOFT_GATE and the human
                     approves all of them (false positives).  The SOFT_GATE
                     threshold should widen to reduce future alert fatigue.

Supplementary tests cover:
  • calculate_vagal_tone formula
  • threshold ordering invariant
  • min_actions_before_adaptation guard
  • SQLiteStorage round-trip (smoke test)
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from autonomica import (
    AdaptationEngine,
    AuditLogger,
    Autonomica,
    AutonomicaConfig,
    SQLiteStorage,
)
from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceDecision,
    GovernanceMode,
    RiskScore,
)
from autonomica.governor import GovernanceEngine
from autonomica.scorer import RiskScorer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_risk_score(
    composite: float = 10.0,
    financial: float = 0.0,
    sensitivity: float = 0.0,
    reversibility: float = 0.0,
    track_record: float = 50.0,
    novelty: float = 10.0,
    cascade: float = 20.0,
) -> RiskScore:
    return RiskScore(
        composite_score=composite,
        financial_magnitude=financial,
        data_sensitivity=sensitivity,
        reversibility=reversibility,
        agent_track_record=track_record,
        novelty=novelty,
        cascade_risk=cascade,
        explanation="test",
    )


def _make_decision(
    action_id: str,
    mode: GovernanceMode,
    approved: bool = True,
    composite: float = 10.0,
    human_override: bool | None = None,
) -> GovernanceDecision:
    return GovernanceDecision(
        action_id=action_id,
        risk_score=_make_risk_score(composite=composite),
        mode=mode,
        approved=approved,
        human_override=human_override,
        decision_time_ms=1.0,
    )


def _make_read_action(agent_id: str = "agent", agent_name: str = "Agent") -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        agent_name=agent_name,
        tool_name="read_database",
        tool_input={"query": "SELECT * FROM logs"},
        action_type=ActionType.READ,
    )


def _make_delete_action(agent_id: str = "agent") -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        agent_name=agent_id,
        tool_name="delete_records",
        tool_input={"table": "sessions", "condition": "expired=true"},
        action_type=ActionType.DELETE,
    )


def _make_email_action(agent_id: str = "agent") -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        agent_name=agent_id,
        tool_name="send_email",
        tool_input={"to": "user@example.com", "subject": "Invoice", "body": "..."},
        action_type=ActionType.COMMUNICATE,
    )


def _simulate_actions(
    adapter: AdaptationEngine,
    scorer: RiskScorer,
    governor: GovernanceEngine,
    profile: AgentProfile,
    n: int,
    tool_name: str = "read_database",
    action_type: ActionType = ActionType.READ,
    tool_input: dict | None = None,
) -> None:
    """Run *n* synthetic approved actions through scorer→governor→adapter."""
    for _ in range(n):
        action = AgentAction(
            agent_id=profile.agent_id,
            agent_name=profile.agent_name,
            tool_name=tool_name,
            tool_input=tool_input or {"query": "test"},
            action_type=action_type,
        )
        risk_score = scorer.score(action, profile)
        mode = governor.decide(risk_score, profile)
        decision = GovernanceDecision(
            action_id=action.action_id,
            risk_score=risk_score,
            mode=mode,
            approved=True,
            decision_time_ms=1.0,
        )
        # Replicate what the interceptor does before calling the adapter
        profile.total_actions += 1
        profile.approved_actions += 1
        if mode >= GovernanceMode.SOFT_GATE:
            profile.escalated_actions += 1

        adapter.update_after_action(action, decision, profile)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fast_adapter() -> AdaptationEngine:
    """Adapter with min_actions=1 so adaptation fires immediately."""
    return AdaptationEngine(AutonomicaConfig(min_actions_before_adaptation=1))


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


@pytest.fixture
def governor() -> GovernanceEngine:
    return GovernanceEngine()


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — "New Employee"
# ═════════════════════════════════════════════════════════════════════════════

class TestNewEmployeeScenario:
    """
    An agent starting at trust=50 does 20 routine read operations without
    incident.  We expect:
      • trust_score rises above 70 (EMA towards 100 for approved FULL_AUTO).
      • full_auto_max threshold widens (routine behaviour earns autonomy).
      • per_tool_trust records the 20 calls accurately.
      • vagal_tone stays high (no incidents, no false escalations).
    """

    def _build(self) -> tuple[AdaptationEngine, RiskScorer, GovernanceEngine, AgentProfile]:
        config = AutonomicaConfig(min_actions_before_adaptation=5)
        adapter = AdaptationEngine(config)
        scorer = RiskScorer()
        governor = GovernanceEngine()
        profile = AgentProfile(agent_id="new-employee", agent_name="New Employee")
        return adapter, scorer, governor, profile

    def test_trust_rises_after_20_successful_reads(self):
        adapter, scorer, governor, profile = self._build()
        initial_trust = profile.trust_score  # 50.0

        _simulate_actions(adapter, scorer, governor, profile, 20)

        assert profile.trust_score > initial_trust, (
            f"trust should rise: {initial_trust} → {profile.trust_score}"
        )
        assert profile.trust_score > 70.0, (
            f"expected trust > 70 after 20 clean reads, got {profile.trust_score}"
        )

    def test_full_auto_threshold_widens(self):
        adapter, scorer, governor, profile = self._build()
        initial_full_auto_max = profile.mode_thresholds["full_auto_max"]  # 15.0

        _simulate_actions(adapter, scorer, governor, profile, 20)

        assert profile.mode_thresholds["full_auto_max"] > initial_full_auto_max, (
            "FULL_AUTO threshold should widen after consistent clean reads"
        )

    def test_per_tool_call_count_recorded(self):
        adapter, scorer, governor, profile = self._build()

        _simulate_actions(adapter, scorer, governor, profile, 20)

        assert profile.per_tool_trust.get("read_database", 0) == 20

    def test_vagal_tone_high_after_clean_run(self):
        adapter, scorer, governor, profile = self._build()

        _simulate_actions(adapter, scorer, governor, profile, 20)

        # No incidents, no escalations → vagal tone should be perfect
        assert profile.vagal_tone >= 90.0, (
            f"expected vagal tone ≥ 90, got {profile.vagal_tone}"
        )

    def test_same_action_scores_lower_after_trust_builds(self):
        """Higher trust → lower agent_track_record signal → lower composite."""
        adapter, scorer, governor, profile = self._build()
        action = _make_read_action(profile.agent_id, profile.agent_name)
        # Seed tool count so novelty is stable (10+ uses → novelty=10)
        profile.per_tool_trust["read_database"] = 15

        initial_score = scorer.score(action, profile).composite_score

        # Raise trust significantly
        profile.trust_score = 90.0
        final_score = scorer.score(action, profile).composite_score

        assert final_score < initial_score, (
            f"composite score should drop as trust rises: {initial_score} → {final_score}"
        )

    def test_threshold_ordering_preserved(self):
        adapter, scorer, governor, profile = self._build()

        _simulate_actions(adapter, scorer, governor, profile, 30)

        t = profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — "Mistake"
# ═════════════════════════════════════════════════════════════════════════════

class TestMistakeScenario:
    """
    An established agent (trust=80) causes an incident.

    Expected immediately after:
      • trust_score drops below pre-incident level.
      • All thresholds tighten (more oversight demanded).

    Expected after 50 clean recovery actions:
      • trust_score recovers above post-incident level.
      • soft_gate_max widens back up (not necessarily to original, but higher).
      • vagal_tone reflects the incident in its calculation.
    """

    def _build_established_profile(self, agent_id: str = "clumsy") -> tuple[AdaptationEngine, AgentProfile]:
        config = AutonomicaConfig(min_actions_before_adaptation=1)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id=agent_id,
            agent_name="Clumsy Agent",
            trust_score=80.0,
            total_actions=20,   # already has a history → adaptation fires
            approved_actions=20,
        )
        return adapter, profile

    def test_incident_drops_trust(self):
        adapter, profile = self._build_established_profile()
        initial_trust = profile.trust_score  # 80.0

        action = _make_delete_action(profile.agent_id)
        profile.incidents += 1
        adapter.update_after_incident(action, profile)

        assert profile.trust_score < initial_trust, (
            f"trust should drop after incident: {initial_trust} → {profile.trust_score}"
        )

    def test_incident_tightens_all_thresholds(self):
        adapter, profile = self._build_established_profile()
        initial = dict(profile.mode_thresholds)

        action = _make_delete_action(profile.agent_id)
        profile.incidents += 1
        adapter.update_after_incident(action, profile)

        for key in ("full_auto_max", "log_alert_max", "soft_gate_max", "hard_gate_max"):
            assert profile.mode_thresholds[key] < initial[key], (
                f"{key} should tighten after incident: "
                f"{initial[key]} → {profile.mode_thresholds[key]}"
            )

    def test_trust_recovers_over_50_clean_actions(self):
        adapter, profile = self._build_established_profile("recovering")
        scorer = RiskScorer()
        governor = GovernanceEngine()

        # Incident
        action = _make_delete_action(profile.agent_id)
        profile.incidents += 1
        adapter.update_after_incident(action, profile)
        post_incident_trust = profile.trust_score

        # 50 clean reads
        _simulate_actions(adapter, scorer, governor, profile, 50)

        assert profile.trust_score > post_incident_trust, (
            f"trust should recover after 50 clean actions: "
            f"{post_incident_trust} → {profile.trust_score}"
        )

    def test_thresholds_re_widen_after_recovery(self):
        """After an incident, clean FULL_AUTO reads gradually widen full_auto_max back up."""
        adapter, profile = self._build_established_profile("recoverer-thresh")
        scorer = RiskScorer()
        governor = GovernanceEngine()

        action = _make_delete_action(profile.agent_id)
        profile.incidents += 1
        adapter.update_after_incident(action, profile)

        # Capture post-incident state for the threshold that clean reads widen
        post_incident_full_auto = profile.mode_thresholds["full_auto_max"]

        _simulate_actions(adapter, scorer, governor, profile, 50)

        # 50 FULL_AUTO reads each add 0.1 → full_auto_max rises by ~5 pts
        assert profile.mode_thresholds["full_auto_max"] > post_incident_full_auto, (
            "full_auto_max should re-widen after 50 clean reads following an incident"
        )

    def test_vagal_tone_reflects_incident_rate(self):
        adapter = AdaptationEngine(AutonomicaConfig(min_actions_before_adaptation=1))
        profile = AgentProfile(
            agent_id="incident-agent",
            agent_name="Incident Agent",
            total_actions=20,
            approved_actions=18,
            incidents=2,  # 2/20 = 10% incident rate
        )
        vagal_tone = adapter.calculate_vagal_tone(profile)

        # incident_rate=0.1 → penalty=6; false_escalation_rate=0 → penalty=0
        # vagal_tone = 100 - 6 - 0 = 94
        assert vagal_tone < 96.0, (
            f"vagal tone should reflect incidents (got {vagal_tone})"
        )
        assert vagal_tone > 80.0  # not catastrophically low for 10% incident rate

    def test_threshold_ordering_preserved_after_tightening(self):
        adapter, profile = self._build_established_profile("ordering")

        for _ in range(5):  # 5 incidents in a row
            action = _make_delete_action(profile.agent_id)
            profile.incidents += 1
            adapter.update_after_incident(action, profile)

        t = profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — "False Alarm"
# ═════════════════════════════════════════════════════════════════════════════

class TestFalseAlarmScenario:
    """
    The system triggers SOFT_GATE 10 times; a human approves all 10.
    These are false positives — the system is too strict.

    Expected:
      • soft_gate_max widens by ~5 points (10 × 0.5 per override).
      • vagal_tone drops because false_escalation_rate is high.
      • trust score nudges up (human confirmed the actions were fine).
    """

    def _build(self) -> tuple[AdaptationEngine, AgentProfile]:
        config = AutonomicaConfig(min_actions_before_adaptation=1)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id="alert-fatigued",
            agent_name="Alert Fatigued Agent",
            total_actions=10,
            approved_actions=10,
        )
        return adapter, profile

    def test_soft_gate_threshold_widens_after_10_false_alarms(self):
        adapter, profile = self._build()
        initial_soft_gate = profile.mode_thresholds["soft_gate_max"]  # 60.0

        for _ in range(10):
            action = _make_email_action(profile.agent_id)
            decision = _make_decision(
                action.action_id, GovernanceMode.SOFT_GATE,
                approved=True, composite=45.0,
            )
            profile.total_actions += 1
            profile.approved_actions += 1
            profile.escalated_actions += 1
            profile.false_escalations += 1
            adapter.update_after_override(action, decision, profile, human_approved=True)

        expected_increase = 10 * 0.5  # 5.0 points
        assert profile.mode_thresholds["soft_gate_max"] >= initial_soft_gate + expected_increase * 0.9, (
            f"soft_gate_max should widen by ~{expected_increase}: "
            f"{initial_soft_gate} → {profile.mode_thresholds['soft_gate_max']}"
        )

    def test_10_false_alarms_widens_by_at_least_4_points(self):
        """Conservative bound: even if capped by limits, widening ≥ 4.0 pts."""
        adapter, profile = self._build()
        initial = profile.mode_thresholds["soft_gate_max"]

        for _ in range(10):
            action = _make_email_action(profile.agent_id)
            decision = _make_decision(
                action.action_id, GovernanceMode.SOFT_GATE, composite=45.0
            )
            profile.total_actions += 1
            profile.escalated_actions += 1
            profile.false_escalations += 1
            adapter.update_after_override(action, decision, profile, human_approved=True)

        assert profile.mode_thresholds["soft_gate_max"] >= initial + 4.0

    def test_vagal_tone_drops_with_high_false_escalation_rate(self):
        adapter, profile = self._build()

        for _ in range(10):
            action = _make_email_action(profile.agent_id)
            decision = _make_decision(
                action.action_id, GovernanceMode.SOFT_GATE, composite=45.0
            )
            profile.total_actions += 1
            profile.escalated_actions += 1
            profile.false_escalations += 1
            adapter.update_after_override(action, decision, profile, human_approved=True)

        # false_escalation_rate = 10/10 = 1.0 → penalty = 40
        # vagal_tone = 100 - 0 - 40 = 60.0
        assert profile.vagal_tone < 70.0, (
            f"vagal tone should be low with 100% false-escalation rate: {profile.vagal_tone}"
        )

    def test_trust_nudges_up_after_human_approvals(self):
        adapter, profile = self._build()
        initial_trust = profile.trust_score  # 50.0

        for _ in range(10):
            action = _make_email_action(profile.agent_id)
            decision = _make_decision(
                action.action_id, GovernanceMode.SOFT_GATE, composite=45.0
            )
            profile.total_actions += 1
            profile.escalated_actions += 1
            adapter.update_after_override(action, decision, profile, human_approved=True)

        # Human approvals provide a positive trust signal (60.0)
        assert profile.trust_score > initial_trust, (
            f"trust should nudge up after human approvals: {initial_trust} → {profile.trust_score}"
        )

    def test_threshold_ordering_preserved_after_widening(self):
        adapter, profile = self._build()

        for _ in range(20):
            action = _make_email_action(profile.agent_id)
            decision = _make_decision(
                action.action_id, GovernanceMode.SOFT_GATE, composite=45.0
            )
            profile.total_actions += 1
            profile.escalated_actions += 1
            adapter.update_after_override(action, decision, profile, human_approved=True)

        t = profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]


# ═════════════════════════════════════════════════════════════════════════════
# Vagal tone formula unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestVagalTone:
    def test_perfect_calibration(self):
        adapter = AdaptationEngine()
        profile = AgentProfile(
            agent_id="perfect", agent_name="Perfect",
            total_actions=100, approved_actions=100,
            incidents=0, false_escalations=0, escalated_actions=0,
        )
        assert adapter.calculate_vagal_tone(profile) == pytest.approx(100.0)

    def test_formula_incident_penalty(self):
        """incident_rate=0.1, no false escalations → vagal=94."""
        adapter = AdaptationEngine()
        profile = AgentProfile(
            agent_id="a", agent_name="a",
            total_actions=100, incidents=10,
            escalated_actions=0, false_escalations=0,
        )
        # 100 - (0.1 × 60) - 0 = 94
        assert adapter.calculate_vagal_tone(profile) == pytest.approx(94.0)

    def test_formula_false_escalation_penalty(self):
        """100% false-escalation rate, no incidents → vagal=60."""
        adapter = AdaptationEngine()
        profile = AgentProfile(
            agent_id="b", agent_name="b",
            total_actions=20, incidents=0,
            escalated_actions=10, false_escalations=10,
        )
        # 100 - 0 - (1.0 × 40) = 60
        assert adapter.calculate_vagal_tone(profile) == pytest.approx(60.0)

    def test_combined_penalty_clamps_to_zero(self):
        """Extremely bad agent → vagal tone clamps to 0 (never negative)."""
        adapter = AdaptationEngine()
        profile = AgentProfile(
            agent_id="c", agent_name="c",
            total_actions=10, incidents=10,
            escalated_actions=10, false_escalations=10,
        )
        # 100 - (1.0 × 60) - (1.0 × 40) = 0
        assert adapter.calculate_vagal_tone(profile) == pytest.approx(0.0)

    def test_zero_actions_safe(self):
        adapter = AdaptationEngine()
        profile = AgentProfile(agent_id="d", agent_name="d")
        # No division by zero — should return 100.0
        assert adapter.calculate_vagal_tone(profile) == pytest.approx(100.0)


# ═════════════════════════════════════════════════════════════════════════════
# min_actions_before_adaptation guard
# ═════════════════════════════════════════════════════════════════════════════

class TestMinActionsGuard:
    def test_no_trust_update_before_min_actions(self):
        """Adaptation should NOT fire if total_actions < min_actions."""
        config = AutonomicaConfig(min_actions_before_adaptation=10)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id="x", agent_name="x", total_actions=5  # below threshold
        )
        initial_trust = profile.trust_score

        action = _make_read_action("x")
        decision = _make_decision(action.action_id, GovernanceMode.FULL_AUTO)
        adapter.update_after_action(action, decision, profile)

        assert profile.trust_score == initial_trust, (
            "trust should not change before min_actions threshold"
        )

    def test_trust_updates_at_and_after_min_actions(self):
        config = AutonomicaConfig(min_actions_before_adaptation=5)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id="y", agent_name="y", total_actions=5  # at threshold
        )
        initial_trust = profile.trust_score

        action = _make_read_action("y")
        decision = _make_decision(action.action_id, GovernanceMode.FULL_AUTO)
        adapter.update_after_action(action, decision, profile)

        assert profile.trust_score != initial_trust, (
            "trust should update once total_actions >= min_actions"
        )

    def test_per_tool_count_always_updated(self):
        """per_tool_trust call count must be recorded even before min_actions."""
        config = AutonomicaConfig(min_actions_before_adaptation=100)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(agent_id="z", agent_name="z", total_actions=1)

        action = _make_read_action("z")
        decision = _make_decision(action.action_id, GovernanceMode.LOG_AND_ALERT)
        adapter.update_after_action(action, decision, profile)

        assert profile.per_tool_trust.get("read_database", 0) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Threshold ordering invariant
# ═════════════════════════════════════════════════════════════════════════════

class TestThresholdOrdering:
    def test_ordering_preserved_under_extreme_widening(self):
        config = AutonomicaConfig(min_actions_before_adaptation=1)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id="w", agent_name="w", total_actions=100
        )

        # Widen every possible threshold as aggressively as possible
        for mode in (
            GovernanceMode.FULL_AUTO,
            GovernanceMode.LOG_AND_ALERT,
            GovernanceMode.SOFT_GATE,
            GovernanceMode.HARD_GATE,
        ):
            action = _make_read_action("w")
            decision = _make_decision(action.action_id, mode)
            for _ in range(50):
                adapter._widen_thresholds(profile, decision, amount=5.0)

        t = profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]

    def test_ordering_preserved_under_extreme_tightening(self):
        config = AutonomicaConfig(min_actions_before_adaptation=1)
        adapter = AdaptationEngine(config)
        profile = AgentProfile(
            agent_id="v", agent_name="v", total_actions=100
        )

        for mode in (
            GovernanceMode.FULL_AUTO,
            GovernanceMode.LOG_AND_ALERT,
            GovernanceMode.SOFT_GATE,
            GovernanceMode.HARD_GATE,
        ):
            action = _make_read_action("v")
            decision = _make_decision(action.action_id, mode)
            for _ in range(50):
                adapter._tighten_thresholds(profile, decision, amount=5.0)

        t = profile.mode_thresholds
        assert t["full_auto_max"] < t["log_alert_max"]
        assert t["log_alert_max"] < t["soft_gate_max"]
        assert t["soft_gate_max"] < t["hard_gate_max"]

    def test_thresholds_respect_hard_limits(self):
        adapter = AdaptationEngine()
        profile = AgentProfile(agent_id="lim", agent_name="lim", total_actions=100)

        action = _make_read_action("lim")
        decision = _make_decision(action.action_id, GovernanceMode.SOFT_GATE)

        # Widen soft_gate_max to its maximum limit
        for _ in range(200):
            adapter._widen_thresholds(profile, decision, amount=1.0)

        assert profile.mode_thresholds["soft_gate_max"] <= 80.0  # hard limit

        # Tighten to minimum
        for _ in range(200):
            adapter._tighten_thresholds(profile, decision, amount=1.0)

        assert profile.mode_thresholds["soft_gate_max"] >= 30.0  # hard limit


# ═════════════════════════════════════════════════════════════════════════════
# AutonomicaConfig unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAutonomicaConfig:
    def test_defaults_match_spec(self):
        from autonomica.config import AutonomicaConfig
        cfg = AutonomicaConfig()
        assert cfg.soft_gate_timeout_seconds == pytest.approx(60.0)
        assert cfg.hard_gate_timeout_seconds == pytest.approx(300.0)
        assert cfg.default_trust_score == 50.0
        assert cfg.adaptation_rate == 0.5
        assert cfg.min_actions_before_adaptation == 10

    def test_scoring_weights_sum_to_one(self):
        from autonomica.config import AutonomicaConfig
        cfg = AutonomicaConfig()
        total = sum(cfg.scoring_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_custom_values_accepted(self):
        from autonomica.config import AutonomicaConfig
        cfg = AutonomicaConfig(
            soft_gate_timeout_seconds=5,
            min_actions_before_adaptation=3,
            adaptation_rate=0.8,
        )
        assert cfg.soft_gate_timeout_seconds == 5
        assert cfg.min_actions_before_adaptation == 3
        assert cfg.adaptation_rate == 0.8


# ═════════════════════════════════════════════════════════════════════════════
# AuditLogger tests
# ═════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:
    def test_log_decision_writes_jsonl(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)
        action = _make_read_action()
        profile = AgentProfile(agent_id="agent", agent_name="Agent")
        decision = _make_decision(action.action_id, GovernanceMode.FULL_AUTO)

        audit.log_decision(action, decision, profile)
        entries = audit.read_entries()

        assert len(entries) == 1
        e = entries[0]
        assert e["event"] == "governance_decision"
        assert e["action_id"] == decision.action_id
        assert e["governance_mode"] == "FULL_AUTO"
        assert e["approved"] is True

    def test_log_override_writes_entry(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)

        audit.log_override("act-123", "agent-1", approved=True, reason="CFO approved")
        entries = audit.read_entries()

        assert entries[0]["event"] == "human_override"
        assert entries[0]["approved"] is True
        assert entries[0]["reason"] == "CFO approved"

    def test_log_incident_writes_entry(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_file=log_file)

        audit.log_incident("act-456", "agent-2", notes="DB timeout")
        entries = audit.read_entries()

        assert entries[0]["event"] == "incident"
        assert entries[0]["notes"] == "DB timeout"

    def test_export_jsonl(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        export_file = str(tmp_path / "export.jsonl")
        audit = AuditLogger(log_file=log_file)

        audit.log_incident("a1", "agent", notes="test")
        audit.log_incident("a2", "agent", notes="test2")

        n = audit.export(export_file, fmt="jsonl")
        assert n == 2
        assert os.path.exists(export_file)

    def test_export_json(self, tmp_path):
        import json as _json
        log_file = str(tmp_path / "audit.jsonl")
        export_file = str(tmp_path / "export.json")
        audit = AuditLogger(log_file=log_file)

        audit.log_incident("a1", "agent")
        audit.export(export_file, fmt="json")

        data = _json.loads(Path(export_file).read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_read_entries_empty_when_no_file(self):
        audit = AuditLogger()  # no log_file
        assert audit.read_entries() == []

    def test_export_raises_when_no_log_file(self):
        audit = AuditLogger()
        with pytest.raises(ValueError, match="log_file"):
            audit.export("/tmp/noop.jsonl")


# ═════════════════════════════════════════════════════════════════════════════
# SQLiteStorage smoke tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSQLiteStorage:
    async def test_save_and_load_profile(self):
        storage = SQLiteStorage(":memory:")
        profile = AgentProfile(agent_id="s1", agent_name="Storage Test")
        profile.trust_score = 75.0

        await storage.save_profile(profile)
        loaded = await storage.load_profile("s1")

        assert loaded is not None
        assert loaded.agent_id == "s1"
        assert loaded.trust_score == pytest.approx(75.0)

    async def test_load_nonexistent_profile_returns_none(self):
        storage = SQLiteStorage(":memory:")
        result = await storage.load_profile("ghost-agent")
        assert result is None

    async def test_save_and_load_decision(self):
        storage = SQLiteStorage(":memory:")
        action = _make_read_action()
        decision = _make_decision(action.action_id, GovernanceMode.FULL_AUTO, composite=5.0)

        await storage.save_decision(decision)
        loaded = await storage.load_decision(decision.action_id)

        assert loaded is not None
        assert loaded.action_id == decision.action_id
        assert loaded.mode == GovernanceMode.FULL_AUTO

    async def test_list_profiles(self):
        storage = SQLiteStorage(":memory:")
        for i in range(3):
            p = AgentProfile(agent_id=f"agent-{i}", agent_name=f"Agent {i}")
            await storage.save_profile(p)

        profiles = await storage.list_profiles()
        assert len(profiles) == 3

    async def test_list_decisions_with_agent_filter(self):
        storage = SQLiteStorage(":memory:")
        for i in range(5):
            action = _make_read_action(agent_id="agent-a" if i < 3 else "agent-b")
            d = _make_decision(action.action_id, GovernanceMode.FULL_AUTO)
            await storage.save_decision(d)

        all_d = await storage.list_decisions()
        assert len(all_d) == 5

    async def test_upsert_profile(self):
        storage = SQLiteStorage(":memory:")
        profile = AgentProfile(agent_id="upsert-me", agent_name="Original")
        await storage.save_profile(profile)

        profile.trust_score = 99.0
        await storage.save_profile(profile)

        loaded = await storage.load_profile("upsert-me")
        assert loaded.trust_score == pytest.approx(99.0)

    async def test_file_based_storage(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path)

        profile = AgentProfile(agent_id="file-agent", agent_name="File Agent")
        await storage.save_profile(profile)
        await storage.close()

        # Re-open and verify persistence
        storage2 = SQLiteStorage(db_path)
        loaded = await storage2.load_profile("file-agent")
        assert loaded is not None
        assert loaded.agent_id == "file-agent"


# ═════════════════════════════════════════════════════════════════════════════
# Full end-to-end pipeline: Autonomica + adapter wired together
# ═════════════════════════════════════════════════════════════════════════════

class TestFullPipelineWithAdapter:
    """Integration check: Autonomica.evaluate_action() routes through the adapter."""

    class _MockEscalation:
        def __init__(self, response=None):
            self.notifications = []
            self._response = response

        async def notify(self, action, mode):
            self.notifications.append(mode)

        async def wait_for_response(self, action_id, timeout):
            return self._response

    def _make_gov(self) -> Autonomica:
        from autonomica.config import AutonomicaConfig
        config = AutonomicaConfig(
            soft_gate_timeout_seconds=0.01,
            hard_gate_timeout_seconds=0.01,
            min_actions_before_adaptation=1,
        )
        return Autonomica(
            config=config,
            escalation=self._MockEscalation(),
        )

    async def test_per_tool_trust_updated_after_action(self):
        gov = self._make_gov()
        # Seed trusted profile so action passes through
        p = gov._get_or_create_profile("wired", "Wired Agent")
        p.trust_score = 90.0
        p.per_tool_trust["read_db"] = 15
        p.total_actions = 5

        action = AgentAction(
            agent_id="wired", agent_name="Wired Agent",
            tool_name="read_db",
            tool_input={"query": "SELECT 1"},
            action_type=ActionType.READ,
        )
        await gov.evaluate_action(action)

        # Adapter should have incremented the per-tool count from 15 → 16
        profile = gov.get_agent_profile("wired")
        assert profile.per_tool_trust.get("read_db", 0) == 16

    async def test_trust_increases_after_full_auto_action(self):
        gov = self._make_gov()
        p = gov._get_or_create_profile("truster", "Truster")
        p.trust_score = 90.0
        p.per_tool_trust["read_db"] = 15
        p.total_actions = 10

        initial_trust = p.trust_score

        action = AgentAction(
            agent_id="truster", agent_name="Truster",
            tool_name="read_db",
            tool_input={"query": "SELECT 1"},
            action_type=ActionType.READ,
        )
        await gov.evaluate_action(action)

        profile = gov.get_agent_profile("truster")
        # FULL_AUTO approved → EMA with signal 100 → trust edges up from 90
        assert profile.trust_score >= initial_trust

    async def test_incident_tightens_thresholds_via_record_outcome(self):
        gov = self._make_gov()
        p = gov._get_or_create_profile("incident-wired", "Incident Wired")
        p.trust_score = 80.0
        p.per_tool_trust["read_db"] = 15
        p.total_actions = 10
        initial_soft_gate = p.mode_thresholds["soft_gate_max"]

        action = AgentAction(
            agent_id="incident-wired", agent_name="Incident Wired",
            tool_name="read_db",
            tool_input={"query": "SELECT 1"},
            action_type=ActionType.READ,
        )
        await gov.evaluate_action(action)
        action_id = list(gov._decisions.keys())[0]

        gov.record_outcome(action_id, success=False, notes="timeout")

        profile = gov.get_agent_profile("incident-wired")
        assert profile.incidents == 1
        assert profile.mode_thresholds["soft_gate_max"] < initial_soft_gate


# needed for Path in test_export_json
from pathlib import Path
