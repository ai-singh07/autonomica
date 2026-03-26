"""Adaptation Engine — learns from governance decisions over time (§5.4).

The 'vagal tone' calibrator.  After every decision it:
  • Records per-tool call counts (feeds the novelty scorer).
  • Updates the agent's trust score via an exponential moving average (EMA).
  • Gradually widens thresholds when the system is over-escalating.
  • Tightens thresholds after human rejections or incidents.
  • Re-calculates vagal tone (calibration quality metric).

Threshold adaptation only kicks in after ``min_actions_before_adaptation``
actions have been observed to prevent wild swings from early noise.

Threshold limits
----------------
Each threshold has hard lower/upper bounds so they can never drift into
meaningless territory.  The ordering invariant ``full_auto_max <
log_alert_max < soft_gate_max < hard_gate_max`` is enforced after every
adjustment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from autonomica.models import ActionType, AgentAction, AgentProfile, GovernanceDecision, GovernanceMode

# ── Per-threshold (min, max) bounds ───────────────────────────────────────────
_THRESHOLD_LIMITS: dict[str, tuple[float, float]] = {
    "full_auto_max": (1.0, 35.0),
    "log_alert_max": (10.0, 55.0),
    "soft_gate_max": (30.0, 80.0),
    "hard_gate_max": (55.0, 98.0),
}

# Ordered from least to most restrictive (used for ordering enforcement)
_THRESHOLD_KEYS: tuple[str, ...] = (
    "full_auto_max",
    "log_alert_max",
    "soft_gate_max",
    "hard_gate_max",
)

# Map from governance mode to the threshold that upper-bounds that mode
_MODE_TO_THRESHOLD: dict[GovernanceMode, str] = {
    GovernanceMode.FULL_AUTO: "full_auto_max",
    GovernanceMode.LOG_AND_ALERT: "log_alert_max",
    GovernanceMode.SOFT_GATE: "soft_gate_max",
    GovernanceMode.HARD_GATE: "hard_gate_max",
    # QUARANTINE has no threshold to adjust
}

# Fallback: when no governance decision is available, infer the relevant
# threshold from the action type alone.
_ACTION_TYPE_TO_THRESHOLD: dict[ActionType, str] = {
    ActionType.READ:        "full_auto_max",
    ActionType.WRITE:       "log_alert_max",
    ActionType.COMMUNICATE: "soft_gate_max",
    ActionType.DELETE:      "hard_gate_max",
    ActionType.FINANCIAL:   "hard_gate_max",
}

# Base penalty magnitude fed into the dampened incident formula.
# Actual penalty = _INCIDENT_BASE_PENALTY × (1 − trust_score / 200)
# Examples:  trust=90 → 1.0 × 0.55 = 0.55
#            trust=50 → 1.0 × 0.75 = 0.75
_INCIDENT_BASE_PENALTY: float = 1.0

# Trust-signal by approved mode (what value to feed into the EMA)
_APPROVE_SIGNAL: dict[GovernanceMode, float] = {
    GovernanceMode.FULL_AUTO: 100.0,
    GovernanceMode.LOG_AND_ALERT: 85.0,
    GovernanceMode.SOFT_GATE: 70.0,
    GovernanceMode.HARD_GATE: 80.0,
}

# Minimum gap maintained between adjacent thresholds
_MIN_THRESHOLD_GAP: float = 2.0

# How much a single FULL_AUTO/LOG_AND_ALERT approval widens that threshold.
# Small so a run of 10 clean actions opens thresholds by ~1 point.
_AUTO_WIDEN_PER_ACTION: float = 0.1


class AdaptationEngine:
    """Learns from governance decisions over time. The 'vagal tone' calibrator."""

    def __init__(self, config: Any = None) -> None:
        """
        Args:
            config: Optional AutonomicaConfig (or any object with matching
                    attributes).  Falls back to spec defaults when None.
        """
        if config is not None:
            self._enabled = bool(getattr(config, "adaptation_enabled", False))
            self._adaptation_rate = float(
                getattr(config, "adaptation_rate", 0.5)
            )
            self._min_actions = int(
                getattr(config, "min_actions_before_adaptation", 10)
            )
        else:
            self._enabled = False
            self._adaptation_rate = 0.5
            self._min_actions = 10

        # EMA alpha for trust score.
        # adaptation_rate = 0.5 → alpha = 0.05 (slow, stable drift).
        # adaptation_rate = 1.0 → alpha = 0.10 (faster response).
        self._trust_alpha: float = self._adaptation_rate / 10.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_after_action(
        self,
        action: AgentAction,
        decision: GovernanceDecision,
        profile: AgentProfile,
    ) -> None:
        """Called immediately after every governance decision.

        Updates:
          1. Per-tool call count (novelty scoring input for next action).
          2. Trust score EMA for non-gated approvals.
          3. Slow threshold widening for consistently routine behaviour.
          4. Vagal tone.

        The interceptor is responsible for the basic counters
        (total_actions, approved_actions, escalated_actions); this method
        handles the adaptive part only.
        """
        # 1. Record that this agent used this tool (novelty counter).
        #    Always runs regardless of adaptation_enabled so the novelty
        #    scorer sees accurate call counts.
        current_count = profile.per_tool_trust.get(action.tool_name, 0)
        profile.per_tool_trust[action.tool_name] = current_count + 1

        # When adaptation is disabled, skip all threshold/trust mutations.
        if not self._enabled:
            profile.vagal_tone = self.calculate_vagal_tone(profile)
            profile.updated_at = datetime.now(timezone.utc)
            return

        # Only adapt after enough observations
        if profile.total_actions < self._min_actions:
            profile.vagal_tone = self.calculate_vagal_tone(profile)
            profile.updated_at = datetime.now(timezone.utc)
            return

        # 2. Trust EMA for approved, non-gated actions
        if decision.approved and decision.mode <= GovernanceMode.LOG_AND_ALERT:
            signal = _APPROVE_SIGNAL.get(decision.mode, 85.0)
            self._update_trust_ema(profile, signal)

        # 3. Slow threshold widening for clean FULL_AUTO / LOG_AND_ALERT actions
        #    Rationale: consistent good behaviour earns gradual autonomy.
        if decision.approved and decision.mode <= GovernanceMode.LOG_AND_ALERT:
            key = _MODE_TO_THRESHOLD.get(decision.mode)
            if key:
                self._adjust_threshold(profile, key, +_AUTO_WIDEN_PER_ACTION)

        # 4. Recalculate vagal tone
        profile.vagal_tone = self.calculate_vagal_tone(profile)
        profile.updated_at = datetime.now(timezone.utc)

    def update_after_override(
        self,
        action: AgentAction,
        decision: GovernanceDecision,
        profile: AgentProfile,
        human_approved: bool,
    ) -> None:
        """Called when a human responds to a SOFT_GATE or HARD_GATE.

        ``human_approved=True``  → false escalation: system was too strict
                                   → widen that mode's threshold.
        ``human_approved=False`` → human veto: system was right to escalate
                                   (or possibly too lenient on gate size)
                                   → tighten that mode's threshold.

        No-op when ``adaptation_enabled=False``.
        """
        if not self._enabled:
            profile.vagal_tone = self.calculate_vagal_tone(profile)
            profile.updated_at = datetime.now(timezone.utc)
            return

        if human_approved:
            # False escalation — widen the triggered mode's threshold
            self._widen_thresholds(profile, decision, amount=0.5)
            if profile.total_actions >= self._min_actions:
                # A human approval is a positive signal, but weaker than
                # an un-gated success (the agent could have been luckier).
                self._update_trust_ema(profile, 60.0)
        else:
            # Human rejected — tighten
            self._tighten_thresholds(profile, decision, amount=1.0)
            if profile.total_actions >= self._min_actions:
                self._update_trust_ema(profile, 20.0)

        profile.vagal_tone = self.calculate_vagal_tone(profile)
        profile.updated_at = datetime.now(timezone.utc)

    def update_after_incident(
        self,
        action: AgentAction,
        profile: AgentProfile,
        decision: GovernanceDecision | None = None,
    ) -> None:
        """Called when a post-action failure is recorded (record_outcome false).

        Applies a trust penalty and tightens only the RELEVANT threshold using
        a dampened, trust-proportional formula:

            actual_penalty = base_penalty × (1 − trust_score / 200)

        A trusted agent (trust=90) receives a smaller penalty (≈0.55) than a
        new agent (trust=50, penalty ≈0.75), because a reliable agent's single
        error is statistically less alarming.  Only the threshold that governs
        the mode where the incident occurred is tightened; unrelated thresholds
        are left untouched to prevent the yo-yo effect where a DELETE incident
        needlessly tightens the full_auto_max for routine reads.
        """
        if not self._enabled or profile.total_actions < self._min_actions:
            profile.vagal_tone = self.calculate_vagal_tone(profile)
            profile.updated_at = datetime.now(timezone.utc)
            return

        # Determine which threshold is relevant for this incident.
        # Prefer the mode from the governance decision (most accurate); fall
        # back to the action type when no decision is available (e.g. tests).
        relevant_key: str | None = None
        if decision is not None:
            relevant_key = _MODE_TO_THRESHOLD.get(decision.mode)
        if relevant_key is None:
            # QUARANTINE has no threshold entry in _MODE_TO_THRESHOLD, or
            # decision was not supplied — infer from action type.
            relevant_key = _ACTION_TYPE_TO_THRESHOLD.get(
                action.action_type, "hard_gate_max"
            )

        # Dampened penalty: proportional to how much the system still distrusts
        # the agent.  High-trust agents get a lighter tap; new agents get more.
        actual_penalty = _INCIDENT_BASE_PENALTY * (1.0 - profile.trust_score / 200.0)
        self._adjust_threshold(profile, relevant_key, -actual_penalty)

        # Trust penalty: incident signal = 0 (large, but smoothed by EMA)
        self._update_trust_ema(profile, 0.0)

        profile.vagal_tone = self.calculate_vagal_tone(profile)
        profile.updated_at = datetime.now(timezone.utc)

    def calculate_vagal_tone(self, profile: AgentProfile) -> float:
        """Measure how well-calibrated this agent's governance is.

        Formula (from spec §5.4)::

            100 - (incident_rate × 60) - (false_escalation_rate × 40)

        Perfect vagal tone (100): zero incidents AND zero false escalations.
        Low vagal tone: either too many incidents (system too loose) or too
        many unnecessary escalations (system too tight).
        """
        total = max(profile.total_actions, 1)
        incident_rate = profile.incidents / total
        false_escalation_rate = profile.false_escalations / max(
            profile.escalated_actions, 1
        )

        vagal_tone = (
            100.0
            - (incident_rate * 60.0)
            - (false_escalation_rate * 40.0)
        )
        return round(max(0.0, min(100.0, vagal_tone)), 2)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _widen_thresholds(
        self,
        profile: AgentProfile,
        decision: GovernanceDecision,
        amount: float = 0.5,
    ) -> None:
        """Increase autonomy: push the triggered mode's upper bound up."""
        key = _MODE_TO_THRESHOLD.get(decision.mode)
        if key:
            self._adjust_threshold(profile, key, +amount)

    def _tighten_thresholds(
        self,
        profile: AgentProfile,
        decision: GovernanceDecision,
        amount: float = 1.0,
    ) -> None:
        """Decrease autonomy: push the triggered mode's upper bound down."""
        key = _MODE_TO_THRESHOLD.get(decision.mode)
        if key:
            self._adjust_threshold(profile, key, -amount)

    def _adjust_threshold(
        self,
        profile: AgentProfile,
        key: str,
        delta: float,
    ) -> None:
        """Apply delta to a threshold, clamp to limits, then re-enforce ordering."""
        lo, hi = _THRESHOLD_LIMITS[key]
        current = profile.mode_thresholds[key]
        profile.mode_thresholds[key] = round(
            max(lo, min(hi, current + delta)), 2
        )
        self._enforce_ordering(profile)

    def _enforce_ordering(self, profile: AgentProfile) -> None:
        """Ensure full_auto_max < log_alert_max < soft_gate_max < hard_gate_max.

        If an adjustment causes a violation, the violating threshold is pushed
        up (capped at its own maximum) to restore the invariant.
        """
        t = profile.mode_thresholds
        for i in range(1, len(_THRESHOLD_KEYS)):
            prev_key = _THRESHOLD_KEYS[i - 1]
            curr_key = _THRESHOLD_KEYS[i]
            min_needed = t[prev_key] + _MIN_THRESHOLD_GAP
            if t[curr_key] < min_needed:
                _, hi = _THRESHOLD_LIMITS[curr_key]
                t[curr_key] = round(min(min_needed, hi), 2)

    def _update_trust_ema(self, profile: AgentProfile, signal: float) -> None:
        """Exponential moving average update for trust score.

        trust_new = alpha × signal + (1 − alpha) × trust_old

        Where alpha = adaptation_rate / 10  (default 0.05).
        This gives slow, stable drift so a single action can't flip the score.
        """
        alpha = self._trust_alpha
        new_trust = alpha * signal + (1.0 - alpha) * profile.trust_score
        profile.trust_score = round(max(0.0, min(100.0, new_trust)), 2)
