"""Risk Scorer — evaluates the risk of an agent action.

The 'amygdala' of the system. Uses heuristic scoring (no ML).
Each signal produces a 0-100 sub-score; the composite is a weighted average.
Must complete in <10ms.
"""
from __future__ import annotations

import time
from typing import Any

from autonomica.models import ActionType, AgentAction, AgentProfile, RiskScore

# ---------------------------------------------------------------------------
# Financial amount field names we scan for
# ---------------------------------------------------------------------------
_AMOUNT_KEYS: frozenset[str] = frozenset({
    "amount", "value", "price", "total", "cost", "fee",
    "payment", "balance", "sum", "charge", "invoice_amount",
    "transaction_amount",
})

# ---------------------------------------------------------------------------
# Default sensitivity keyword lists (mirrors AutonomicaConfig defaults)
# ---------------------------------------------------------------------------
_DEFAULT_PII_PATTERNS: tuple[str, ...] = (
    "email", "ssn", "phone", "credit_card", "address", "dob",
)
_DEFAULT_FINANCIAL_DATA_PATTERNS: tuple[str, ...] = (
    "account_number", "routing", "balance", "salary",
)
_DEFAULT_HEALTH_DATA_PATTERNS: tuple[str, ...] = (
    "diagnosis", "medication", "patient", "medical",
)

# ---------------------------------------------------------------------------
# Reversibility scores per ActionType (spec table)
# ---------------------------------------------------------------------------
_REVERSIBILITY_SCORES: dict[ActionType, float] = {
    ActionType.READ: 0.0,
    ActionType.WRITE: 30.0,
    ActionType.COMMUNICATE: 60.0,
    ActionType.FINANCIAL: 70.0,
    ActionType.DELETE: 80.0,
}


class RiskScorer:
    """Evaluates the risk of an agent action. The 'amygdala' of the system."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "financial_magnitude": 0.25,
        "data_sensitivity": 0.20,
        "reversibility": 0.20,
        "agent_track_record": 0.15,
        "novelty": 0.10,
        "cascade_risk": 0.10,
    }

    def __init__(self, config: Any = None) -> None:
        """
        Args:
            config: Optional AutonomicaConfig. When None, defaults are used.
                    Accepted fields: scoring_weights, pii_patterns,
                    financial_data_patterns, health_data_patterns.
        """
        if config is not None:
            self._weights = getattr(config, "scoring_weights", self.DEFAULT_WEIGHTS)
            self._pii = tuple(getattr(config, "pii_patterns", _DEFAULT_PII_PATTERNS))
            self._financial_data = tuple(
                getattr(config, "financial_data_patterns", _DEFAULT_FINANCIAL_DATA_PATTERNS)
            )
            self._health_data = tuple(
                getattr(config, "health_data_patterns", _DEFAULT_HEALTH_DATA_PATTERNS)
            )
            self._financial_thresholds = getattr(
                config, "financial_thresholds",
                {"low": 100.0, "medium": 10_000.0, "high": 100_000.0, "critical": 1_000_000.0},
            )
        else:
            self._weights = self.DEFAULT_WEIGHTS
            self._pii = _DEFAULT_PII_PATTERNS
            self._financial_data = _DEFAULT_FINANCIAL_DATA_PATTERNS
            self._health_data = _DEFAULT_HEALTH_DATA_PATTERNS
            self._financial_thresholds = {
                "low": 100.0,
                "medium": 10_000.0,
                "high": 100_000.0,
                "critical": 1_000_000.0,
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, action: AgentAction, agent_profile: AgentProfile) -> RiskScore:
        """Score an action against an agent profile. Completes in <10ms."""
        financial = self._score_financial_magnitude(action)
        sensitivity = self._score_data_sensitivity(action)
        reversibility = self._score_reversibility(action)
        track_record = self._score_track_record(agent_profile)
        novelty = self._score_novelty(action, agent_profile)
        cascade = self._score_cascade_risk(action)

        composite = (
            financial * self._weights["financial_magnitude"]
            + sensitivity * self._weights["data_sensitivity"]
            + reversibility * self._weights["reversibility"]
            + track_record * self._weights["agent_track_record"]
            + novelty * self._weights["novelty"]
            + cascade * self._weights["cascade_risk"]
        )
        composite = round(min(max(composite, 0.0), 100.0), 2)

        explanation = self._build_explanation(
            action, agent_profile,
            financial, sensitivity, reversibility, track_record, novelty, cascade, composite,
        )

        return RiskScore(
            composite_score=composite,
            financial_magnitude=financial,
            data_sensitivity=sensitivity,
            reversibility=reversibility,
            agent_track_record=track_record,
            novelty=novelty,
            cascade_risk=cascade,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Individual signal scorers
    # ------------------------------------------------------------------

    def _score_financial_magnitude(self, action: AgentAction) -> float:
        """Parse tool_input for amount-like fields and return a risk score."""
        max_amount = 0.0
        for key, val in action.tool_input.items():
            if key.lower() in _AMOUNT_KEYS and isinstance(val, (int, float)):
                max_amount = max(max_amount, abs(float(val)))

        t = self._financial_thresholds
        if max_amount >= t["critical"]:    # $1M+
            return 100.0
        if max_amount >= t["high"]:        # $100K–$1M
            return 80.0
        if max_amount >= t["medium"]:      # $10K–$100K
            return 50.0
        if max_amount >= t["low"]:         # $100–$10K
            return 20.0
        return 0.0                         # $0–$100

    def _score_data_sensitivity(self, action: AgentAction) -> float:
        """Keyword scan of all tool_input keys and string values."""
        text = self._flatten_input(action.tool_input)

        # Highest tier wins
        if any(p in text for p in self._health_data):
            return 80.0
        if any(p in text for p in self._financial_data):
            return 70.0
        if any(p in text for p in self._pii):
            return 60.0
        return 0.0

    def _score_reversibility(self, action: AgentAction) -> float:
        """Map ActionType to a static reversibility score."""
        return _REVERSIBILITY_SCORES.get(action.action_type, 0.0)

    def _score_track_record(self, agent_profile: AgentProfile) -> float:
        """100 - trust_score. Trusted agents get lower risk on this signal."""
        return round(100.0 - agent_profile.trust_score, 2)

    def _score_novelty(self, action: AgentAction, agent_profile: AgentProfile) -> float:
        """
        Check how many times this agent has called this tool.

        per_tool_trust stores call counts for novelty scoring:
          - Tool not in dict (first call) → 70
          - Call count < 10               → 40
          - Call count >= 10              → 10
        """
        call_count = agent_profile.per_tool_trust.get(action.tool_name)
        if call_count is None:
            return 70.0
        if call_count < 10:
            return 40.0
        return 10.0

    def _score_cascade_risk(self, action: AgentAction) -> float:
        """
        MVP: default 20. If metadata contains 'cascade_downstream_agents' (int),
        use min(N * 15, 100).
        """
        n = action.metadata.get("cascade_downstream_agents")
        if n is not None:
            return min(float(n) * 15.0, 100.0)
        return 20.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_input(tool_input: dict) -> str:
        """Produce a lowercase string of all keys and string values for scanning."""
        parts: list[str] = []
        for key, val in tool_input.items():
            parts.append(key.lower())
            if isinstance(val, str):
                parts.append(val.lower())
            elif isinstance(val, dict):
                parts.append(RiskScorer._flatten_input(val))
        return " ".join(parts)

    def _build_explanation(
        self,
        action: AgentAction,
        agent_profile: AgentProfile,
        financial: float,
        sensitivity: float,
        reversibility: float,
        track_record: float,
        novelty: float,
        cascade: float,
        composite: float,
    ) -> str:
        parts = [
            f"Composite score: {composite:.1f}/100",
            f"  financial_magnitude={financial:.0f} "
            f"(weight {self._weights['financial_magnitude']:.0%})",
            f"  data_sensitivity={sensitivity:.0f} "
            f"(weight {self._weights['data_sensitivity']:.0%})",
            f"  reversibility={reversibility:.0f} "
            f"[{action.action_type.value}] "
            f"(weight {self._weights['reversibility']:.0%})",
            f"  agent_track_record={track_record:.0f} "
            f"[trust_score={agent_profile.trust_score:.0f}] "
            f"(weight {self._weights['agent_track_record']:.0%})",
            f"  novelty={novelty:.0f} "
            f"[tool={action.tool_name!r}] "
            f"(weight {self._weights['novelty']:.0%})",
            f"  cascade_risk={cascade:.0f} "
            f"(weight {self._weights['cascade_risk']:.0%})",
        ]
        return "\n".join(parts)
