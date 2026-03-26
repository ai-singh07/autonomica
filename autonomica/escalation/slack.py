"""Slack webhook escalation — posts rich governance alerts to a Slack channel.

Sends a colour-coded Slack message for every governance event.
``wait_for_response`` returns None immediately in MVP: the human operator
responds via the dashboard API (``POST /api/governance/override``), which
calls ``record_human_override()`` and resolves the pending gate Future.

Colour scheme
-------------
FULL_AUTO    →  #4CAF50  green   (informational)
LOG_AND_ALERT → #2196F3  blue    (informational)
SOFT_GATE    →  #FF9800  amber   (warning  — proceed unless vetoed)
HARD_GATE    →  #F44336  red     (danger   — blocked until approved)
QUARANTINE   →  #9C27B0  purple  (critical — fully blocked)

Usage::

    from autonomica.escalation.slack import SlackEscalation

    gov = Autonomica(
        escalation=SlackEscalation("https://hooks.slack.com/services/T.../B.../...")
    )
"""
from __future__ import annotations

import time
from typing import Any, Callable, Coroutine, Optional

import httpx

from autonomica.escalation.base import BaseEscalation
from autonomica.models import AgentAction, GovernanceMode, RiskScore

# ── Colour codes per mode ──────────────────────────────────────────────────────
_COLOURS: dict[GovernanceMode, str] = {
    GovernanceMode.FULL_AUTO: "#4CAF50",
    GovernanceMode.LOG_AND_ALERT: "#2196F3",
    GovernanceMode.SOFT_GATE: "#FF9800",
    GovernanceMode.HARD_GATE: "#F44336",
    GovernanceMode.QUARANTINE: "#9C27B0",
}

# ── Emoji per mode ─────────────────────────────────────────────────────────────
_EMOJI: dict[GovernanceMode, str] = {
    GovernanceMode.FULL_AUTO: "✅",
    GovernanceMode.LOG_AND_ALERT: "📋",
    GovernanceMode.SOFT_GATE: "⚠️",
    GovernanceMode.HARD_GATE: "🚨",
    GovernanceMode.QUARANTINE: "🔴",
}

# ── Human-readable action instructions ────────────────────────────────────────
_INSTRUCTIONS: dict[GovernanceMode, str] = {
    GovernanceMode.FULL_AUTO: "Approved automatically — no action needed.",
    GovernanceMode.LOG_AND_ALERT: "Proceeding automatically — FYI only.",
    GovernanceMode.SOFT_GATE: (
        "Will proceed automatically after timeout unless you veto. "
        "POST /api/governance/override with approved=false to block."
    ),
    GovernanceMode.HARD_GATE: (
        "BLOCKED — waiting for your explicit approval. "
        "POST /api/governance/override with approved=true to allow."
    ),
    GovernanceMode.QUARANTINE: "BLOCKED — full review required before this agent can proceed.",
}

# Type alias for the injectable sender (makes unit-testing easy)
_SenderFn = Callable[[str, dict], Coroutine[Any, Any, None]]


class SlackEscalation(BaseEscalation):
    """Posts governance alerts to Slack via incoming webhook.

    Args:
        webhook_url: Slack incoming-webhook URL.
        _sender:     Optional async callable ``(url, payload) → None``.
                     Injected during tests to avoid real HTTP calls.
                     When None, uses ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        webhook_url: str,
        _sender: Optional[_SenderFn] = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._sender = _sender or self._default_sender

    # ── BaseEscalation interface ──────────────────────────────────────────────

    async def notify(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> None:
        """Build and send a Slack message for the governance event."""
        payload = self._build_payload(action, mode, risk_score)
        try:
            await self._sender(self._webhook_url, payload)
        except Exception:
            # Never let a Slack failure block the governance pipeline.
            pass

    async def wait_for_response(
        self, action_id: str, timeout: float
    ) -> Optional[bool]:
        """MVP: always return None (timeout).

        Human responses arrive via the API's POST /api/governance/override
        endpoint, which resolves the gate Future directly.
        """
        return None

    # ── Message builder ───────────────────────────────────────────────────────

    def _build_payload(
        self, action: AgentAction, mode: GovernanceMode, risk_score: RiskScore
    ) -> dict[str, Any]:
        emoji = _EMOJI.get(mode, "⚙️")
        colour = _COLOURS.get(mode, "#607D8B")
        title = (
            f"{emoji} {mode.name}: "
            f"Agent *{action.agent_name}* wants to call `{action.tool_name}`"
        )

        # Flatten tool_input into a readable string (truncate if large)
        raw_input = str(action.tool_input)
        if len(raw_input) > 300:
            raw_input = raw_input[:297] + "..."

        fields: list[dict[str, Any]] = [
            {"title": "Agent ID",     "value": action.agent_id,                              "short": True},
            {"title": "Tool",         "value": f"`{action.tool_name}`",                      "short": True},
            {"title": "Action Type",  "value": action.action_type.value,                     "short": True},
            {"title": "Mode",         "value": mode.name,                                    "short": True},
            {"title": "Risk Score",   "value": f"{risk_score.composite_score:.1f}/100",      "short": True},
            {"title": "Explanation",  "value": risk_score.explanation,                       "short": False},
            {"title": "Tool Input",   "value": f"```{raw_input}```",                         "short": False},
            {"title": "Next Step",    "value": _INSTRUCTIONS[mode],                          "short": False},
        ]

        return {
            "attachments": [
                {
                    "color": colour,
                    "title": title,
                    "fields": fields,
                    "footer": f"Autonomica  •  action_id: {action.action_id}",
                    "ts": int(time.time()),
                    "mrkdwn_in": ["text", "title", "fields"],
                }
            ]
        }

    # ── Default HTTP sender ───────────────────────────────────────────────────

    @staticmethod
    async def _default_sender(url: str, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
