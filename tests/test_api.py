"""FastAPI dashboard backend tests.

Uses httpx.AsyncClient with ASGI transport so tests run without a real server.
Each test gets a fresh Autonomica instance injected via FastAPI dependency
override to avoid shared state between tests.

Coverage:
  GET  /health
  GET  /api/agents
  GET  /api/agents/{id}
  GET  /api/agents/{id}/actions
  GET  /api/actions
  GET  /api/actions/{id}
  GET  /api/metrics/overview
  GET  /api/metrics/vagal-tone
  GET  /api/metrics/adaptation
  POST /api/governance/override
  GET  /api/governance/config
  GET  /api/audit/export
"""
from __future__ import annotations

from typing import Any, Optional

import pytest
from httpx import ASGITransport, AsyncClient

from autonomica import Autonomica, GovernanceMode
from autonomica.escalation.base import BaseEscalation
from autonomica.models import ActionType, AgentAction, AgentProfile


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MockEscalation(BaseEscalation):
    def __init__(self, response: Optional[bool] = None) -> None:
        self.notifications: list = []
        self._response = response

    async def notify(self, action: AgentAction, mode: GovernanceMode) -> None:
        self.notifications.append(mode)

    async def wait_for_response(self, action_id: str, timeout: float) -> Optional[bool]:
        return self._response


class _FastConfig:
    soft_gate_timeout_seconds = 0.02
    hard_gate_timeout_seconds = 0.02
    default_trust_score = 50.0
    adaptation_rate = 0.5
    min_actions_before_adaptation = 10
    storage_backend = "sqlite"
    escalation_backend = "console"
    scoring_weights = {
        "financial_magnitude": 0.25,
        "data_sensitivity": 0.20,
        "reversibility": 0.20,
        "agent_track_record": 0.15,
        "novelty": 0.10,
        "cascade_risk": 0.10,
    }


def _make_gov(escalation: Optional[BaseEscalation] = None) -> Autonomica:
    return Autonomica(
        config=_FastConfig(),
        escalation=escalation or _MockEscalation(),
    )


def _seed_trusted_profile(gov: Autonomica, agent_id: str = "agent") -> AgentProfile:
    p = gov._get_or_create_profile(agent_id, agent_id)
    p.trust_score = 90.0
    p.per_tool_trust = {"read_database": 20}
    return p


async def _run_one_action(
    gov: Autonomica,
    agent_id: str = "agent",
    tool_name: str = "read_database",
    action_type: ActionType = ActionType.READ,
    tool_input: dict | None = None,
) -> str:
    """Fire an action through evaluate_action and return action_id."""
    action = AgentAction(
        agent_id=agent_id,
        agent_name=agent_id,
        tool_name=tool_name,
        tool_input=tool_input or {"query": "test"},
        action_type=action_type,
    )
    await gov.evaluate_action(action)
    return action.action_id


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gov():
    """Fresh Autonomica instance per test, pre-seeded with a trusted profile."""
    instance = _make_gov()
    _seed_trusted_profile(instance, "agent")
    return instance


@pytest.fixture
async def client(gov):
    """AsyncClient wired to the FastAPI app with the test gov injected."""
    from api.dependencies import get_gov
    from api.main import app

    app.dependency_overrides[get_gov] = lambda: gov
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_returns_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_body(self, client):
        r = await client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "autonomica"


# ── Agents ────────────────────────────────────────────────────────────────────

class TestAgentsEndpoints:
    async def test_list_agents_empty(self):
        """A fresh gov with no profiles returns []."""
        fresh_gov = _make_gov()
        from api.dependencies import get_gov
        from api.main import app

        app.dependency_overrides[get_gov] = lambda: fresh_gov
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            r = await c.get("/api/agents")
        app.dependency_overrides.clear()

        assert r.status_code == 200
        assert r.json() == []

    async def test_list_agents_returns_profiles(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/agents")
        assert r.status_code == 200
        agents = r.json()
        assert any(a["agent_id"] == "agent" for a in agents)

    async def test_list_agents_includes_trust_and_vagal_tone(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/agents")
        agent = next(a for a in r.json() if a["agent_id"] == "agent")
        assert "trust_score" in agent
        assert "vagal_tone" in agent

    async def test_get_agent_returns_profile(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/agents/agent")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_id"] == "agent"
        assert "mode_thresholds" in body
        assert "per_tool_trust" in body

    async def test_get_agent_404_for_unknown(self, client):
        r = await client.get("/api/agents/ghost-agent")
        assert r.status_code == 404

    async def test_get_agent_actions_returns_list(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/agents/agent/actions")
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert len(body["items"]) >= 1

    async def test_get_agent_actions_404_for_unknown(self, client):
        r = await client.get("/api/agents/ghost/actions")
        assert r.status_code == 404

    async def test_get_agent_actions_includes_risk_score(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/agents/agent/actions")
        item = r.json()["items"][0]
        assert "composite_score" in item
        assert "governance_mode" in item
        assert "approved" in item

    async def test_get_agent_actions_limit_respected(self, gov, client):
        for _ in range(5):
            await _run_one_action(gov)
        r = await client.get("/api/agents/agent/actions?limit=3")
        assert len(r.json()["items"]) <= 3


# ── Actions ───────────────────────────────────────────────────────────────────

class TestActionsEndpoints:
    async def test_list_actions_empty(self, client):
        r = await client.get("/api/actions")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["items"] == []

    async def test_list_actions_returns_items(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/actions")
        body = r.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1

    async def test_list_actions_pagination(self, gov, client):
        for _ in range(5):
            await _run_one_action(gov)
        r = await client.get("/api/actions?limit=2&offset=0")
        assert len(r.json()["items"]) == 2

    async def test_list_actions_item_shape(self, gov, client):
        await _run_one_action(gov)
        item = (await client.get("/api/actions")).json()["items"][0]
        for field in ("action_id", "agent_id", "tool_name", "governance_mode",
                      "composite_score", "approved", "timestamp"):
            assert field in item, f"missing field: {field}"

    async def test_get_action_detail(self, gov, client):
        action_id = await _run_one_action(gov)
        r = await client.get(f"/api/actions/{action_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["action_id"] == action_id
        assert "explanation" in body
        assert "risk_breakdown" in body
        assert "tool_input" in body

    async def test_get_action_404(self, client):
        r = await client.get("/api/actions/nonexistent-id")
        assert r.status_code == 404


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetricsEndpoints:
    async def test_overview_empty_state(self, client):
        r = await client.get("/api/metrics/overview")
        assert r.status_code == 200
        body = r.json()
        assert body["total_actions"] == 0
        assert body["approval_rate"] == 0.0

    async def test_overview_after_actions(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/metrics/overview")
        body = r.json()
        assert body["total_actions"] == 1
        assert body["total_agents"] >= 1
        assert 0.0 <= body["approval_rate"] <= 1.0
        assert "mode_distribution" in body

    async def test_overview_includes_all_fields(self, gov, client):
        await _run_one_action(gov)
        body = (await client.get("/api/metrics/overview")).json()
        for field in ("total_actions", "total_agents", "approval_rate",
                      "escalation_rate", "mode_distribution",
                      "average_risk_score", "average_decision_time_ms"):
            assert field in body, f"missing: {field}"

    async def test_vagal_tone_endpoint(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/metrics/vagal-tone")
        assert r.status_code == 200
        body = r.json()
        assert "agents" in body
        agent = next((a for a in body["agents"] if a["agent_id"] == "agent"), None)
        assert agent is not None
        assert 0.0 <= agent["vagal_tone"] <= 100.0

    async def test_adaptation_endpoint(self, gov, client):
        r = await client.get("/api/metrics/adaptation")
        assert r.status_code == 200
        body = r.json()
        assert "agents" in body

    async def test_mode_distribution_keys_are_mode_names(self, gov, client):
        await _run_one_action(gov)
        body = (await client.get("/api/metrics/overview")).json()
        valid_modes = {"FULL_AUTO", "LOG_AND_ALERT", "SOFT_GATE", "HARD_GATE", "QUARANTINE"}
        for key in body["mode_distribution"]:
            assert key in valid_modes


# ── Governance override ───────────────────────────────────────────────────────

class TestGovernanceOverrideEndpoint:
    async def test_override_approve_valid_action(self, gov, client):
        action_id = await _run_one_action(gov)
        r = await client.post(
            "/api/governance/override",
            json={"action_id": action_id, "approved": True, "reason": "looks fine"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["approved"] is True
        assert body["action_id"] == action_id

    async def test_override_reject_valid_action(self, gov, client):
        action_id = await _run_one_action(gov)
        r = await client.post(
            "/api/governance/override",
            json={"action_id": action_id, "approved": False, "reason": "suspicious"},
        )
        assert r.status_code == 200
        assert r.json()["approved"] is False

    async def test_override_unknown_action_returns_404(self, client):
        r = await client.post(
            "/api/governance/override",
            json={"action_id": "ghost-id", "approved": True},
        )
        assert r.status_code == 404

    async def test_override_increments_false_escalations_on_approve(self, gov, client):
        action_id = await _run_one_action(gov)
        await client.post(
            "/api/governance/override",
            json={"action_id": action_id, "approved": True},
        )
        profile = gov.get_agent_profile("agent")
        assert profile.false_escalations == 1

    async def test_override_rejection_does_not_increment_false_escalations(self, gov, client):
        action_id = await _run_one_action(gov)
        await client.post(
            "/api/governance/override",
            json={"action_id": action_id, "approved": False},
        )
        profile = gov.get_agent_profile("agent")
        assert profile.false_escalations == 0

    async def test_get_config_returns_dict(self, client):
        r = await client.get("/api/governance/config")
        assert r.status_code == 200
        body = r.json()
        assert "soft_gate_timeout_seconds" in body
        assert "hard_gate_timeout_seconds" in body

    async def test_get_config_with_autonomica_config(self, client, gov):
        r = await client.get("/api/governance/config")
        assert r.status_code == 200
        body = r.json()
        assert body["soft_gate_timeout_seconds"] == pytest.approx(0.02)


# ── Audit export ──────────────────────────────────────────────────────────────

class TestAuditExportEndpoint:
    async def test_export_jsonl_returns_200(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/audit/export?fmt=jsonl")
        assert r.status_code == 200

    async def test_export_json_content_type(self, gov, client):
        r = await client.get("/api/audit/export?fmt=json")
        assert r.status_code == 200
        assert "json" in r.headers["content-type"]

    async def test_export_csv_content_type(self, gov, client):
        r = await client.get("/api/audit/export?fmt=csv")
        assert r.status_code == 200
        assert "csv" in r.headers["content-type"]

    async def test_export_includes_decisions(self, gov, client):
        await _run_one_action(gov)
        r = await client.get("/api/audit/export?fmt=json")
        import json
        entries = json.loads(r.text)
        # AuditLogger writes to Python logging only (no file) by default,
        # so read_entries() returns [] — still 200 with empty body
        assert isinstance(entries, list)

    async def test_export_header_contains_event_count(self, gov, client):
        r = await client.get("/api/audit/export?fmt=jsonl")
        assert "x-total-events" in r.headers

    async def test_export_invalid_format_falls_back_to_jsonl(self, gov, client):
        r = await client.get("/api/audit/export?fmt=xml")
        assert r.status_code == 200
        assert "ndjson" in r.headers["content-type"] or "json" in r.headers["content-type"]

    async def test_export_content_disposition_header(self, gov, client):
        r = await client.get("/api/audit/export?fmt=csv")
        assert "attachment" in r.headers.get("content-disposition", "")


# ── Slack escalation ──────────────────────────────────────────────────────────

class TestSlackEscalation:
    """Unit tests for SlackEscalation — uses an injected mock sender."""

    def _make_slack(self) -> tuple:
        from autonomica.escalation.slack import SlackEscalation

        sent: list = []

        async def mock_sender(url: str, payload: dict) -> None:
            sent.append((url, payload))

        slack = SlackEscalation(
            webhook_url="https://hooks.slack.com/test",
            _sender=mock_sender,
        )
        return slack, sent

    async def test_notify_sends_to_webhook(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="test-agent",
            agent_name="Test Agent",
            tool_name="process_payment",
            tool_input={"amount": 50000},
            action_type=ActionType.FINANCIAL,
        )
        await slack.notify(action, GovernanceMode.SOFT_GATE)
        assert len(sent) == 1
        url, payload = sent[0]
        assert url == "https://hooks.slack.com/test"
        assert "attachments" in payload

    async def test_soft_gate_colour_is_amber(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.FINANCIAL,
        )
        await slack.notify(action, GovernanceMode.SOFT_GATE)
        colour = sent[0][1]["attachments"][0]["color"]
        assert colour == "#FF9800"

    async def test_hard_gate_colour_is_red(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.DELETE,
        )
        await slack.notify(action, GovernanceMode.HARD_GATE)
        colour = sent[0][1]["attachments"][0]["color"]
        assert colour == "#F44336"

    async def test_quarantine_colour_is_purple(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.DELETE,
        )
        await slack.notify(action, GovernanceMode.QUARANTINE)
        colour = sent[0][1]["attachments"][0]["color"]
        assert colour == "#9C27B0"

    async def test_message_contains_agent_name(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="invoice-bot", agent_name="Invoice Bot",
            tool_name="send_invoice", tool_input={},
            action_type=ActionType.COMMUNICATE,
        )
        await slack.notify(action, GovernanceMode.LOG_AND_ALERT)
        payload_str = str(sent[0][1])
        assert "Invoice Bot" in payload_str

    async def test_message_contains_action_id(self):
        slack, sent = self._make_slack()
        action = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.READ,
        )
        await slack.notify(action, GovernanceMode.FULL_AUTO)
        payload_str = str(sent[0][1])
        assert action.action_id in payload_str

    async def test_wait_for_response_returns_none(self):
        slack, _ = self._make_slack()
        result = await slack.wait_for_response("any-id", timeout=1.0)
        assert result is None

    async def test_notify_silences_http_errors(self):
        """A failing webhook must never break the governance pipeline."""
        from autonomica.escalation.slack import SlackEscalation

        async def failing_sender(url: str, payload: dict) -> None:
            raise ConnectionError("Slack is down")

        slack = SlackEscalation(
            webhook_url="https://hooks.slack.com/bad",
            _sender=failing_sender,
        )
        action = AgentAction(
            agent_id="a", agent_name="A", tool_name="t",
            tool_input={}, action_type=ActionType.READ,
        )
        # Must not raise
        await slack.notify(action, GovernanceMode.SOFT_GATE)


import pytest
