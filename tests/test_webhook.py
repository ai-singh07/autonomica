"""Webhook escalation tests — unit tests for the WebhookEscalation backend."""
from __future__ import annotations

import pytest
import respx
from httpx import Response
from datetime import datetime

from autonomica.models import AgentAction, ActionType, GovernanceMode, RiskScore
from autonomica.escalation.webhook import WebhookEscalation

@pytest.fixture
def sample_action():
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="read_db",
        tool_input={"query": "SELECT * FROM users"},
        action_type=ActionType.READ,
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def sample_risk_score():
    return RiskScore(
        composite_score=45.0,
        financial_magnitude=10.0,
        data_sensitivity=50.0,
        reversibility=30.0,
        agent_track_record=40.0,
        novelty=70.0,
        cascade_risk=20.0,
        explanation="Medium risk due to data sensitivity."
    )

@respx.mock
@pytest.mark.asyncio
async def test_webhook_notify_success(sample_action, sample_risk_score):
    url = "https://api.test/webhook"
    respx.post(url).respond(200)
    
    escalation = WebhookEscalation(url)
    await escalation.notify(sample_action, GovernanceMode.SOFT_GATE, sample_risk_score)
    
    assert respx.calls.count == 1

@respx.mock
@pytest.mark.asyncio
async def test_webhook_with_signature(sample_action, sample_risk_score):
    url = "https://api.test/webhook"
    secret = "test-secret"
    respx.post(url).respond(200)
    
    escalation = WebhookEscalation(url, secret=secret)
    await escalation.notify(sample_action, GovernanceMode.SOFT_GATE, sample_risk_score)
    
    assert "X-Autonomica-Signature" in respx.calls.last.request.headers

@respx.mock
@pytest.mark.asyncio
async def test_webhook_retry_on_5xx(sample_action, sample_risk_score):
    url = "https://api.test/webhook"
    # Mock first call as 500, second as 200
    respx.post(url).side_effect = [Response(500), Response(200)]
    
    escalation = WebhookEscalation(url)
    await escalation.notify(sample_action, GovernanceMode.SOFT_GATE, sample_risk_score)
    
    assert respx.calls.count == 2
