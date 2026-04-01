"""Integration tests — OpenTelemetry tracing support."""
from __future__ import annotations

import pytest
from datetime import datetime

from autonomica.models import AgentAction, ActionType
from autonomica.interceptor import Autonomica

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    OTEL_SDK_AVAILABLE = True
except ImportError:
    OTEL_SDK_AVAILABLE = False

@pytest.fixture
def otel_setup():
    if not OTEL_SDK_AVAILABLE:
        pytest.skip("opentelemetry-sdk not installed")
    
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter

@pytest.mark.asyncio
async def test_otel_spans_emitted(otel_setup):
    exporter = otel_setup
    gov = Autonomica()
    action = AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="read_db",
        tool_input={"query": "test"},
        action_type=ActionType.READ,
        timestamp=datetime.utcnow()
    )
    
    await gov.evaluate_action(action)
    
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "autonomica.evaluate"
    assert span.attributes["agent.id"] == "test-agent"
    assert span.attributes["tool.name"] == "read_db"
    assert "governance.mode" in span.attributes
    assert "risk.score" in span.attributes
    assert "decision.approved" in span.attributes
