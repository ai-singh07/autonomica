"""Shared fixtures for Autonomica tests."""

import pytest
from autonomica.models import ActionType, AgentAction, AgentProfile


@pytest.fixture
def agent_profile() -> AgentProfile:
    return AgentProfile(agent_id="test-agent", agent_name="Test Agent")


@pytest.fixture
def read_action() -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="read_database",
        tool_input={"query": "SELECT * FROM users LIMIT 10"},
        action_type=ActionType.READ,
    )


@pytest.fixture
def financial_action() -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="process_payment",
        tool_input={"amount": 50000.0, "recipient": "vendor@example.com"},
        action_type=ActionType.FINANCIAL,
    )


@pytest.fixture
def delete_action() -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="delete_records",
        tool_input={"table": "users", "condition": "inactive=true"},
        action_type=ActionType.DELETE,
    )


@pytest.fixture
def communicate_action() -> AgentAction:
    return AgentAction(
        agent_id="test-agent",
        agent_name="Test Agent",
        tool_name="send_email",
        tool_input={
            "to": "user@example.com",
            "subject": "Your invoice",
            "body": "Please find attached...",
        },
        action_type=ActionType.COMMUNICATE,
    )
