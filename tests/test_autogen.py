import asyncio
import pytest
from unittest.mock import MagicMock

from autonomica.integrations.autogen import (
    wrap_autogen_callable,
    govern_autogen_agent,
)
from autonomica.models import AgentAction, GovernanceDecision, GovernanceMode, RiskScore

def test_wrap_autogen_callable_sync():
    """wrap_autogen_callable wraps a sync function correctly."""
    def mock_fn(x: int) -> int:
        return x + 1
    
    mock_autonomica = MagicMock()
    # Mock evaluate_action_sync returning an approved decision
    mock_autonomica.evaluate_action_sync.return_value = GovernanceDecision(
        action_id="123",
        approved=True,
        mode=GovernanceMode.MONITOR,
        risk_score=RiskScore(composite_score=0.1, explanation="Safe")
    )
    
    wrapped = wrap_autogen_callable(
        mock_fn,
        autonomica=mock_autonomica,
        agent_id="test_agent"
    )
    
    result = wrapped(x=10)
    assert result == 11
    mock_autonomica.evaluate_action_sync.assert_called_once()

@pytest.mark.asyncio
async def test_wrap_autogen_callable_async():
    """wrap_autogen_callable wraps an async function correctly."""
    async def mock_fn(x: int) -> int:
        return x + 1
    
    mock_autonomica = MagicMock()
    # Mock evaluate_action returning an approved decision
    # (Since it's async, wrap_autogen_callable uses evaluate_action)
    f = asyncio.Future()
    f.set_result(GovernanceDecision(
        action_id="123",
        approved=True,
        mode=GovernanceMode.MONITOR,
        risk_score=RiskScore(composite_score=0.1, explanation="Safe")
    ))
    mock_autonomica.evaluate_action.return_value = f
    
    wrapped = wrap_autogen_callable(
        mock_fn,
        autonomica=mock_autonomica,
        agent_id="test_agent"
    )
    
    result = await wrapped(x=10)
    assert result == 11
    mock_autonomica.evaluate_action.assert_called_once()

def test_blocked_action_returns_string():
    """Blocked action returns the [AUTONOMICA] string without calling the function."""
    mock_fn = MagicMock()
    mock_autonomica = MagicMock()
    mock_autonomica.evaluate_action_sync.return_value = GovernanceDecision(
        action_id="123",
        approved=False,
        mode=GovernanceMode.HARD_GATE,
        risk_score=RiskScore(composite_score=0.9, explanation="Dangerous")
    )
    
    wrapped = wrap_autogen_callable(
        mock_fn,
        autonomica=mock_autonomica,
        agent_id="test_agent"
    )
    
    result = wrapped(x=10)
    assert "[AUTONOMICA]" in result
    assert "Action blocked" in result
    mock_fn.assert_not_called()

def test_govern_autogen_agent_patches_function_map():
    """govern_autogen_agent patches all entries in function_map."""
    def fn1(): pass
    def fn2(): pass
    
    agent = MagicMock()
    agent.function_map = {"f1": fn1, "f2": fn2}
    
    mock_autonomica = MagicMock()
    govern_autogen_agent(agent, autonomica=mock_autonomica)
    
    assert agent.function_map["f1"] != fn1
    assert agent.function_map["f2"] != fn2
    # Verify they are wrapped
    agent.function_map["f1"]()
    mock_autonomica.evaluate_action_sync.assert_called()

def test_approved_action_calls_through():
    """Approved action calls through to the original function."""
    mock_fn = MagicMock(return_value="success")
    mock_autonomica = MagicMock()
    mock_autonomica.evaluate_action_sync.return_value = GovernanceDecision(
        action_id="123",
        approved=True,
        mode=GovernanceMode.MONITOR,
        risk_score=RiskScore(composite_score=0.1, explanation="Safe")
    )
    
    wrapped = wrap_autogen_callable(
        mock_fn,
        autonomica=mock_autonomica,
        agent_id="test_agent"
    )
    
    result = wrapped()
    assert result == "success"
    mock_fn.assert_called_once()
