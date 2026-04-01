import pytest
from unittest.mock import MagicMock
from autonomica.integrations.crewai import wrap_crewai_tools, GovernedCrewAITool
from autonomica.models import ActionType, GovernanceDecision, GovernanceMode, RiskScore

class MockTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    def _run(self, **kwargs):
        return f"Result of {self.name}"

def test_wrap_crewai_tools():
    autonomica = MagicMock()
    tools = [MockTool("test_tool", "A test tool")]
    governed_tools = wrap_crewai_tools(tools, autonomica, agent_id="test_agent")
    
    assert len(governed_tools) == 1
    assert isinstance(governed_tools[0], GovernedCrewAITool)
    assert governed_tools[0].name == "test_tool"

def test_governed_tool_approved():
    autonomica = MagicMock()
    risk = RiskScore(
        composite_score=0.1,
        financial_magnitude=0.0,
        data_sensitivity=0.0,
        reversibility=0.0,
        agent_track_record=0.0,
        novelty=0.0,
        cascade_risk=0.0,
        explanation="Safe"
    )
    decision = GovernanceDecision(
        action_id="test_id",
        risk_score=risk,
        mode=GovernanceMode.FULL_AUTO,
        approved=True,
        decision_time_ms=0.1
    )
    autonomica.evaluate_action_sync.return_value = decision
    
    original_tool = MockTool("test_tool", "A test tool")
    tools = [original_tool]
    governed_tools = wrap_crewai_tools(tools, autonomica, agent_id="test_agent")
    
    result = governed_tools[0]._run(param="val")
    assert result == "Result of test_tool"
    autonomica.evaluate_action_sync.assert_called_once()
    autonomica.record_outcome.assert_called_once()

def test_governed_tool_blocked():
    autonomica = MagicMock()
    risk = RiskScore(
        composite_score=0.9,
        financial_magnitude=0.0,
        data_sensitivity=0.0,
        reversibility=0.0,
        agent_track_record=0.0,
        novelty=0.0,
        cascade_risk=0.0,
        explanation="Dangerous"
    )
    decision = GovernanceDecision(
        action_id="test_id",
        risk_score=risk,
        mode=GovernanceMode.QUARANTINE,
        approved=False,
        decision_time_ms=0.1
    )
    autonomica.evaluate_action_sync.return_value = decision
    
    original_tool = MockTool("test_tool", "A test tool")
    tools = [original_tool]
    governed_tools = wrap_crewai_tools(tools, autonomica, agent_id="test_agent")
    
    result = governed_tools[0]._run(param="val")
    assert "[AUTONOMICA] Action blocked" in result
    autonomica.evaluate_action_sync.assert_called_once()
    autonomica.record_outcome.assert_not_called()

@pytest.mark.asyncio
async def test_governed_tool_arun():
    autonomica = MagicMock()
    risk = RiskScore(
        composite_score=0.1,
        financial_magnitude=0.0,
        data_sensitivity=0.0,
        reversibility=0.0,
        agent_track_record=0.0,
        novelty=0.0,
        cascade_risk=0.0,
        explanation="Safe"
    )
    decision = GovernanceDecision(
        action_id="test_id",
        risk_score=risk,
        mode=GovernanceMode.FULL_AUTO,
        approved=True,
        decision_time_ms=0.1
    )
    autonomica.evaluate_action.return_value = decision
    
    original_tool = MockTool("test_tool", "A test tool")
    tools = [original_tool]
    governed_tools = wrap_crewai_tools(tools, autonomica, agent_id="test_agent")
    
    result = await governed_tools[0]._arun(param="val")
    assert result == "Result of test_tool"
    autonomica.evaluate_action.assert_called_once()

def test_action_type_inference():
    autonomica = MagicMock()
    tools = [
        MockTool("delete_user", "Removes a user"),
        MockTool("send_email", "Sends an email"),
        MockTool("update_record", "Writes to db"),
        MockTool("get_data", "Reads data")
    ]
    governed_tools = wrap_crewai_tools(tools, autonomica, agent_id="test_agent")
    
    assert governed_tools[0]._infer_action_type() == ActionType.DELETE
    assert governed_tools[1]._infer_action_type() == ActionType.COMMUNICATE
    assert governed_tools[2]._infer_action_type() == ActionType.WRITE
    assert governed_tools[3]._infer_action_type() == ActionType.READ
