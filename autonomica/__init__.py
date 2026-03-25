"""Autonomica — Runtime adaptive governance for AI agents.

Typical usage::

    from autonomica import Autonomica
    from autonomica.integrations.langchain import wrap_langchain_tools

    gov = Autonomica()
    governed_tools = wrap_langchain_tools(tools, gov, agent_id="my-agent")
"""

from autonomica.adapter import AdaptationEngine
from autonomica.audit import AuditLogger
from autonomica.config import AutonomicaConfig
from autonomica.interceptor import Autonomica
from autonomica.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    GovernanceDecision,
    GovernanceMode,
    RiskScore,
)
from autonomica.storage.base import BaseStorage
from autonomica.storage.sqlite import SQLiteStorage

__all__ = [
    # Main class
    "Autonomica",
    # Configuration
    "AutonomicaConfig",
    # Sub-engines (useful for testing / custom setups)
    "AdaptationEngine",
    "AuditLogger",
    # Storage
    "BaseStorage",
    "SQLiteStorage",
    # Models
    "ActionType",
    "AgentAction",
    "AgentProfile",
    "GovernanceDecision",
    "GovernanceMode",
    "RiskScore",
]
