"""Autonomica — Runtime adaptive governance for AI agents.

Typical usage::

    from autonomica import govern

    @govern(agent_id="finance-bot", action_type="financial")
    def process_payment(amount: float, recipient: str) -> str:
        return f"Paid ${amount} to {recipient}"
"""

from autonomica.adapter import AdaptationEngine
from autonomica.audit import AuditLogger
from autonomica.config import AutonomicaConfig
from autonomica.decorator import GovernanceBlocked, govern
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
    # Decorator (primary API)
    "govern",
    "GovernanceBlocked",
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
