"""Agent profile endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_gov
from autonomica.models import GovernanceMode

router = APIRouter(prefix="/agents", tags=["agents"])


def _profile_summary(profile: Any) -> dict:
    return {
        "agent_id": profile.agent_id,
        "agent_name": profile.agent_name,
        "total_actions": profile.total_actions,
        "approved_actions": profile.approved_actions,
        "escalated_actions": profile.escalated_actions,
        "incidents": profile.incidents,
        "false_escalations": profile.false_escalations,
        "trust_score": round(profile.trust_score, 2),
        "vagal_tone": round(profile.vagal_tone, 2),
        "mode_thresholds": profile.mode_thresholds,
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }


@router.get("", summary="List all agent profiles")
async def list_agents(gov=Depends(get_gov)) -> list[dict]:
    """Return a summary for every agent seen since startup."""
    profiles = list(gov._profiles.values())
    profiles.sort(key=lambda p: p.updated_at, reverse=True)
    return [_profile_summary(p) for p in profiles]


@router.get("/{agent_id}", summary="Get agent profile detail")
async def get_agent(agent_id: str, gov=Depends(get_gov)) -> dict:
    """Return the full profile including adaptive thresholds and per-tool trust."""
    profile = gov.get_agent_profile(agent_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return {
        **_profile_summary(profile),
        "per_tool_trust": profile.per_tool_trust,
    }


@router.get("/{agent_id}/actions", summary="List action history for an agent")
async def get_agent_actions(
    agent_id: str,
    limit: int = 50,
    gov=Depends(get_gov),
) -> dict:
    """Return the most recent governance decisions for a given agent."""
    profile = gov.get_agent_profile(agent_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    items = []
    for action_id, decision in gov._decisions.items():
        if gov._action_agents.get(action_id) != agent_id:
            continue
        action = gov._actions.get(action_id)
        items.append(
            {
                "action_id": action_id,
                "agent_id": agent_id,
                "tool_name": action.tool_name if action else "unknown",
                "action_type": action.action_type.value if action else "unknown",
                "composite_score": round(decision.risk_score.composite_score, 2),
                "governance_mode": decision.mode.name,
                "approved": decision.approved,
                "human_override": decision.human_override,
                "decision_time_ms": round(decision.decision_time_ms, 2),
                "timestamp": decision.timestamp.isoformat(),
            }
        )

    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"agent_id": agent_id, "total": len(items), "items": items[:limit]}
