"""Action log endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_gov

router = APIRouter(prefix="/actions", tags=["actions"])


def _decision_row(action_id: str, decision, action) -> dict:
    return {
        "action_id": action_id,
        "agent_id": action.agent_id if action else "unknown",
        "agent_name": action.agent_name if action else "unknown",
        "tool_name": action.tool_name if action else "unknown",
        "action_type": action.action_type.value if action else "unknown",
        "composite_score": round(decision.risk_score.composite_score, 2),
        "governance_mode": decision.mode.name,
        "approved": decision.approved,
        "human_override": decision.human_override,
        "decision_time_ms": round(decision.decision_time_ms, 2),
        "timestamp": decision.timestamp.isoformat(),
        "risk_breakdown": {
            "financial_magnitude": decision.risk_score.financial_magnitude,
            "data_sensitivity": decision.risk_score.data_sensitivity,
            "reversibility": decision.risk_score.reversibility,
            "agent_track_record": decision.risk_score.agent_track_record,
            "novelty": decision.risk_score.novelty,
            "cascade_risk": decision.risk_score.cascade_risk,
        },
    }


@router.get("", summary="List recent governance decisions (paginated)")
async def list_actions(
    limit: int = 50,
    offset: int = 0,
    gov=Depends(get_gov),
) -> dict:
    """Return all recent governance decisions, newest first."""
    all_items = [
        _decision_row(aid, d, gov._actions.get(aid))
        for aid, d in gov._decisions.items()
    ]
    all_items.sort(key=lambda x: x["timestamp"], reverse=True)
    page = all_items[offset : offset + limit]
    return {
        "total": len(all_items),
        "offset": offset,
        "limit": limit,
        "items": page,
    }


@router.get("/{action_id}", summary="Get a single action + decision detail")
async def get_action(action_id: str, gov=Depends(get_gov)) -> dict:
    decision = gov.get_decision(action_id)
    if decision is None:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found.")
    action = gov._actions.get(action_id)
    row = _decision_row(action_id, decision, action)
    # Add full explanation
    row["explanation"] = decision.risk_score.explanation
    if action:
        row["tool_input"] = action.tool_input
        row["metadata"] = action.metadata
    return row
