"""Governance control endpoints — human override and config."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_gov

router = APIRouter(prefix="/governance", tags=["governance"])


# ── Request / response models ─────────────────────────────────────────────────

class OverrideRequest(BaseModel):
    action_id: str
    approved: bool
    reason: str = ""


class OverrideResponse(BaseModel):
    status: str
    action_id: str
    approved: bool
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/override", response_model=OverrideResponse, summary="Human override for a pending action")
async def human_override(
    request: OverrideRequest,
    gov=Depends(get_gov),
) -> OverrideResponse:
    """
    Approve or reject an action that is waiting at a SOFT_GATE or HARD_GATE.

    - ``approved=true``  → allow the action to proceed.
    - ``approved=false`` → block the action.

    This resolves the gate immediately without waiting for the timeout.
    The adaptation engine then widens or tightens the agent's thresholds
    based on the override decision.
    """
    decision = gov.get_decision(request.action_id)
    if decision is None:
        raise HTTPException(
            status_code=404,
            detail=f"Action '{request.action_id}' not found.",
        )

    gov.record_human_override(
        request.action_id,
        approved=request.approved,
        reason=request.reason,
    )

    verb = "approved" if request.approved else "rejected"
    return OverrideResponse(
        status="ok",
        action_id=request.action_id,
        approved=request.approved,
        message=f"Action {verb}. Governance pipeline notified.",
    )


@router.get("/config", summary="Get current governance configuration")
async def get_config(gov=Depends(get_gov)) -> dict:
    """Return the active AutonomicaConfig (or defaults if none was provided)."""
    cfg = gov._config
    if cfg is None:
        return {
            "soft_gate_timeout_seconds": 60,
            "hard_gate_timeout_seconds": 300,
            "default_trust_score": 50.0,
            "adaptation_rate": 0.5,
            "min_actions_before_adaptation": 10,
            "storage_backend": "memory",
            "escalation_backend": "console",
            "note": "Using built-in defaults (no AutonomicaConfig passed).",
        }
    return {
        "soft_gate_timeout_seconds": cfg.soft_gate_timeout_seconds,
        "hard_gate_timeout_seconds": cfg.hard_gate_timeout_seconds,
        "default_trust_score": cfg.default_trust_score,
        "adaptation_rate": cfg.adaptation_rate,
        "min_actions_before_adaptation": cfg.min_actions_before_adaptation,
        "storage_backend": cfg.storage_backend,
        "escalation_backend": cfg.escalation_backend,
        "scoring_weights": cfg.scoring_weights,
    }
