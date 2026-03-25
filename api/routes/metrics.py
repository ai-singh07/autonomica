"""Metrics and observability endpoints."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from api.dependencies import get_gov
from autonomica.models import GovernanceMode

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/overview", summary="Dashboard overview metrics")
async def metrics_overview(gov=Depends(get_gov)) -> dict:
    """Aggregate metrics across all agents and all decisions."""
    decisions = list(gov._decisions.values())
    profiles = list(gov._profiles.values())
    total = len(decisions)

    if total == 0:
        return {
            "total_actions": 0,
            "total_agents": len(profiles),
            "approval_rate": 0.0,
            "escalation_rate": 0.0,
            "incident_rate": 0.0,
            "mode_distribution": {},
            "average_risk_score": 0.0,
            "average_decision_time_ms": 0.0,
            "average_trust_score": 0.0,
            "average_vagal_tone": 0.0,
        }

    approved = sum(1 for d in decisions if d.approved)
    escalated = sum(1 for d in decisions if d.mode >= GovernanceMode.SOFT_GATE)
    mode_dist = dict(Counter(d.mode.name for d in decisions))
    avg_score = sum(d.risk_score.composite_score for d in decisions) / total
    avg_time = sum(d.decision_time_ms for d in decisions) / total

    total_incidents = sum(p.incidents for p in profiles)
    avg_trust = sum(p.trust_score for p in profiles) / max(len(profiles), 1)
    avg_vagal = sum(p.vagal_tone for p in profiles) / max(len(profiles), 1)

    return {
        "total_actions": total,
        "total_agents": len(profiles),
        "approval_rate": round(approved / total, 3),
        "escalation_rate": round(escalated / total, 3),
        "incident_rate": round(total_incidents / max(total, 1), 3),
        "mode_distribution": mode_dist,
        "average_risk_score": round(avg_score, 2),
        "average_decision_time_ms": round(avg_time, 2),
        "average_trust_score": round(avg_trust, 2),
        "average_vagal_tone": round(avg_vagal, 2),
    }


@router.get("/vagal-tone", summary="Vagal tone across all agents")
async def vagal_tone(gov=Depends(get_gov)) -> dict:
    """Return calibration quality (vagal tone) for every agent."""
    profiles = sorted(
        gov._profiles.values(),
        key=lambda p: p.vagal_tone,
        reverse=True,
    )
    return {
        "agents": [
            {
                "agent_id": p.agent_id,
                "agent_name": p.agent_name,
                "vagal_tone": round(p.vagal_tone, 2),
                "incidents": p.incidents,
                "false_escalations": p.false_escalations,
                "total_actions": p.total_actions,
            }
            for p in profiles
        ]
    }


@router.get("/adaptation", summary="Current adaptive threshold state per agent")
async def adaptation_state(gov=Depends(get_gov)) -> dict:
    """Show how thresholds have drifted from their defaults for each agent."""
    defaults = {
        "full_auto_max": 15.0,
        "log_alert_max": 35.0,
        "soft_gate_max": 60.0,
        "hard_gate_max": 85.0,
    }
    agents = []
    for p in gov._profiles.values():
        deltas = {
            k: round(p.mode_thresholds.get(k, defaults[k]) - defaults[k], 2)
            for k in defaults
        }
        agents.append(
            {
                "agent_id": p.agent_id,
                "agent_name": p.agent_name,
                "trust_score": round(p.trust_score, 2),
                "current_thresholds": p.mode_thresholds,
                "threshold_deltas": deltas,
                "total_actions": p.total_actions,
            }
        )
    return {"agents": agents}
