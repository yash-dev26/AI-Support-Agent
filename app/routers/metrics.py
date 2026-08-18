import logging

from fastapi import APIRouter, HTTPException

from app.core.deps import get_metrics_col

logger = logging.getLogger("support_agent.metrics")
router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics():
    """Escalation rate, avg time-to-resolution, tool usage breakdown."""
    try:
        col = get_metrics_col()
        total = col.count_documents({"event_type": {"$in": ["resolved_by_agent", "escalated"]}})
        escalated = col.count_documents({"event_type": "escalated"})
        resolved_by_human = list(col.find({"event_type": "resolved_by_human"}))
        resolved_by_agent = list(col.find({"event_type": "resolved_by_agent"}))
    except Exception:
        logger.exception("Failed to compute metrics")
        raise HTTPException(status_code=503, detail="Metrics store unavailable.")

    avg_resolution_ms = (
        sum(e["resolution_latency_ms"] for e in resolved_by_human) / len(resolved_by_human)
        if resolved_by_human else None
    )
    avg_agent_ms = (
        sum(e["latency_ms"] for e in resolved_by_agent) / len(resolved_by_agent)
        if resolved_by_agent else None
    )

    return {
        "total_conversations": total,
        "escalation_rate": round(escalated / total, 3) if total else None,
        "avg_agent_latency_ms": avg_agent_ms,
        "avg_human_resolution_ms": avg_resolution_ms,
        "total_escalations": escalated,
        "total_human_resolutions": len(resolved_by_human),
    }


