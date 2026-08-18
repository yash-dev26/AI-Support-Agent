import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.deps import get_mongo_client

logger = logging.getLogger("support_agent.health")
router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Basic liveness + dependency check. Returns 503 if Mongo isn't reachable."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
    except Exception:
        logger.exception("Health check failed: Mongo unreachable")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "mongo": "unreachable"})

    return {"status": "ok", "mongo": "connected"}


