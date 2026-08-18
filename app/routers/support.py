import time
import logging
import asyncio

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from langgraph.types import Command

from app.core.deps import get_graph_app, log_event
from app.core.rate_limit import limiter
from app.core.auth import get_current_token, TokenPayload, require_support_agent
from app.core.agent_logging import log_response
from app.services import escalation_store
from app.services import money_detection
from app.services import ticket_store
from app.services.ws_manager import manager

logger = logging.getLogger("support_agent.support")
router = APIRouter(prefix="/support", tags=["support"])


class ResolveRequest(BaseModel):
    resolution: str
    confirmed: bool = False


class ReplyRequest(BaseModel):
    text: str


@router.get("/pending")
def list_pending(current: TokenPayload = Depends(get_current_token)):
    """All threads currently paused on a human-in-the-loop interrupt.
    Requires a support_agent token — this is internal tooling, not
    something a customer should be able to browse.

    Scans every thread that has EVER escalated (escalation_store's distinct
    thread_ids), not the seeded mock-user list — a thread for any user_id,
    seeded or not, shows up here the moment it actually escalates. Still a
    full scan per request, which is fine at demo/portfolio scale but would
    need a dedicated open-interrupts index at any real scale.
    """
    require_support_agent(current)
    graph_app = get_graph_app()
    pending = []

    try:
        known_threads = escalation_store.get_distinct_thread_ids()
    except Exception:
        logger.exception("Failed to read known threads from escalation_store")
        raise HTTPException(status_code=503, detail="Could not read thread list.")

    for user_id in known_threads:
        config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
        try:
            state = graph_app.get_state(config)
        except Exception:
            logger.warning("Skipping user %s — failed to read state", user_id, exc_info=True)
            continue
        interrupts = getattr(state, "interrupts", ())
        if interrupts:
            payload = interrupts[0].value or {}
            ticket = None
            ticket_id = payload.get("ticket_id")
            if ticket_id:
                try:
                    ticket = _serialize_ticket(ticket_store.get_ticket(ticket_id))
                except Exception:
                    logger.warning("Failed to load ticket %s for thread %s", ticket_id, user_id, exc_info=True)
            pending.append({
                "thread_id": user_id,
                "query": payload.get("query"),
                "message": payload.get("message"),
                "issue_type": payload.get("issue_type"),
                "ticket": ticket,
                "message_count": len(escalation_store.get_messages(user_id)),
            })
    return {"pending": pending}


def _serialize_ticket(ticket: dict | None) -> dict | None:
    """Mongo docs store datetimes as datetime objects, not JSON-safe
    strings — this is the one place that conversion happens so every
    endpoint returning a ticket does it the same way."""
    if not ticket:
        return None
    out = dict(ticket)
    for field in ("created_at", "resolved_at"):
        if out.get(field) is not None:
            out[field] = out[field].isoformat()
    return out


@router.get("/tickets")
def list_open_tickets(current: TokenPayload = Depends(get_current_token)):
    """All currently-open support tickets. Requires a support_agent token.
    Complements /support/pending: pending is "what's paused on the graph
    right now", this is "what tickets exist" — useful once a demo or real
    deployment wants a ticket queue view independent of interrupt state."""
    require_support_agent(current)
    try:
        tickets = ticket_store.list_open_tickets()
    except Exception:
        logger.exception("Failed to list open tickets")
        raise HTTPException(status_code=503, detail="Could not read ticket store.")
    return {"tickets": [_serialize_ticket(t) for t in tickets]}


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, current: TokenPayload = Depends(get_current_token)):
    """A single ticket's detail, regardless of open/resolved status.
    Requires a support_agent token."""
    require_support_agent(current)
    try:
        ticket = ticket_store.get_ticket(ticket_id)
    except Exception:
        logger.exception("Failed to load ticket %s", ticket_id)
        raise HTTPException(status_code=503, detail="Could not read ticket store.")
    if not ticket:
        raise HTTPException(status_code=404, detail="No ticket found with that id.")
    return _serialize_ticket(ticket)


@router.get("/thread/{thread_id}")
def get_thread(thread_id: str, current: TokenPayload = Depends(get_current_token)):
    """Full escalation conversation for a thread — the support agent's
    detail view. Includes whether it's still pending (open interrupt) so
    the frontend knows when to stop polling. Requires a support_agent token."""
    require_support_agent(current)
    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}
    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read state for thread %s", thread_id)
        raise HTTPException(status_code=503, detail="Could not reach state store.")

    interrupts = getattr(state, "interrupts", ())
    pending = bool(interrupts)
    ticket = None
    if pending:
        payload = interrupts[0].value or {}
        ticket_id = payload.get("ticket_id")
        if ticket_id:
            try:
                ticket = _serialize_ticket(ticket_store.get_ticket(ticket_id))
            except Exception:
                logger.warning("Failed to load ticket %s for thread %s", ticket_id, thread_id, exc_info=True)
    messages = escalation_store.get_messages(thread_id)
    return {"thread_id": thread_id, "pending": pending, "ticket": ticket, "messages": messages}


@router.post("/thread/{thread_id}/reply")
@limiter.limit("30/minute")
async def reply_to_thread(request: Request, thread_id: str, req: ReplyRequest, current: TokenPayload = Depends(get_current_token)):
    """Support agent sends a message to the user WITHOUT resolving yet —
    for gathering more info before writing the final resolution. Pushed
    live via the user's open websocket if connected. Requires a
    support_agent token."""
    require_support_agent(current)
    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}
    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read state for thread %s", thread_id)
        raise HTTPException(status_code=503, detail="Could not reach state store.")

    if not getattr(state, "interrupts", ()):
        raise HTTPException(status_code=404, detail="This thread isn't currently escalated.")

    escalation_store.add_message(thread_id, "support", req.text)
    try:
        delivered = await manager.notify(thread_id, {"status": "message", "sender": "support", "text": req.text})
    except Exception:
        logger.warning("Failed to push message to thread %s", thread_id, exc_info=True)
        delivered = False

    return {"status": "sent", "delivered_to_user": delivered}


@router.post("/resolve/{thread_id}")
@limiter.limit("20/minute")
def resolve(request: Request, thread_id: str, req: ResolveRequest, current: TokenPayload = Depends(get_current_token)):
    """Support agent resumes a paused thread with their resolution.
    Requires a support_agent token.

    The reply that comes back from this — and the one pushed to the user —
    is the support agent's resolution text verbatim, not an LLM
    paraphrase of it. See route_after_tools in app/graph/graph.py for why.

    If the resolution looks money-related (a refund, credit, chargeback,
    or currency amount — see money_detection.py) and confirmed=False,
    this returns a confirmation_required response instead of actually
    resuming the graph. Nothing is touched — no message logged, no
    interrupt resumed — until the agent resubmits with confirmed=true.
    This is enforced server-side, not just in the UI, so a request straight
    to the API can't skip the confirmation step either.
    """
    require_support_agent(current)

    if money_detection.looks_money_related(req.resolution) and not req.confirmed:
        return {
            "status": "confirmation_required",
            "message": "This resolution appears to involve a refund, credit, or monetary "
                       "amount. Confirm to proceed.",
        }

    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": thread_id, "user_id": thread_id}}

    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read state for thread %s", thread_id)
        raise HTTPException(status_code=503, detail="Could not reach state store.")

    if not getattr(state, "interrupts", ()):
        raise HTTPException(status_code=404, detail="No pending interrupt for this thread.")

    escalation_store.add_message(thread_id, "support", req.resolution)

    resume_start = time.monotonic()
    resume_command = Command(resume={"data": req.resolution})
    last_messages = []
    try:
        for event in graph_app.stream(resume_command, config=config, stream_mode="values"):
            if "messages" in event:
                last_messages = event["messages"]
    except Exception:
        logger.exception("Resume failed for thread %s", thread_id)
        raise HTTPException(
            status_code=502,
            detail="Resuming the agent failed (upstream model or DB error). "
                   "The thread is still marked as pending — safe to retry.",
        )

    log_event(
        "resolved_by_human", thread_id,
        resolution_latency_ms=int((time.monotonic() - resume_start) * 1000),
    )
    reply = last_messages[-1].content if last_messages else ""
    log_response(logger, thread_id, reply, status="resolved_by_human")

    # Push the resolution to the original user if they have an open
    # websocket connection (POST /chat/{user_id} already returned and closed
    # by the time an escalation is resolved, so there's no live request to
    # write the answer back to — this is a separate connection for exactly
    # that handoff). If they're not connected, they'll pick it up via
    # GET /chat/{thread_id}/status instead.
    try:
        delivered = asyncio.run(manager.notify(thread_id, {"status": "resolved", "reply": reply}))
    except Exception:
        logger.warning("Failed to push resolution to thread %s", thread_id, exc_info=True)
        delivered = False

    return {"status": "resumed", "final_reply": reply, "delivered_to_user": delivered}


