import time
import logging
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.deps import get_graph_app, log_event
from app.core.rate_limit import limiter
from app.core.auth import get_current_token, TokenPayload, require_matching_user, _decode
from app.services.ws_manager import manager
from app.services import escalation_store
from app.services import guardrails

logger = logging.getLogger("support_agent.chat")
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/chat/{user_id}")
@limiter.limit("15/minute")
def chat(request: Request, user_id: str, req: ChatRequest, current: TokenPayload = Depends(get_current_token)):
    """Send a message to the agent for a given user. Returns either a normal
    reply or an escalation notice if the agent paused for human input.

    Requires a token whose sub matches user_id (or a support_agent token).
    Rate limited per user_id (15/min) — this is the endpoint that drives
    LLM calls, so it's the one actually worth protecting against a runaway
    client or accidental infinite-retry loop.
    """
    require_matching_user(user_id, current)

    try:
        blocked, refusal = asyncio.run(guardrails.check_message(req.message))
    except Exception:
        logger.exception("Guardrails check raised unexpectedly for user %s — failing open", user_id)
        blocked, refusal = False, None

    if blocked:
        # Checked before the escalation-state branch below on purpose —
        # this should protect a human agent from injection/abuse in the
        # side-channel too, not just the LLM.
        log_event("blocked_by_guardrails", user_id)
        return {"status": "blocked", "message": refusal}

    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}

    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read graph state for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not reach state store. Try again shortly.")

    if getattr(state, "interrupts", ()):
        # Once a thread is escalated, the user talks ONLY to the human —
        # this message never reaches the LLM/graph again until the support
        # agent resolves it. It goes into the escalation side-channel
        # instead, where the support agent sees it in their thread view
        # (GET /support/thread/{thread_id}) and can reply directly.
        escalation_store.add_message(user_id, "user", req.message)
        return {
            "status": "escalated",
            "message": "Message sent to support. Waiting for a reply...",
        }

    start = time.monotonic()
    last_messages = []
    try:
        for chunk in graph_app.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode="values",
        ):
            if "messages" in chunk:
                last_messages = chunk["messages"]
    except Exception:
        logger.exception("Agent run failed for user %s", user_id)
        raise HTTPException(
            status_code=502,
            detail="The agent failed to process this message (upstream model or DB error). "
                   "No state was corrupted; please retry.",
        )

    try:
        new_state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read post-run state for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not confirm run outcome. Check /support/pending.")

    if getattr(new_state, "interrupts", ()):
        payload = new_state.interrupts[0].value or {}
        escalation_store.add_message(user_id, "user", req.message)
        log_event("escalated", user_id, query=payload.get("query"))
        return {
            "status": "escalated",
            "message": "Your request has been escalated to a human agent. Please wait.",
        }

    log_event("resolved_by_agent", user_id, latency_ms=int((time.monotonic() - start) * 1000))
    reply = last_messages[-1].content if last_messages else ""
    return {"status": "ok", "reply": reply}


@router.get("/chat/{user_id}/status")
def chat_status(user_id: str, current: TokenPayload = Depends(get_current_token)):
    """Polling fallback for clients that don't hold a websocket open.
    Lets a user (or their frontend) check whether an escalated request has
    since been resolved by a human, without needing a persistent connection.
    """
    require_matching_user(user_id, current)
    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}

    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read graph state for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not reach state store. Try again shortly.")

    if not state.values.get("messages"):
        return {"status": "no_conversation"}

    if getattr(state, "interrupts", ()):
        payload = state.interrupts[0].value or {}
        return {"status": "escalated", "query": payload.get("query")}

    last_message = state.values["messages"][-1]
    return {"status": "resolved", "reply": getattr(last_message, "content", "")}


@router.websocket("/ws/{user_id}")
async def chat_ws(websocket: WebSocket, user_id: str, token: str | None = None):
    """Holds a connection open so a resolved escalation can be pushed to the
    user in real time instead of requiring them to poll /chat/{user_id}/status.
    Push-only from the server's side — any client messages received here are
    ignored; send actual chat messages via POST /chat/{user_id}.

    Auth travels as ?token=... rather than an Authorization header — browsers'
    native WebSocket API can't set custom headers on the handshake, so a
    query param is the practical option for a browser client. Handled
    manually (not via a Depends()) so an invalid/missing token gets a clean
    websocket close rather than relying on HTTP-exception-handling machinery
    that isn't guaranteed to behave the same way for websocket routes.
    """
    if not token:
        await websocket.close(code=1008)  # policy violation
        return
    try:
        current = _decode(token)
    except HTTPException:
        await websocket.close(code=1008)
        return
    if current.role != "support_agent" and current.sub != user_id:
        await websocket.close(code=1008)
        return

    await manager.connect(user_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id)


@router.get("/ws-test/{user_id}", response_class=HTMLResponse)
def ws_test_page(user_id: str):
    """A plain HTML/JS debug page — NOT a product frontend — that opens the
    websocket and prints whatever it receives. Exists purely so the push
    mechanism can be watched working in a browser tab without installing a
    websocket client (wscat, etc). Swagger's /docs can't exercise websocket
    routes at all, so this fills that specific gap.

    Mints its own token client-side (POST /auth/token) before connecting,
    now that /ws/{user_id} requires one — keeps this debug tool actually
    functional rather than silently broken by the auth requirement.
    """
    return f"""<!DOCTYPE html>
<html>
<head><title>ws test — {user_id}</title></head>
<body style="font-family: monospace; padding: 2rem; background: #111; color: #ddd;">
  <h3>Listening on /ws/{user_id}</h3>
  <p>Trigger an escalation for this user via POST /chat/{user_id}, then resolve
     it via POST /support/resolve/{user_id} in another tab — the resolution
     should appear below the moment it's resolved.</p>
  <pre id="log" style="background:#000; padding:1rem; min-height:200px; white-space:pre-wrap;"></pre>
  <script>
    const log = document.getElementById("log");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";

    fetch("/auth/token", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ user_id: "{user_id}", role: "user" }}),
    }})
      .then(r => r.json())
      .then(data => {{
        const ws = new WebSocket(`${{proto}}//${{location.host}}/ws/{user_id}?token=${{data.access_token}}`);
        ws.onopen = () => log.textContent += "[connected]\\n";
        ws.onmessage = (e) => log.textContent += "[received] " + e.data + "\\n";
        ws.onclose = () => log.textContent += "[disconnected]\\n";
        ws.onerror = (e) => log.textContent += "[error] " + e + "\\n";
      }})
      .catch(e => log.textContent += "[failed to get auth token] " + e + "\\n");
  </script>
</body>
</html>"""
