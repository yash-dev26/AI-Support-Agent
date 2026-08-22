import time
import json
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AIMessage, ToolMessage

from app.core.deps import get_graph_app, log_event
from app.core.rate_limit import limiter
from app.core.auth import get_current_token, TokenPayload, require_matching_user, _decode
from app.core.agent_logging import get_logger, log_response, _classify_result
from app.core.idempotency import IdempotencyCache
from app.services.ws_manager import manager
from app.services import escalation_store
from app.services import guardrails

logger = get_logger("chat")
router = APIRouter(tags=["chat"])

# ~4000 chars is roughly 800-1000 tokens — generous for a real support
# question (even one pasting an error message or order details), while
# still bounding the worst case: without a cap, a single message could
# balloon the prompt sent to the LLM (cost) and, since it's persisted
# into the checkpointed thread, permanently inflate every future turn's
# context too (before _trim_to_recent_turns even gets a chance to help,
# since trimming operates on turn COUNT, not turn SIZE).
MAX_MESSAGE_LENGTH = 4000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # min_length=1 alone would still accept "   " (all whitespace) —
        # that's not a useful message and would burn a full guardrails +
        # graph invocation on nothing.
        if not v.strip():
            raise ValueError("message cannot be blank")
        return v


# Bounds how long a duplicate request from a flaky client is still
# treated as "the same request" — 5 minutes covers a real retry (a
# client that times out and retries within seconds to a few minutes),
# without holding results around indefinitely for keys that will never
# be reused. Separate cache instance from policy_engine's, deliberately
# — see app/core/idempotency.py's module docstring for why.
_idempotency_cache = IdempotencyCache(ttl_seconds=300.0)


def _sse(event: str, data: dict) -> str:
    """Formats one Server-Sent Event. SSE's wire format is just two lines
    of text ending in a blank line — no library needed for something this
    small, and pulling one in would be one more moving part for a format
    this simple."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _extract_tool_trace(new_messages: list) -> list[dict]:
    """Pulls a demo-friendly "what did the agent do" trace out of the
    messages produced during THIS turn (caller passes only the slice
    after the incoming HumanMessage, not the whole thread history).

    Deliberately reads this back out of the message list after the run
    completes, rather than adding a callback/streaming layer to nodes.py
    or tools.py, so the tool functions themselves stay untouched — this
    is purely a presentation-layer concern for the demo sidebar
    (app/graph/tools.py's actual behavior doesn't change either way).

    create_support_ticket is handled specially: when it calls interrupt(),
    execution pauses INSIDE the tool before a ToolMessage is ever produced
    for that call — so "no matching ToolMessage yet" means "escalated and
    waiting on a human", not "still running", for that one tool.
    """
    result_by_call_id = {
        m.tool_call_id: m.content for m in new_messages if isinstance(m, ToolMessage)
    }
    trace = []
    for m in new_messages:
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            name = tc["name"]
            call_id = tc.get("id")
            has_result = call_id in result_by_call_id
            if name == "create_support_ticket":
                status = "escalated"
            elif has_result:
                status = "completed"
            else:
                status = "pending"
            trace.append({"tool": name, "status": status})
    return trace


@router.post("/chat/{user_id}")
@limiter.limit("15/minute")
def chat(request: Request, user_id: str, req: ChatRequest, current: TokenPayload = Depends(get_current_token)):
    """Send a message to the agent for a given user. Returns either a normal
    reply or an escalation notice if the agent paused for human input.

    Requires a token whose sub matches user_id (or a support_agent token).
    Rate limited per user_id (15/min) — this is the endpoint that drives
    LLM calls, so it's the one actually worth protecting against a runaway
    client or accidental infinite-retry loop.

    Pass idempotency_key to make a retried request safe: a client that
    times out waiting for a response has no way to know whether the
    original request actually succeeded server-side. Retrying naively
    would append the SAME user message into the thread a second time,
    burn a second LLM call, and could double-trigger create_support_ticket.
    Passing the same idempotency_key on the retry returns the cached
    result from the first attempt instead of reprocessing it, for up to
    5 minutes after the first attempt.
    """
    require_matching_user(user_id, current)

    cache_key = f"{user_id}:{req.idempotency_key}" if req.idempotency_key else None
    if not cache_key:
        return _process_chat_message(user_id, req)

    return _idempotency_cache.get_or_compute(cache_key, lambda: _process_chat_message(user_id, req))


def _process_chat_message(user_id: str, req: ChatRequest) -> dict:
    """The actual per-turn logic for POST /chat/{user_id}, separated from
    the idempotency-cache wrapper in chat() above so a cache HIT can
    return in one line without this whole body needing an early-return
    guard wrapped around it."""
    try:
        blocked, refusal = asyncio.run(guardrails.check_message(req.message, thread_id=user_id))
    except Exception:
        logger.exception("Guardrails check raised unexpectedly for user %s — failing open", user_id)
        blocked, refusal = False, None

    if blocked:
        # Checked before the escalation-state branch below on purpose —
        # this should protect a human agent from injection/abuse in the
        # side-channel too, not just the LLM.
        log_event("blocked_by_guardrails", user_id)
        log_response(logger, user_id, refusal or "", status="blocked")
        return {"status": "blocked", "message": refusal}

    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}

    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read graph state for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not reach state store. Try again shortly.") from None

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

    prior_message_count = len(state.values.get("messages", []))
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
        ) from None

    tool_calls = _extract_tool_trace(last_messages[prior_message_count:])

    try:
        new_state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read post-run state for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not confirm run outcome. Check /support/pending.") from None

    if getattr(new_state, "interrupts", ()):
        payload = new_state.interrupts[0].value or {}
        escalation_store.add_message(user_id, "user", req.message)
        log_event("escalated", user_id, query=payload.get("query"))
        log_response(logger, user_id, payload.get("message", ""), status="escalated")
        return {
            "status": "escalated",
            "message": "Your request has been escalated to a human agent. Please wait.",
            "tool_calls": tool_calls,
            "ticket_id": payload.get("ticket_id"),
        }

    log_event("resolved_by_agent", user_id, latency_ms=int((time.monotonic() - start) * 1000))
    reply = last_messages[-1].content if last_messages else ""
    log_response(logger, user_id, reply, status="ok")
    return {"status": "ok", "reply": reply, "tool_calls": tool_calls}


@router.post("/chat/{user_id}/stream")
@limiter.limit("15/minute")
def chat_stream(request: Request, user_id: str, req: ChatRequest, current: TokenPayload = Depends(get_current_token)):
    """Same contract and side effects as POST /chat/{user_id} — same
    auth, same guardrail check, same escalation side-channel — but
    streams progress as Server-Sent Events instead of blocking until the
    whole turn finishes and returning one JSON blob.

    This replaced the demo frontend's earlier staged-badge simulation
    (see README's changelog): those badges
    were faked — the whole turn had already finished by the time the
    JSON response existed, so "Calling X..." was just an animated reveal
    of data that was already sitting there. These events are emitted the
    moment each LangGraph superstep actually completes
    (stream_mode="updates" — verified empirically, not assumed: each
    node's update chunk contains only what THAT node changed, which
    arrives from the generator as soon as it happens). If a tool takes
    2 seconds, its "tool_call_end" event is 2 real seconds behind its
    "tool_call_start", not an artificial 550ms later.

    Event types emitted, always ending in "done":
      - "blocked": {"message": "..."} — guardrails rejected the message
      - "already_escalated": {"message": "..."} — thread was already
        paused BEFORE this message arrived; routed to the human
        side-channel, no graph run happens
      - "tool_call_start": {"tool": name}
      - "tool_call_end": {"tool": name, "status": "completed"|"empty"|
        "degraded"|"no_answer"} — status reuses the same classification
        agent_logging.py's tool-call logging already uses, so the
        frontend badge and the server log agree on what "empty" vs
        "degraded" means
      - "agent_reply": {"text": "..."} — the model's final, non-tool-call
        reply for this turn
      - "escalated": {"ticket_id": ..., "message": "..."} —
        create_support_ticket paused the graph this turn; any tool call
        left pending at that point gets a "tool_call_end" with
        status="escalated" first (see the __interrupt__ handling below)
      - "error": {"message": "..."} — state store or agent run failed
      - "done": {} — always the final event no matter which path above
        was taken, so the frontend has one unambiguous "stop listening"
        signal

    Supports the same idempotency_key mechanism as POST /chat/{user_id}
    (see ChatRequest and _idempotency_cache) — a cache hit replays the
    exact same event sequence instantly instead of re-running the graph.

    One real difference from POST /chat/{user_id}'s idempotency handling:
    that endpoint uses IdempotencyCache.get_or_compute, which is safe
    against genuinely CONCURRENT retries (two requests with the same key
    arriving before either has finished) via a per-key lock. This
    endpoint only caches AFTER a run completes — safe against a
    sequential retry (client times out, waits, retries), but two
    simultaneous requests with the same key here could both trigger a
    real graph run. Coalescing concurrent streaming requests properly
    would mean the second caller blocks until the first's stream
    finishes, then replays it — solvable, but genuinely more complex for
    a live SSE response than for a single blocking JSON one, and out of
    scope for this pass. Documented here rather than silently assumed
    equivalent to the non-streaming endpoint's guarantee.
    """
    require_matching_user(user_id, current)

    cache_key = f"{user_id}:{req.idempotency_key}" if req.idempotency_key else None
    if cache_key:
        cached_events = _idempotency_cache.get(cache_key)
        if cached_events is not None:
            log_event("idempotent_replay", user_id)

            def replay():
                for event, data in cached_events:
                    yield _sse(event, data)
            return StreamingResponse(replay(), media_type="text/event-stream")

    recorded: list[tuple[str, dict]] = []

    def event_stream():
        for event, data in _generate_chat_stream_events(user_id, req):
            recorded.append((event, data))
            yield _sse(event, data)
        if cache_key:
            _idempotency_cache.set(cache_key, recorded)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _generate_chat_stream_events(user_id: str, req: ChatRequest):
    """The actual per-superstep narration logic for chat_stream, as a
    plain generator of (event_type, data_dict) tuples rather than
    pre-formatted SSE text. Kept separate from _sse() formatting and from
    chat_stream() itself so idempotent replay (see chat_stream above) can
    record and replay the same (event, data) pairs directly, without
    needing to re-serialize/re-parse JSON on every replay."""
    try:
        blocked, refusal = asyncio.run(guardrails.check_message(req.message, thread_id=user_id))
    except Exception:
        logger.exception("Guardrails check raised unexpectedly for user %s — failing open", user_id)
        blocked, refusal = False, None

    if blocked:
        log_event("blocked_by_guardrails", user_id)
        log_response(logger, user_id, refusal or "", status="blocked")
        yield ("blocked", {"message": refusal})
        yield ("done", {})
        return

    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}

    try:
        state = graph_app.get_state(config)
    except Exception:
        logger.exception("Failed to read graph state for user %s", user_id)
        yield ("error", {"message": "Could not reach state store. Try again shortly."})
        yield ("done", {})
        return

    if getattr(state, "interrupts", ()):
        escalation_store.add_message(user_id, "user", req.message)
        yield ("already_escalated", {"message": "Message sent to support. Waiting for a reply..."})
        yield ("done", {})
        return

    start = time.monotonic()
    # tool_call_id -> tool name, for calls whose ToolMessage hasn't
    # streamed yet. Anything still in here when the loop ends because
    # of an interrupt (rather than the graph reaching END normally)
    # is, by construction, exactly the create_support_ticket call
    # that paused execution before it could produce one.
    pending_calls: dict[str, str] = {}
    final_reply = ""

    try:
        for chunk in graph_app.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value or {}
                for call_id, name in list(pending_calls.items()):
                    yield ("tool_call_end", {"tool": name, "status": "escalated"})
                    pending_calls.pop(call_id, None)
                escalation_store.add_message(user_id, "user", req.message)
                log_event("escalated", user_id, query=payload.get("query"))
                log_response(logger, user_id, payload.get("message", ""), status="escalated")
                yield ("escalated", {
                    "ticket_id": payload.get("ticket_id"),
                    "message": "Your request has been escalated to a human agent. Please wait.",
                })
                yield ("done", {})
                return

            for node_name, node_output in chunk.items():
                node_messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
                if node_name == "chatbot":
                    for msg in node_messages:
                        if not isinstance(msg, AIMessage):
                            continue
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                pending_calls[tc["id"]] = tc["name"]
                                yield ("tool_call_start", {"tool": tc["name"]})
                        else:
                            final_reply = msg.content
                            yield ("agent_reply", {"text": msg.content})
                else:  # "tools" node
                    for msg in node_messages:
                        if not isinstance(msg, ToolMessage):
                            continue
                        name = pending_calls.pop(msg.tool_call_id, "unknown_tool")
                        status = _classify_result(msg.content)
                        yield ("tool_call_end", {"tool": name, "status": status})
    except Exception:
        logger.exception("Agent run failed for user %s", user_id)
        yield ("error", {"message": "The agent failed to process this message. Please retry."})
        yield ("done", {})
        return

    log_event("resolved_by_agent", user_id, latency_ms=int((time.monotonic() - start) * 1000))
    log_response(logger, user_id, final_reply, status="ok")
    yield ("done", {})


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
        raise HTTPException(status_code=503, detail="Could not reach state store. Try again shortly.") from None

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


