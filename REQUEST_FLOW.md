# Request Flow

What actually happens, step by step, for every flow that matters in this
project. Each section references real files and function names — read this
alongside the code, not instead of it.

---

## 1. Authentication

Every protected endpoint requires a JWT. There's no login form —
`POST /auth/token` (`app/routers/auth.py`) issues a token for whatever
`user_id`/`role` is requested, no password check. See
[`DECISIONS.md`](./DECISIONS.md#auth-is-a-stub-but-enforcement-is-real) for
why that's an intentional tradeoff.

```
POST /auth/token {"user_id": "user_001", "role": "user"}
        |
        v
app/core/auth.py: create_access_token()
        |  signs {"sub": "user_001", "role": "user", "exp": ...} with
        |  AUTH_SECRET_KEY (env var, or a random one generated at startup)
        v
{"access_token": "<jwt>", "token_type": "bearer", "expires_in": 3600}
```

Every subsequent request carries this token:
- HTTP routes: `Authorization: Bearer <token>` header
- The websocket route: `?token=<token>` query param (browsers can't set
  custom headers on a WebSocket handshake)

Two roles, enforced by `app/core/auth.py`:
- **`user`** — `require_matching_user()` only allows the token to act as its
  own `sub` (`user_id`). A `user_001` token can't touch `/chat/user_002`.
- **`support_agent`** — `require_support_agent()` gates every `/support/*`
  endpoint and `/docs/upload`. A `support_agent` token can also act on
  behalf of any `user_id` in the chat endpoints (an agent debugging a
  customer's session).

---

## 2. A normal chat message (no escalation)

Two ways to hit this path: `POST /chat/{user_id}` (blocking, one JSON
response — used by `GET /chat/{user_id}/status`'s polling fallback and most
of the test suite) or `POST /chat/{user_id}/stream` (Server-Sent Events,
what the demo frontend uses — narrates progress as it happens). Same auth,
same guardrail check, same escalation-state check, same graph. The
streaming version narrates each LangGraph superstep as it arrives
(`stream_mode="updates"`) instead of returning once at the end. Traced below
for the streaming version; the non-streaming one is identical minus the
event narration.

```
POST /chat/user_001/stream {"message": "What's your refund policy?"}
Authorization: Bearer <user token>
        |
        v
app/routers/chat.py: chat_stream()
        |
        |- 0. Pydantic validates the body: message must be 1-4000 chars
        |      and not whitespace-only; idempotency_key (optional) capped
        |      at 128 chars. A failure here is a 422 from FastAPI's
        |      validation layer, not app code.
        |
        |- 1. Rate limit check (slowapi, 15/min per user_id)
        |
        |- 2. require_matching_user(user_id, token) -- 403 if mismatched
        |
        |- 2a. If idempotency_key was provided and its cache entry is
        |      still fresh (<5 min): replay the same event sequence from
        |      the first attempt and return -- nothing below runs again.
        |      Guards sequential retries for /stream. POST /chat/{user_id}
        |      uses a per-key lock and is also safe under concurrent
        |      retries -- see the note in chat_stream's docstring for the
        |      difference between the two.
        |
        |- 3. guardrails.check_message(message)
        |      NeMo Guardrails input rail (app/services/guardrails.py) --
        |      one LLM call, checks for prompt injection / off-topic.
        |      Blocked -> event "blocked", then "done". Stream ends here.
        |
        |- 4. graph_app.get_state(thread_id) -- check for an existing
        |      open interrupt. None here, so continue.
        |
        |- 5. graph_app.stream({"messages": [...]}, stream_mode="updates")
        |      -- the LangGraph turn, yielding one chunk per completed
        |      superstep:
        |
        |      app/graph/nodes.py: chatbot()
        |        |
        |        |- prepends SYSTEM_PROMPT if not already present
        |        |- _trim_to_recent_turns(): drops turns beyond the most
        |        |    recent MAX_CONTEXT_TURNS (default 12, env-
        |        |    configurable), bounding prompt size/cost/latency for
        |        |    a long-running thread. Cuts on whole HumanMessage-
        |        |    started turns only, never mid-tool-call -- an
        |        |    AIMessage(tool_calls=...) without its matching
        |        |    ToolMessage(s) is rejected outright by the OpenAI API.
        |        |- invokes GPT-4.1 with the trimmed history + tools,
        |        |    wrapped in a retry/backoff decorator (app/core/retry.py):
        |        |    connection errors, rate limits, and upstream 5xxs
        |        |    retry with exponential backoff + jitter (up to 3
        |        |    attempts); auth/bad-request errors are not retried,
        |        |    since retrying those fails identically again. System
        |        |    prompt also tells the model to request independent
        |        |    tool calls together in one turn (e.g. cart + order),
        |        |    since ToolNode already executes a turn's tool calls
        |        |    concurrently via a thread pool.
        |        |- if the response carries usage_metadata, logs token
        |        |    counts locally ([AGENT] token usage: N in / N out /
        |        |    N total) as a fallback that works without LangSmith
        |        |    configured. If LANGSMITH_TRACING+LANGSMITH_API_KEY
        |        |    are set, this call (and every other LLM/tool call in
        |        |    the turn) is also traced to LangSmith automatically
        |        |    via LangChain's own callback system.
        |        \- model decides to call check_policy(query)
        |             <<< chunk yielded: {"chatbot": {"messages": [AIMessage(tool_calls=[...])]}}
        |             >>> stream emits: event "tool_call_start" {"tool": "check_policy"}
        |
        |      app/graph/tools.py: check_policy()
        |        \- app/services/policy_engine.py: answer()
        |             |
        |             |- faq.match_faq(query) -- keyword overlap, no LLM
        |             |    call, never cached (already sub-ms). Hit ->
        |             |    return "[FAQ] ..." immediately.
        |             |
        |             \- Miss -> _cached_answer(normalized_query):
        |                  |- cache hit (same question asked before,
        |                  |    case/whitespace-normalized) -> return the
        |                  |    cached result, zero network calls
        |                  \- cache miss -> vector_store.search(query)
        |                       (Qdrant, embedded mode) -> top-k paragraphs
        |                       with score >= 0.35 -> one more LLM call to
        |                       generate a cited answer from that context
        |                       only (or NO_ANSWER_FOUND if the model can't
        |                       answer from what it retrieved) -- result
        |                       cached for next time, unless the retrieval
        |                       call itself raised (a transient outage is
        |                       never cached as a permanent answer)
        |             <<< chunk yielded: {"tools": {"messages": [ToolMessage(...)]}}
        |             >>> stream emits: event "tool_call_end" {"tool": "check_policy", "status": "completed"}
        |
        |      app/graph/graph.py: route_after_tools()
        |        \- tool was check_policy (not create_support_ticket), so
        |             route back to chatbot -- the model turns the tool's
        |             raw result into a final reply
        |             <<< chunk yielded: {"chatbot": {"messages": [AIMessage(content="...")]}}
        |             >>> stream emits: event "agent_reply" {"text": "Orders can be refunded within 14 days..."}
        |
        v
event: done
{}
```

LLM calls for this path: 1 (guardrails) + 1 (chatbot deciding to call
`check_policy`) + 0 or 1 (RAG generation, only on a cache miss and FAQ
miss) + 1 (chatbot turning the tool result into a reply) = 2-4 calls. The
RAG generation call drops out entirely on a cache hit for a repeated
question.

---

## 3. Escalation — triggered by policy, not by request

The system prompt (`app/graph/nodes.py`) is explicit: `create_support_ticket`
is called only when `check_policy` returns a result starting with
`NO_ANSWER_FOUND`, or the issue genuinely needs manual action no tool
covers. A bare "let me talk to a human" is not itself sufficient.

```
POST /chat/user_001/stream {"message": "My payment was charged twice, order X"}
        |
        |- guardrails check: passes (legitimate complaint)
        |
        |- graph_app.stream(..., stream_mode="updates") -- chatbot calls
        |   check_policy("duplicate charge")
        |   >>> stream emits: event "tool_call_start" {"tool": "check_policy"}
        |
        |   policy_engine.answer() -> cache miss -> vector_store.search()
        |   finds the relevant paragraph in refund_policy.md:
        |     "If a customer reports a duplicate or incorrect charge,
        |      this requires manual verification ... escalated to a
        |      human support agent"
        |   -> RAG generation cites this, result cached, model has real
        |      grounds to escalate
        |   >>> stream emits: event "tool_call_end" {"tool": "check_policy", "status": "completed"}
        |
        |- chatbot calls create_support_ticket(issue_type="duplicate_charge",
        |     details="Customer reports being charged twice for order X...")
        |   >>> stream emits: event "tool_call_start" {"tool": "create_support_ticket"}
        |
        |   app/graph/tools.py: create_support_ticket()
        |     |- ticket_id = f"tkt_{tool_call_id}" -- deterministic from
        |     |    the tool call's own id (InjectedToolCallId), not
        |     |    uuid4(). interrupt() causes LangGraph to re-run this
        |     |    entire function from the top on every resume (see
        |     |    langgraph.types.interrupt's docstring) -- a random id
        |     |    here would mint a fresh duplicate ticket on every human
        |     |    reply. See the README changelog for the bug this was.
        |     |- ticket_store.create_ticket(ticket_id, ...) -- an idempotent
        |     |    upsert ($setOnInsert), so a replay with the same
        |     |    ticket_id is a no-op if it already exists
        |     \- interrupt({"query": ..., "ticket_id": ..., "message": ...})
        |          LangGraph pauses execution here. State is checkpointed
        |          to MongoDB (thread-scoped, via langgraph-checkpoint-mongodb),
        |          with a TTL index (CHECKPOINT_TTL_SECONDS, default 30
        |          days). If this checkpoint outlives that window still
        |          unresolved, MongoDB expires it automatically and the
        |          thread starts fresh on the user's next message. The
        |          ticket (a separate collection) does not expire alongside
        |          it -- see README's Known Limitations. This can only be
        |          resumed once per interrupt.
        |
        |   >>> the graph.stream() generator's next chunk is a distinct
        |       "__interrupt__" key rather than a "tools" node update,
        |       since no ToolMessage was ever produced for this call:
        |       {"__interrupt__": (Interrupt(value={"query": ..., "ticket_id": ..., "message": ...}),)}
        |   >>> stream emits: event "tool_call_end" {"tool": "create_support_ticket", "status": "escalated"}
        |   >>> stream emits: event "escalated" {"ticket_id": "tkt_...", "message": "Your request has been escalated..."}
        |
        v
graph_app.get_state() now shows an open interrupt.
escalation_store.add_message(user_id, "user", original_message)
        |  (app/services/escalation_store.py -- the side-channel, separate
        |   from the graph's own message history)
        v
event: done
{}
```

→ See [`DECISIONS.md`: The escalation side-channel exists because `interrupt()` can only resume once](./DECISIONS.md#the-escalation-side-channel-exists-because-interrupt-can-only-resume-once)

---

## 4. While escalated — the side-channel

Once a thread has an open interrupt, `chat.py`'s state check catches it and
reroutes before touching the graph or the LLM at all:

```
POST /chat/user_001 {"message": "Are you there? Order is ORD-999"}
        |
        |- guardrails check still runs (protects the human agent too)
        |
        |- graph_app.get_state(thread_id).interrupts is non-empty
        |      -> skip the graph entirely
        |
        |- escalation_store.add_message(user_id, "user", message)
        |
        v
{"status": "escalated", "message": "Message sent to support. Waiting..."}
```

The support agent side, viewing the thread:

```
GET /support/thread/user_001
Authorization: Bearer <support_agent token>
        |
        |- require_support_agent(token) -- 403 for a "user" token
        |- graph_app.get_state() -> pending: true
        |- escalation_store.get_messages(user_id) -> full transcript
        v
{"thread_id": "user_001", "pending": true, "messages": [...]}
```

The agent can reply without resolving, to gather more info:

```
POST /support/thread/user_001/reply {"text": "Checking on ORD-999 now."}
        |
        |- escalation_store.add_message(thread_id, "support", text)
        |- ws_manager.notify(thread_id, {"status": "message", ...})
        |      -> pushed live if the user has /ws/user_001 open
        v
{"status": "sent", "delivered_to_user": true|false}
```

---

## 5. Resolving — money confirmation, verbatim delivery, LLM path reopens

```
POST /support/resolve/user_001 {"resolution": "Refund issued for $50."}
        |
        |- require_support_agent(token)
        |
        |- money_detection.looks_money_related(resolution) -> True
        |   (currency regex + keyword list, app/services/money_detection.py)
        |   confirmed is False (default) ->
        |
        v
{"status": "confirmation_required", "message": "..."}
        |   Nothing has happened yet -- no graph resume, no message logged,
        |   no push. The frontend shows a confirm() dialog; if accepted:
        v
POST /support/resolve/user_001 {"resolution": "Refund issued for $50.", "confirmed": true}
        |
        |- escalation_store.add_message(thread_id, "support", resolution)
        |
        |- graph_app.stream(Command(resume={"data": resolution}))
        |      |
        |      |  create_support_ticket()'s interrupt() call returns, with
        |      |  resolution as its return value -- the tool "result"
        |      |
        |      v
        |  app/graph/graph.py: route_after_tools()
        |      |  the tool that just ran was create_support_ticket ->
        |      |  route straight to END, skip chatbot entirely. The
        |      |  resolution reaches the user verbatim -- no second LLM
        |      |  call re-paraphrasing a refund amount.
        |      v
        |  last message in state = the resolution text itself
        |
        |- ws_manager.notify(thread_id, {"status": "resolved", "reply": resolution})
        |      -> pushed live if connected; otherwise the user picks it up
        |        via GET /chat/{user_id}/status on their next poll
        |
        v
{"status": "resumed", "final_reply": "Refund issued for $50.", "delivered_to_user": true}
```

→ See [`DECISIONS.md`: `route_after_tools` skips the LLM after a human resolves](./DECISIONS.md#route_after_tools-skips-the-llm-after-a-human-resolves) and [Money-related resolutions require server-enforced confirmation](./DECISIONS.md#money-related-resolutions-require-server-enforced-confirmation)

After this, `graph_app.get_state(thread_id).interrupts` is empty again — the
next `POST /chat/user_001` goes back to the full LLM path in section 2.

---

## 6. Real-time push — the websocket lifecycle

```
Frontend: after receiving {"status": "escalated"}, immediately opens:
  new WebSocket(`wss://.../ws/user_001?token=<user_token>`)
        |
        v
app/routers/chat.py: chat_ws()
        |- manually decodes ?token=... (not a Depends() -- a failed/missing
        |    token gets a clean websocket close code 1008, not an HTTP
        |    exception, since the two don't share error-handling machinery)
        |- checks current.sub == user_id (or role == support_agent)
        |- ws_manager.connect(user_id, websocket)
        |    (app/services/ws_manager.py -- an in-memory dict, thread_id ->
        |     websocket, single-process only)
        v
Connection held open. Server pushes to it from:
  - support.py's reply_to_thread() -> {"status": "message", ...}
  - support.py's resolve() -> {"status": "resolved", ...}
Frontend closes the connection itself after receiving "resolved".
```

→ See [`DECISIONS.md`: In-memory websocket connections do not need a message queue yet](./DECISIONS.md#in-memory-websocket-connections-do-not-need-a-message-queue-yet)

---

## 7. SSE event catalog (`POST /chat/{user_id}/stream`)

Every event type `chat_stream` (`app/routers/chat.py`) can emit, in the
order a single turn can produce them. Every stream ends with `done`
regardless of path, so a client only needs to listen for one signal.

Both `POST /chat/{user_id}` and this streaming variant accept an optional
`idempotency_key` in the request body — see section 2, steps 0 and 2a. A
retried request with the same key replays the original result (a cached
JSON response for the non-streaming endpoint, the same event sequence for
this one) instead of reprocessing.

| event | data | when |
|---|---|---|
| `blocked` | `{"message": str}` | guardrails rejected the message; nothing else runs |
| `already_escalated` | `{"message": str}` | thread was already paused before this message; routed to the human side-channel, no graph run |
| `tool_call_start` | `{"tool": str}` | the model requested this tool call (fires once per call, even when several are requested in the same turn) |
| `tool_call_end` | `{"tool": str, "status": str}` | that tool call's `ToolMessage` arrived. `status` is one of `completed` / `empty` / `degraded` / `no_answer` / `escalated` — the same classification `agent_logging.py`'s `[TOOL CALL]` log lines use |
| `agent_reply` | `{"text": str}` | the model's final, non-tool-call reply for this turn |
| `escalated` | `{"ticket_id": str \| None, "message": str}` | `create_support_ticket` paused the graph this turn — always preceded by a `tool_call_end` with `status="escalated"` for that same call |
| `error` | `{"message": str}` | the state store or the graph run itself failed |
| `done` | `{}` | always last |

Two calls requested in the same turn (see section 2's parallel-tool-call
note) show up as two consecutive `tool_call_start` events followed by two
consecutive `tool_call_end` events, not interleaved — both calls arrive
together in one `chatbot` node update, execute concurrently via `ToolNode`'s
thread pool, and their results arrive together in one `tools` node update.
