# Request Flow

A walkthrough of what actually happens, step by step, for the flows that
matter in this project. Each section references real files and function
names — this is meant to be read alongside the code, not instead of it.

---

## 1. Authentication

Every protected endpoint requires a JWT. There's no real login form —
`POST /auth/token` (`app/routers/auth.py`) issues a token for whatever
`user_id`/`role` is requested, no password check. See
[`DECISIONS.md`](./DECISIONS.md#auth-is-a-stub-but-enforcement-is-real)
for why that's an honest tradeoff, not an oversight.

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
  custom headers on a WebSocket handshake, so this is the practical option)

Two roles, enforced by `app/core/auth.py`:
- **`user`** — `require_matching_user()` only allows the token to act as
  its own `sub` (`user_id`). A `user_001` token can't touch `/chat/user_002`.
- **`support_agent`** — `require_support_agent()` gates every `/support/*`
  endpoint and `/docs/upload`. A `support_agent` token can also act on
  behalf of *any* `user_id` in the chat endpoints (an agent debugging a
  customer's session).

---

## 2. A normal chat message (no escalation)

```
POST /chat/user_001 {"message": "What's your refund policy?"}
Authorization: Bearer <user token>
        |
        v
app/routers/chat.py: chat()
        |
        |- 1. Rate limit check (slowapi, 15/min per user_id)
        |
        |- 2. require_matching_user(user_id, token) -- 403 if mismatched
        |
        |- 3. guardrails.check_message(message)
        |      NeMo Guardrails input rail (app/services/guardrails.py) --
        |      one LLM call, checks for prompt injection / off-topic.
        |      If blocked: return {"status": "blocked", ...} HERE.
        |      Nothing below this point runs.
        |
        |- 4. graph_app.get_state(thread_id) -- check for an existing
        |      open interrupt. None here, so continue.
        |
        |- 5. graph_app.stream({"messages": [...]}) -- the actual
        |      LangGraph turn:
        |
        |      app/graph/nodes.py: chatbot()
        |        |
        |        |- prepends SYSTEM_PROMPT if not already present
        |        |- invokes GPT-4.1 with the message history + tools
        |        \- model decides to call check_policy(query)
        |
        |      app/graph/tools.py: check_policy()
        |        \- app/services/policy_engine.py: answer()
        |             |
        |             |- faq.match_faq(query) -- keyword overlap,
        |             |    no LLM call. HIT -> return "[FAQ] ..." immediately.
        |             |
        |             \- MISS -> vector_store.search(query) (Qdrant,
        |                  embedded mode) -> top-k paragraphs with a
        |                  score >= 0.35 -> one more LLM call to generate
        |                  a cited answer from ONLY that context, or
        |                  return the NO_ANSWER_FOUND signal if the
        |                  model can't answer from what it retrieved
        |
        |      app/graph/graph.py: route_after_tools()
        |        \- tool was check_policy (not human_interrupt_tool),
        |             so route back to chatbot -- the model turns the
        |             tool's raw result into a final reply
        |
        v
{"status": "ok", "reply": "Orders can be refunded within 14 days..."}
```

Total LLM calls for this path: 1 (guardrails) + 1 (chatbot deciding to
call `check_policy`) + 0 or 1 (RAG generation, only on an FAQ miss) + 1
(chatbot turning the tool result into a reply) = 3-4 calls.

---

## 3. Escalation -- triggered by policy, not by request

The system prompt (`app/graph/nodes.py`) is explicit: `human_interrupt_tool`
is called **only** when `check_policy` returns a result starting with
`NO_ANSWER_FOUND`. A bare "let me talk to a human" is not itself sufficient.

```
POST /chat/user_001 {"message": "My payment was charged twice, order X"}
        |
        |- guardrails check: passes (legitimate complaint)
        |
        |- graph_app.stream(...) -- chatbot calls check_policy("duplicate charge")
        |
        |   policy_engine.answer() -> vector_store.search() finds the
        |   relevant paragraph in refund_policy.md:
        |     "If a customer reports a duplicate or incorrect charge,
        |      this requires manual verification ... escalated to a
        |      human support agent"
        |   -> RAG generation cites this, model has real grounds to escalate
        |
        |- chatbot calls human_interrupt_tool(query="duplicate charge...")
        |
        |   app/graph/tools.py: human_interrupt_tool()
        |     \- interrupt({"query": ..., "message": ...})
        |          LangGraph PAUSES execution here. State is checkpointed
        |          to MongoDB (thread-scoped, via langgraph-checkpoint-mongodb).
        |          This can only be resumed ONCE.
        |
        v
graph_app.get_state() now shows an open interrupt.
escalation_store.add_message(user_id, "user", original_message)
        |  (app/services/escalation_store.py -- the side-channel, separate
        |   from the graph's own message history)
        v
{"status": "escalated", "message": "Your request has been escalated..."}
```

→ See [`DECISIONS.md`: The escalation side-channel exists because `interrupt()` can only resume once](./DECISIONS.md#the-escalation-side-channel-exists-because-interrupt-can-only-resume-once)

---

## 4. While escalated -- the side-channel

Once a thread has an open interrupt, `chat.py`'s very first state check
catches it and reroutes *before* touching the graph or the LLM at all:

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

The support agent side, polling or viewing the thread:

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

The agent can reply without resolving (gathering more info):

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

## 5. Resolving -- money confirmation, verbatim delivery, LLM path reopens

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
        |   NOTHING has happened yet -- no graph resume, no message logged,
        |   no push. The frontend shows a confirm() dialog; if accepted:
        v
POST /support/resolve/user_001 {"resolution": "Refund issued for $50.", "confirmed": true}
        |
        |- escalation_store.add_message(thread_id, "support", resolution)
        |
        |- graph_app.stream(Command(resume={"data": resolution}))
        |      |
        |      |  human_interrupt_tool()'s interrupt() call returns,
        |      |  with resolution as its return value -- the tool "result"
        |      |
        |      v
        |  app/graph/graph.py: route_after_tools()
        |      |  the tool that just ran WAS human_interrupt_tool ->
        |      |  route straight to END, skip chatbot entirely.
        |      |  The resolution reaches the user VERBATIM -- no second
        |      |  LLM call re-paraphrasing a refund amount.
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

After this, `graph_app.get_state(thread_id).interrupts` is empty again --
the next `POST /chat/user_001` goes back to the full LLM path in section 2.

---

## 6. Real-time push -- the websocket lifecycle

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
