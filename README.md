# 🧠 Support Agent Infra — Durable Human-in-the-Loop AI Support

A **stateful, multi-tenant AI support agent** built on LangGraph, with MongoDB-backed
persistence and a real escalation-to-human workflow. The agent can look up a
user's cart, order history, and order status, search company policy docs, and
**pause execution mid-conversation to hand off to a human** — then resume
exactly where it left off, with full state intact.

This is not a RAG project. It's about the harder infra problem underneath any
AI support tool: **what happens when the model can't or shouldn't answer, and
a human needs to safely take over without losing context.**

No production frontend — the API is the real interface (see `/docs`). A
minimal HTML/JS demo is included in `frontend/` purely so the
human-in-the-loop flow can be watched happening, with real-time push and
without typing curl commands — see [Demo](#-demo).

**Two companion docs go deeper than this README does:**
[`REQUEST_FLOW.md`](./REQUEST_FLOW.md) walks through exactly what happens,
step by step with real function names, for a normal chat, an escalation,
and a resolution. [`DECISIONS.md`](./DECISIONS.md) is every non-obvious
choice in one place — what was decided, what else was considered, and what
it costs — in the same format throughout.

---

## 🏗️ Architecture

```
POST /auth/token   [issues a JWT — user or support_agent role]
        │
        ▼
POST /chat/{user_id}   [Bearer token required, rate limited: 15/min per user]
        │
        ▼
   Guardrails input check (NeMo Guardrails) ──► blocked? return refusal,
        │                                        never reaches the agent
        ▼ (allowed)
   LangGraph agent ── tool call ──► get_user_cart / get_order_history / get_latest_order /
        │                            get_order_by_id / check_policy
        │                            (FAQ match → RAG w/ citations →
        │                             NO_ANSWER_FOUND signal)
        │
        ▼ (only on NO_ANSWER_FOUND, never just because the user asked)
   interrupt() ──► MongoDB checkpoint (paused, thread-scoped)
        │
        ▼
   escalation_store ──► live side-channel: user talks ONLY to the human now
        │                (GET/POST /support/thread/{id} — support_agent
        │                 token required — live websocket push)
        ▼
   POST /support/resolve/{thread_id}   [support_agent token, rate limited: 20/min]
        │    ├─ money-related resolution (refund/credit/amount) AND
        │    │  not yet confirmed? → confirmation_required, nothing touched
        │    ▼
        │  Command(resume=...) ──► human's message delivered verbatim,
        │                           LLM path reopens for future turns
        ▼
   Final reply pushed to the original user
```

See [`REQUEST_FLOW.md`](./REQUEST_FLOW.md) for the same thing at full detail.

---

## 🚀 What changed from the original prototype

The first version was two CLI scripts sharing a hardcoded `thread_id="8"`,
with a `while True: continue` busy-wait loop polling Mongo for interrupt
status. That doesn't hold up as "infra." This version:

* Replaces the polling CLI with a **FastAPI service** — no blocking loop, no hardcoded thread
* Supports **many concurrent users**, each with an isolated thread (`user_id` = `thread_id`)
* Adds a **support queue** (`/support/pending`) instead of a single-thread lookup
* Adds **real tools**: mock cart/order lookup + policy doc search, not just the escalation tool
* Adds **metrics** (`/metrics`): escalation rate, avg agent latency, avg human resolution time
* Adds a **concurrency test** that fires multiple simultaneous escalations and verifies no cross-thread state corruption
* Adds a **live escalation side-channel**: once escalated, the user talks ONLY to the human — a real back-and-forth (`GET`/`POST /support/thread/{id}`), not a single request/response
* Adds a **real intent router** for `check_policy`: FAQ keyword match → RAG over a local Qdrant (embedded mode) store with citations → an explicit `NO_ANSWER_FOUND` signal
* Makes **escalation a genuine last resort**: the system prompt gates `create_support_ticket` on `check_policy` returning `NO_ANSWER_FOUND`, not on the user simply asking for a human
* Adds a **typing indicator** and a **demo frontend** (`frontend/`) with a real auto-connecting websocket, no manual steps
* Adds **rate limiting** (`slowapi`, per-user) on every state-changing endpoint that costs money or write access
* Adds **input guardrails** (NeMo Guardrails) — prompt injection and off-topic messages are blocked before they ever reach the agent
* Adds a **money-confirmation gate** on resolving a thread — a refund/credit/currency-amount resolution requires an explicit second confirmation, enforced server-side, before it actually reaches the user
* Adds **JWT auth** — every protected endpoint requires a token, `user` tokens only act as their own `user_id`, `support_agent` tokens are required for every `/support/*` endpoint, verified with real 401/403 tests, not just added and assumed
* Adds a **live user context sidebar** and **agent tool-execution badges** to the demo frontend, and a formal **ticket system** (`create_support_ticket`, `/support/tickets`) with a replay-safe ticket id — see [`CHANGES.md`](CHANGES.md) for the story of the replay bug this caught
* Adds **structured, tagged logging** (`app/core/agent_logging.py`) — every app-level log line carries a `[AGENT]` / `[TOOL CALL]` / `[GUARDRAIL STATUS]` / `[RESPONSE]` tag with a timestamp and thread id, and third-party framework logging (NeMo Guardrails, LangChain, LangGraph, pymongo, httpx...) is turned down to WARNING so it doesn't drown out what the agent is actually doing. Distinct from `log_event()`, which writes structured events to Mongo for the `/metrics` endpoint — this is for a human reading stdout/`docker logs`, not for querying later.

---

## 🖥️ Demo

**Option A — `/ui/` demo frontend (recommended — real push, zero manual steps):**

Just start the server (`uvicorn app.main:app --reload`) and open
`http://localhost:8000/ui/`. Plain HTML/JS, no separate process to run,
served same-origin by FastAPI itself. Send a message that escalates ("let
me talk to a human...") and the page opens `/ws/{user_id}` automatically
the instant it sees `{"status": "escalated"}` — no polling, no separate
tab, no manual websocket connection. Switch to the **Support Queue** tab,
resolve it, and watch the resolution land in the chat tab live.

Auth is transparent here — the frontend mints its own tokens client-side
(`POST /auth/token`, no real login form; see
[`DECISIONS.md`](./DECISIONS.md#auth-is-a-stub-but-enforcement-is-real))
and attaches them to every request. Nothing to do manually.

This is a demo prototype, not the project. It exists to make the HITL
loop watchable in under a minute; the actual interface is still the API.

**Option B — Swagger UI:** open `http://localhost:8000/docs`. `POST
/auth/token` with `{"user_id": "user_001", "role": "user"}` (or
`"role": "support_agent"` for the `/support/*` endpoints), copy the
`access_token`, click the padlock **Authorize** button at the top of the
page, and paste it in — Swagger then attaches it to every request you make
from there on. Every REST endpoint is clickable and testable this way. It
can't exercise the websocket route (`/ws/{user_id}`) — Swagger doesn't
support testing websockets — but `GET /ws-test/{user_id}` is a plain
HTML/JS debug page (no install needed, mints its own token) that opens
that connection directly if you want to verify the push mechanism in
isolation, separate from the demo client.

**Option C — Terminal walkthrough** (good for a README GIF):

```bash
# Get a user token
USER_TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "role": "user"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Get a support-agent token
AGENT_TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "agent_demo", "role": "support_agent"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Upload a policy doc (optional — five are seeded already)
curl -X POST -F "file=@app/data/docs/refund_policy.md" \
  -H "Authorization: Bearer $AGENT_TOKEN" \
  http://localhost:8000/docs/upload

# Chat as a user
curl -X POST http://localhost:8000/chat/user_001 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d '{"message": "What'\''s in my cart?"}'

# Trigger an escalation
curl -X POST http://localhost:8000/chat/user_001 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -d '{"message": "I was double charged and need this manually reversed."}'

# See it queued for support
curl http://localhost:8000/support/pending -H "Authorization: Bearer $AGENT_TOKEN"

# Resolve as the support agent — confirmed:true since this resolution is
# money-related (see DECISIONS.md); without it, this returns
# confirmation_required and doesn't actually resolve anything
curl -X POST http://localhost:8000/support/resolve/user_001 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AGENT_TOKEN" \
  -d '{"resolution": "Refund issued, duplicate charge reversed.", "confirmed": true}'

# Check metrics
curl http://localhost:8000/metrics
```

---

## 📁 Project Structure

```
.
├── app/
│   ├── main.py                  # App assembly + lifespan only — no endpoint logic
│   ├── core/
│   │   ├── deps.py                # Shared state (graph instance, mongo client) for routers
│   │   ├── rate_limit.py            # slowapi Limiter, keyed per user_id/thread_id
│   │   └── auth.py                   # JWT issuance/verification, role enforcement
│   ├── services/
│   │   ├── mock_db.py             # Seeded fake commerce DB (SQLite) — no storefront needed
│   │   ├── escalation_store.py     # Live user<->human side-channel during an open escalation
│   │   ├── faq.py                   # Fast keyword-matched FAQ path — no embedding/LLM call
│   │   ├── vector_store.py           # Qdrant (embedded mode, local folder) — RAG persistence
│   │   ├── policy_engine.py           # Router: FAQ -> RAG w/ citations -> NO_ANSWER_FOUND
│   │   ├── ws_manager.py               # In-memory websocket connection map for live push
│   │   ├── guardrails.py                # NeMo Guardrails input check (injection + topic)
│   │   └── money_detection.py            # Flags refund/credit/currency resolutions
│   ├── data/
│   │   ├── docs/                    # Company policy docs indexed into Qdrant (committed)
│   │   │   ├── refund_policy.md        # incl. the duplicate-charge escalation trigger
│   │   │   ├── shipping_policy.md       # incl. stuck-in-transit / lost-package triggers
│   │   │   ├── warranty_policy.md        # incl. damaged-on-arrival / out-of-window triggers
│   │   │   ├── returns_and_exchanges.md   # incl. restocking-fee-dispute trigger
│   │   │   └── account_and_billing.md      # incl. fraud / account-closure triggers
│   │   └── guardrails_config/
│   │       └── config.yml             # Consolidated injection + topic-control rail (see below)
│   │         (commerce.db and qdrant/ also live here — both gitignored,
│   │          rebuilt fresh from seed data on every startup)
│   ├── graph/
│   │   ├── graph.py            # Pure wiring — includes route_after_tools (skip LLM
│   │   │                          after a human resolves, deliver their answer verbatim)
│   │   ├── nodes.py             # LLM setup + chatbot node + escalation-as-last-resort prompt
│   │   ├── tools.py              # All tool definitions (cart, orders, check_policy, escalation)
│   │   └── state.py              # Graph State schema
│   └── routers/
│       ├── chat.py               # POST /chat/{user_id} (routes to LLM or escalation
│       │                            side-channel depending on state), GET .../status,
│       │                            WS /ws/{user_id}, GET /ws-test/{user_id}
│       ├── support.py             # GET /support/pending, GET/POST /support/thread/{id}
│       │                            (live reply without resolving), POST .../resolve/{id}
│       ├── metrics.py              # GET /metrics
│       ├── docs.py                  # POST /docs/upload (also indexes into Qdrant)
│       ├── health.py                 # GET /health
│       └── auth.py                    # POST /auth/token (stub login, real enforcement downstream)
├── frontend/                   # Demo frontend — plain HTML/JS, served by FastAPI itself
│   └── index.html                # Real websocket, typing indicator, live escalation chat,
│                                    support thread detail view (reply + resolve)
├── tests/
│   ├── test_mock_db.py            # Unit tests, no server/creds needed — run in CI
│   ├── test_ws_manager.py
│   ├── test_graph_routing.py       # route_after_tools (skip-LLM-after-resolve) logic
│   ├── test_faq.py
│   ├── test_vector_store.py         # Real Qdrant read/write/search, fake embeddings
│   ├── test_policy_engine.py         # FAQ -> RAG -> NO_ANSWER_FOUND routing decisions
│   ├── test_escalation_store.py       # Side-channel persistence (mongomock)
│   ├── test_escalation_flow.py         # Full router integration: escalate -> side-channel
│   │                                      -> reply -> resolve -> LLM path reopens
│   ├── test_rate_limit.py               # Confirms slowapi actually 429s against the real app
│   ├── test_money_detection.py           # Refund/credit/currency detection for the confirm gate
│   ├── test_guardrails.py                 # check_message logic (fake rails, no live LLM call)
│   ├── test_auth.py                        # Token lifecycle, role enforcement, 401/403 edge cases
│   └── test_concurrent_escalation.py    # Integration test — needs live server, run manually
│   (also: manual_eval_escalation.py — scripted check that escalation is
│    actually a last resort against a live LLM, not part of CI)
├── .github/workflows/ci.yml    # Compiles + runs unit tests on every push/PR
├── REQUEST_FLOW.md             # Step-by-step walkthrough of every real flow
├── DECISIONS.md                # Every non-obvious choice: decision / alternatives / tradeoff
├── Dockerfile
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real MONGODB_URI and OPENAI_API_KEY
```

No separate credentials needed for guardrails — `app/data/guardrails_config/config.yml`
reuses the same `OPENAI_API_KEY` and points at `gpt-4.1`, same as the main agent.

Run:

```bash
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs`, or `http://localhost:8000/ui/` for
the demo frontend. Qdrant indexes `app/data/docs/*.md` automatically on
startup — no separate indexing step. `user_001`, `user_002`, and
`user_003` have deterministic, policy-relevant scenarios seeded (a
duplicate charge, an order outside the refund window, a stuck shipment —
see `mock_db.py`) so you can test real escalation behavior without
hand-crafting order data first.

Run the concurrency test (server must be running):

```bash
python tests/test_concurrent_escalation.py
```

Run the escalation-behavior eval (server must be running, needs real
credentials — this is the one thing that can't be verified without a
live LLM call):

```bash
python tests/manual_eval_escalation.py
```

Checks 8 scenarios against the running server — duplicate charges and
fraud reports should escalate, FAQ questions and out-of-window refund
requests shouldn't — and reports pass/fail with a non-zero exit code on
any hard failure.

**Or with Docker:**

```bash
docker build -t support-agent-infra .
docker run --env-file .env -p 8000:8000 support-agent-infra
```

---

## 🔑 Key Design Decisions

The full reasoning for every non-obvious choice — what was decided, what
else was considered, and what it costs — lives in
[`DECISIONS.md`](./DECISIONS.md). Short version:

- **Mock commerce DB, not a real storefront** — the point is agent
  orchestration, not e-commerce.
- **`check_policy` is real semantic search, not a full RAG stack** — FAQ
  keyword match → Qdrant (embedded, real embeddings) → cited answer or a
  `NO_ANSWER_FOUND` signal. Proportionate to what this project needs, not
  duplicating a separate RAG project's retrieval stack.
- **Escalation is gated on that `NO_ANSWER_FOUND` signal, not on request**
  — a bare "let me talk to a human" isn't itself sufficient; the system
  prompt requires a real attempt first.
- **Resolving a thread skips the LLM entirely** (`route_after_tools`) —
  the human's resolution reaches the user verbatim, not re-paraphrased.
- **The escalation side-channel exists because `interrupt()` can only
  resume once** — LangGraph's real constraint, not a design preference.
- **`thread_id` is just `user_id`** — simpler, sufficient to prove
  multi-tenant isolation, at the cost of one active conversation per user.
- **Three users have scripted, deterministic data** tied to real policy
  triggers, so testing escalation behavior is reproducible.
- **Rate limiting is per-`user_id`, not per IP** — multi-tenant, so IP
  isn't the right unit; the GET polling routes are deliberately exempt.
- **`topic_safety` was tried and abandoned for `self check input`** — real
  friction (a second named model NeMo's module needs), not a stylistic
  choice.
- **Guardrails fail open, not closed** — availability over paranoid safety
  for a support bot; every failure is still logged.
- **Auth is a stub, but enforcement is real** — no password check, but a
  `user` token genuinely can't act as another user, and every 401/403 path
  is tested.
- **Cart/order tools take `user_id` from injected session config, not an
  LLM argument** — closes a real cross-user data leak the earlier version
  had, found by actually using the app, not caught by any test beforehand.
- **Money-related resolutions require server-enforced confirmation** — a
  JS dialog alone can be skipped by hitting the API directly; the backend
  checks too.
- **The demo frontend is plain HTML/JS, not Streamlit** (which was tried
  first) — a real persistent websocket needs a client that can hold one
  open, which Streamlit's rerun-per-interaction model can't do.

---

## ⚠️ Known limitations (intentional scope cuts)

* `check_policy`'s RAG path uses naive paragraph-level chunking and a flat
  Qdrant collection — no reranking, no hybrid retrieval. See above for why.
* `/support/pending` scans all known users' state per request; fine for a
  demo, would need a dedicated "open interrupts" index at real scale.
* `escalation_store` messages aren't fed back into the graph's own message
  history — the human's final resolution is, but the back-and-forth before
  it isn't part of what the LLM sees on future turns. A deliberate cut;
  see the design-decisions entry above.
* Auth is a real stub, not real security — `POST /auth/token` issues a
  token for any `user_id`/`role` with no credential check, since there's
  no real user database to check against. Enforcement downstream of that
  is genuine (see `DECISIONS.md`), but don't mistake this for production auth.
* Single support queue, no assignment/priority logic.
* Endpoints are sync (`def`, not `async def`) — deliberate, since the
  Mongo checkpointer and LangGraph `.stream()` calls are themselves
  synchronous; FastAPI runs sync routes in a threadpool, so this is
  correct as-is rather than an oversight. (`/support/thread/{id}/reply`
  is `async def` since it only awaits the websocket push — no synchronous
  graph call in that path.)
* Rate limiting is in-memory (slowapi's default backend) — fine for a
  single process, but buckets don't share state across multiple instances.
  Would need a Redis-backed `limits` storage string at real scale.
* `money_detection.py` is a keyword list + a currency regex, not an
  exhaustive financial-intent classifier — it's meant to catch the obvious
  cases reliably, not every possible phrasing. A false positive costs an
  agent one extra click; a false negative means a refund resolution ships
  without the confirmation step, which is the real risk worth knowing about.
* Guardrails checks add one extra LLM call per message (before the agent's
  own call) — more latency and cost per turn than before, in exchange for
  actually blocking injection/off-topic messages instead of relying on the
  system prompt alone.

These are called out explicitly rather than hidden, because knowing what to
cut and why is itself part of the engineering signal.

---

## 🗺️ Roadmap

In priority order:

1. **Demo recording** — a short terminal/browser walkthrough for the README,
   the highest-leverage thing left for anyone not reading the code.
2. **Better `/metrics`** — currently escalation rate and avg latency only.
   The more interesting signal for this specific project is per-tool usage:
   how often `check_policy` resolves via FAQ vs RAG vs actually escalates.
3. **Open-interrupts index** — replace `/support/pending`'s full user scan
   with a dedicated index once this needs to run at any real scale.
4. **Redis-backed rate limiting** — only matters once this runs as more
   than one process; the in-memory buckets don't share state across instances.
5. **Real credential-backed auth** — swap `/auth/token`'s internals for
   actual password/OAuth verification once there's a real user database
   to check against. Everything downstream (`require_matching_user`,
   `require_support_agent`) stays the same.

Already done, not just planned: richer seed data with three deterministic,
policy-relevant scenarios (`mock_db.py`), a five-document policy corpus
spanning shipping/warranty/returns/billing instead of one paragraph,
`tests/manual_eval_escalation.py` (a scripted check of whether escalation
is genuinely a last resort against a live model), per-user rate limiting
on every state-changing endpoint, NeMo Guardrails input checks for prompt
injection and off-topic messages, a server-enforced confirmation gate on
money-related resolutions, and JWT auth with real role-based enforcement
(`user` vs `support_agent`).


