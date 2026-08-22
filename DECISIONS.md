# Decisions and Tradeoffs

Every non-obvious choice in this project, in one place, in the same format:
what was decided, what else was considered, and what it costs. `README.md`
has the short version of some of these. See
[`REQUEST_FLOW.md`](./REQUEST_FLOW.md) for where each of these fires during
a request.

---

## Escalation is gated on a signal, not on request

**Decision:** `create_support_ticket` is called only when `check_policy`
returns a result starting with `NO_ANSWER_FOUND`. The system prompt
(`app/graph/nodes.py`) explicitly instructs the model not to escalate just
because the user asks to talk to a human — it has to make a real attempt
first.

**Alternatives considered:** The model escalates whenever it decides to, or
whenever the user asks. Simpler to implement, and still looks like a
working escalation flow in a demo.

**Tradeoff:** The system prompt has to do real work, so behavior depends on
the model actually following instructions. `tests/test_policy_engine.py`
proves the signal is produced correctly; `tests/manual_eval_escalation.py`
is the live-model check, meant to run with real credentials. A model that
ignores the instruction would silently degrade this to "escalate whenever
asked," with nothing in the code to catch it — that's what the eval script
is for.

---

## `route_after_tools` skips the LLM after a human resolves

**Decision:** When `create_support_ticket` is the tool that just ran,
`app/graph/graph.py`'s `route_after_tools` routes straight to `END` instead
of back through `chatbot`. The support agent's resolution text reaches the
user verbatim.

**Alternatives considered:** LangGraph's default `tools → chatbot` wiring
routes every tool result — including a human's answer — back through the
LLM to turn it into a reply. That's the standard ReAct pattern, and it's
what this project did before this change.

**Tradeoff:** Saves one LLM call and removes the risk of the model
rewording a refund amount or policy commitment a person deliberately
phrased a certain way. Every other tool result (cart lookup, order status,
`check_policy`) still needs that second LLM call to become a coherent
sentence — this is a special case, not a general optimization.

---

## The escalation side-channel exists because `interrupt()` can only resume once

**Decision:** `app/services/escalation_store.py` is a separate, append-only
message log, entirely outside the LangGraph checkpointed conversation. Once
a thread is escalated, every user message goes there — not through the
graph — until the agent resolves.

**Alternatives considered:** Routing every escalated message back through
`Command(resume=...)` for an ongoing back-and-forth. Doesn't work:
LangGraph's `interrupt()` pauses once and expects exactly one resume — you
can't repeatedly resume the same paused call to simulate a live
conversation.

**Tradeoff:** The side-channel's messages aren't part of the model's own
message history — only the final resolution text is, since that's what
actually gets passed to `Command(resume=...)`. If a customer explains
important context to the human agent that never makes it into the
resolution text verbatim, the model won't see it on future turns. A more
complete version would fold the side-channel transcript into graph state on
resolve; not done here.

---

## `check_policy` is real semantic search, not a full RAG stack

**Decision:** `app/services/policy_engine.py` does FAQ keyword match →
Qdrant (embedded mode) vector search with real OpenAI embeddings → an LLM
call that must answer only from retrieved context, citing sources, or
return `NO_ANSWER_FOUND`. What it doesn't have: chunking beyond naive
paragraph splitting, reranking, hybrid dense+sparse retrieval, or a query
planner.

**Note on "embedded mode":** `qdrant-client` supports talking to a real
Qdrant server over the network (needs a URL/API key) or running entirely
in-process via `QdrantClient(path=...)`, writing its index to a local
folder (`app/data/qdrant/`) with no server and no network call. This
project uses the latter — the only credential the RAG path needs is
`OPENAI_API_KEY`, for the embedding calls.

**Alternatives considered:** Building the same hybrid retrieval stack
already built for a separate RAG-focused project (hybrid dense+sparse
fusion, semantic caching, a planner). Duplicating that here would
demonstrate the same skill twice instead of two different ones.

**Tradeoff:** Retrieval quality here is good enough to prove the escalation
logic works, not necessarily good enough for a large or messy document
corpus.

---

## `topic_safety` was tried and abandoned for `self check input`

**Decision:** NeMo Guardrails' input rail
(`app/data/guardrails_config/config.yml`) consolidates prompt-injection
detection and topic control into one `self check input` rail, rather than
using NeMo's dedicated `topic_safety` module.

**Alternatives considered:** `topic_safety` is the module built for this —
it exists specifically for topic classification. Tried first.

**What went wrong:** `topic_safety` requires a separately registered named
model under `models:` in the config (built for a dedicated topic-classifier
deployment, e.g. Llama Guard) — it doesn't resolve against the
already-configured main model the way `self_check_input` does. Confirmed
by building the config and hitting `InvalidRailsConfigurationError`.

**Tradeoff:** One consolidated LLM call instead of two, no second model to
stand up — but injection detection and topic relevance are now coupled in
one instruction block, less precise than two independently-tunable rails.

---

## Guardrails checks fail open, not closed

**Decision:** If the guardrails check itself errors (API timeout, network
issue), `app/services/guardrails.py` lets the message through to the normal
agent rather than blocking it.

**Alternatives considered:** Fail closed — block every message if the
safety check can't run.

**Tradeoff:** Prioritizes availability over paranoid safety. A
safety-critical system might reasonably choose the opposite. Every failure
is logged (`logger.exception`), so an outage is visible in logs — but a
user experiencing it just sees the bot working normally, with no sign
guardrails were skipped for that message.

---

## Rate limiting is per-`user_id`, and deliberately not applied everywhere

**Decision:** `slowapi`, keyed by `user_id`/`thread_id` extracted from the
URL path (`app/core/rate_limit.py`), not by IP. Applied to every
state-changing or cost-bearing endpoint: `POST /chat/{user_id}` (15/min),
`POST /docs/upload` (5/min), `POST /support/resolve/{thread_id}` (20/min),
`POST /support/thread/{id}/reply` (30/min). Not applied to GET polling
endpoints (`/support/pending`, `/support/thread/{id}`).

**Alternatives considered:** Per-IP limiting (slowapi's default) — this is
multi-tenant, and many legitimate users can share an IP (NAT, shared
network). What's worth protecting against is one user hammering the LLM,
not too many requests from one network. Rate limiting the GET polling
routes was also considered and rejected: the frontend polls
`/support/thread/{id}` every 2.5 seconds, and a limit there would break the
demo's own polling loop.

**Tradeoff:** In-memory bucket storage (slowapi's default) — fine for a
single process, buckets don't share state across multiple instances. A real
multi-instance deployment needs a Redis-backed `limits` storage string.

---

## Auth is a stub, but enforcement is real

**Decision:** `POST /auth/token` (`app/routers/auth.py`) issues a JWT for
any `user_id`/`role` requested — no password or credential check, because
there's no real user database in this project. Everything downstream of
that is real: `app/core/auth.py`'s `require_matching_user` and
`require_support_agent` are enforced on every protected route, verified by
tests asserting 401/403 on missing, forged, expired, and mismatched tokens
(`tests/test_auth.py`).

**Alternatives considered:** Skipping auth entirely — simpler, but this
project has no real user accounts to authenticate against, so building
real credential-based login (password hashing, a user table) would just be
theater around real enforcement, not more real enforcement.

**Tradeoff:** Anyone can mint a token claiming to be any `user_id` or
`role` — this is not production security. What's demonstrated is the
pattern: token issuance, verification, role-based access control, and
websocket auth via query param (browsers can't set custom headers on a WS
handshake). A real system swaps `/auth/token`'s internals for actual
credential verification; nothing downstream changes.

---

## Tool-level authorization uses injected config, not LLM-supplied arguments

**Decision:** `get_user_cart` and `get_order_history` (`app/graph/tools.py`)
take zero LLM-visible arguments — they read the current `user_id` from an
injected `RunnableConfig`, bound from the authenticated session in
`chat.py`, not from anything the model outputs. `get_order_by_id` still
takes an LLM-supplied `order_id` (the user has to say which order), but
checks the order's owning `user_id` against the injected one before
returning any data.

**What this fixes:** the earlier version had these tools take `user_id` as
a normal LLM-controllable argument. Two problems followed: the model had no
way to know the current user's id unless the user stated it in the
conversation, and nothing stopped the model from calling
`get_order_by_id(order_id=...)` for an order belonging to a different user
and returning that data — a cross-user data leak via tool-calling.

**Alternatives considered:** Fixing only the usability bug by injecting
`user_id` into the system prompt as text. Would resolve the symptom, not
the underlying issue — the model could still be convinced (accidentally or
via prompt injection) to call a tool with a different `user_id`, since the
argument would still be LLM-controllable.

**Tradeoff:** None on the cart/history tools — removing an argument the
model shouldn't have controlled. `get_order_by_id`'s "not found" response
is deliberately identical whether an order doesn't exist or isn't the
current user's, so the response itself doesn't leak that a given order_id
is real.

---

## In-memory websocket connections do not need a message queue yet

**Decision:** `app/services/ws_manager.py` is a plain Python dict mapping
`thread_id → websocket`, held in process memory.

**Alternatives considered:** A message queue (Redis pub/sub) to broadcast
push notifications across processes.

**Tradeoff:** This works because the process that resolves an escalation is
the same process holding that user's open connection — true for a
single-process deployment, which is what this project runs as. A message
queue only becomes necessary with multiple server instances behind a load
balancer, where the instance handling `POST /support/resolve/{id}` might
not be the instance holding the user's websocket. Same reasoning as not
indexing `/support/pending` yet.

---

## A mock commerce DB instead of a real storefront

**Decision:** `app/services/mock_db.py` is a seeded SQLite database with
users, orders, and cart items — no storefront, no checkout flow, no payment
integration.

**Alternatives considered:** Building an actual e-commerce frontend for the
agent to sit in front of.

**Tradeoff:** The point of this project is agent orchestration and HITL
state management, not e-commerce — a seeded DB gives the agent realistic
structured data to query without weeks of frontend work. The cost: nothing
here demonstrates real payment/checkout integration.

---

## Three users have deterministic, scripted data instead of everything being random

**Decision:** `user_001`, `user_002`, and `user_003` in `mock_db.py` have
fixed scenarios instead of random data — a duplicate charge, an order
outside the refund window, a shipment stuck processing for 18 days. Each
maps directly onto a real escalation trigger in the policy docs.

**Alternatives considered:** Fully random seed data for every user, same as
the rest of the seeded set.

**Tradeoff:** Testing whether escalation happens for the right reason
becomes reproducible instead of requiring order data to be hand-crafted
through the API first — `tests/manual_eval_escalation.py` depends on this
directly. Cost: these three users behave differently from the rest of the
seeded set.

---

## `thread_id` is just `user_id`

**Decision:** No separate session/thread concept — the LangGraph checkpoint
thread, the escalation side-channel key, and the websocket connection key
are all literally the `user_id`.

**Alternatives considered:** A separate `thread_id` per conversation,
allowing one user multiple concurrent support threads.

**Tradeoff:** Simpler by a wide margin, sufficient to prove multi-tenant
thread isolation — the harder problem interrupts introduce. Cost: a user
can only have one active conversation (and one active escalation) at a
time, which a real support system would probably not want as a hard
constraint.

---

## Money-related resolutions require server-enforced confirmation

**Decision:** `app/services/money_detection.py` (a keyword list plus a
currency-amount regex) flags resolutions that look like a refund, credit,
chargeback, or currency amount. `POST /support/resolve/{thread_id}` returns
`confirmation_required` — touching nothing — unless the request carries
`confirmed: true`.

**Alternatives considered:** A frontend-only confirmation dialog (`confirm()`
before submitting). Simpler, but trivially bypassed by hitting the API
directly instead of the UI.

**Tradeoff:** `money_detection.py` is a heuristic, not an exhaustive
financial-intent classifier — a false positive costs an agent one extra
click; a false negative means a refund resolution ships without the
confirmation step.

---

## Two demo frontends existed at different points; only one remains

**Decision:** `frontend/index.html` — plain HTML/JS, served same-origin by
FastAPI, holds a real persistent websocket, connects automatically the
instant a message escalates.

**Alternatives considered and actually built, then removed:** An earlier
version used Streamlit. It worked, but Streamlit reruns its entire script
on every interaction, so it can't hold a persistent connection open —
it had to poll instead. Once the plain-HTML version proved the real push
mechanism worked (verified with Playwright against a real browser, not
just unit tests), the Streamlit version was deleted rather than kept as a
weaker second example.

**Tradeoff:** None significant — a straightforward improvement, recorded
here because "tried something, shipped it, replaced it once something
better existed" is itself a decision worth being able to explain.

---

## Ticket ids are derived from the tool call id, not generated fresh

**Decision:** `create_support_ticket` (`app/graph/tools.py`) sets
`ticket_id = f"tkt_{tool_call_id}"`, where `tool_call_id` is injected via
`InjectedToolCallId`. `ticket_store.create_ticket` is an idempotent upsert
keyed on that id.

**What this fixes:** `interrupt()` causes LangGraph to re-run a node's
entire function body from the top on every resume. The first
implementation generated `ticket_id` via `uuid4()` before calling
`interrupt()` — every human reply to an escalated thread replayed that
line and minted a new, different ticket id, leaving the original ticket
open forever while a duplicate accumulated per reply.

**Alternatives considered:** Moving the ticket-creation call to after
`interrupt()` returns. Doesn't work — the whole point of a ticket is to
give a human something to look at while the thread is paused, before they
reply.

**Tradeoff:** None. `tool_call_id` is part of the already-checkpointed
`AIMessage` and is stable across replay, so deriving from it is strictly
more correct than the random-id version, at no extra cost. Covered by an
integration test that drives a real LangGraph interrupt/resume cycle
(`tests/test_ticket_replay_safety.py`), not a mocked graph — a mock
wouldn't reproduce LangGraph's actual replay behavior.

---

## The idempotency cache uses a per-key lock, not a plain check-then-set

**Decision:** `app/core/idempotency.py`'s `IdempotencyCache.get_or_compute`
acquires a lock scoped to the request's idempotency key before computing,
re-checking the cache after acquiring the lock.

**Alternatives considered:** Check the cache, compute on a miss, then write
the result — no locking. Simpler, and sufficient for a client that waits
for a response before retrying.

**Tradeoff:** The naive version isn't safe against a client that retries
before the first request finishes: two threads can both observe a miss and
both execute the request. The per-key lock (not a single global lock, which
would serialize unrelated requests too) closes that gap for
`POST /chat/{user_id}`. `POST /chat/{user_id}/stream` doesn't get the same
guarantee — coalescing two live SSE streams would mean the second caller
blocking until the first finishes, then replaying it, which wasn't built.
Documented as a narrower guarantee in that endpoint's own docstring.

---

## Checkpoint expiry uses MongoDB's native TTL index, not a custom worker

**Decision:** `MongoDBSaver.from_conn_string(MONGODB_URI, ttl=...)`
(`app/main.py`) — the checkpoint library's own built-in TTL parameter,
creating a real `expireAfterSeconds` index on `created_at` for both the
checkpoints and checkpoint_writes collections.

**Alternatives considered:** A cron job or background worker that scans for
and deletes old checkpoints.

**Tradeoff:** None — this is strictly less code and less to run than a
custom cleanup process, verified by reading `langgraph-checkpoint-mongodb`'s
own source before relying on it rather than assuming from the parameter
name. The one real interaction worth knowing: a ticket tied to a thread
that expires while still escalated does not expire with it —
`ticket_store`'s collection has no TTL of its own, so it stays `open`
until a human resolves it. See README's Known Limitations.
