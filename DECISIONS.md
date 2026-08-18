# Decisions and Tradeoffs

Every non-obvious choice in this project, in one place, in the same
format: what was decided, what else was considered, and what it costs.
`README.md` has the short version of some of these; this is the full
version. See [`REQUEST_FLOW.md`](./REQUEST_FLOW.md) for where each of
these actually fires during a request.

---

## Escalation is gated on a signal, not on request

**Decision:** `create_support_ticket` is called only when `check_policy`
returns a result starting with `NO_ANSWER_FOUND`. The system prompt
(`app/graph/nodes.py`) explicitly instructs the model not to escalate
just because the user asks to talk to a human — it has to make a real
attempt first.

**Alternatives considered:** The obvious naive version — the model
escalates whenever it decides to, or whenever the user asks — is what
most toy human-in-the-loop demos do. It's simpler to implement and would
still *look* like a working escalation flow in a demo.

**Tradeoff:** This makes the system prompt do real work, which means its
behavior is genuinely dependent on the model actually following
instructions — verified as far as possible without live credentials
(`tests/test_policy_engine.py` proves the signal is produced correctly;
`tests/manual_eval_escalation.py` is the live-model check, meant to be
run with real credentials, not something I could fully verify myself).
A model that ignores the instruction would silently degrade this back to
"escalate whenever asked," with nothing in the code that would catch it —
that's what the eval script is for.

---

## `route_after_tools` skips the LLM after a human resolves

**Decision:** When `create_support_ticket` is the tool that just ran,
`app/graph/graph.py`'s `route_after_tools` routes straight to `END`
instead of back through `chatbot`. The support agent's resolution text
reaches the user **verbatim**.

**Alternatives considered:** LangGraph's default `tools → chatbot` wiring
would route every tool result — including a human's answer — back through
the LLM to "turn it into a reply." That's the standard ReAct pattern, and
it's what the project did before this was added.

**Tradeoff:** Saves one LLM call (latency, cost) and removes the risk of
the model subtly rewording a refund amount or policy commitment a person
deliberately phrased a certain way. The cost: every other tool result
(cart lookup, order status, `check_policy`) still needs that second LLM
call to become a coherent sentence, so this is a special case, not a
general optimization — worth knowing if you're reading the routing logic
expecting it to be uniform.

---

## The escalation side-channel exists because `interrupt()` can only resume once

**Decision:** `app/services/escalation_store.py` is a separate,
append-only message log, entirely outside the LangGraph checkpointed
conversation. Once a thread is escalated, every message from the user
goes there — not through the graph — until the agent resolves.

**Alternatives considered:** Routing every escalated message back through
`Command(resume=...)` for an ongoing back-and-forth. This doesn't work:
LangGraph's `interrupt()` pauses once and expects exactly one resume: you
can't repeatedly resume the same paused call to simulate a live
conversation.

**Tradeoff:** The side-channel's messages aren't part of the AI's own
message history — only the final resolution text is, since that's what
actually gets passed to `Command(resume=...)`. If a customer explains
important context to the human agent that never makes it into the
resolution text verbatim, the model won't see it on future turns. A more
complete version would fold the side-channel transcript into the graph
state on resolve; not done here as a deliberate scope cut.

---

## `check_policy` is real semantic search, not a full RAG stack

**Decision:** `app/services/policy_engine.py` does FAQ keyword match →
Qdrant (embedded mode) vector search with real OpenAI embeddings → an LLM
call that must answer only from retrieved context, citing sources, or
return `NO_ANSWER_FOUND`. What it deliberately does *not* have: chunking
beyond naive paragraph splitting, reranking, hybrid dense+sparse
retrieval, or a query planner.

**Note on "embedded mode":** `qdrant-client` supports talking to a real
Qdrant server over the network (which would need a URL/API key in `.env`)
or running entirely in-process via `QdrantClient(path=...)`, writing its
index straight to a local folder (`app/data/qdrant/`) with no server and
no network call. This project uses the latter — the only credential the
RAG path actually needs is `OPENAI_API_KEY`, for the embedding calls
themselves.

**Alternatives considered:** Building the same hybrid retrieval stack
already built for a separate project (Adaptive RAG — hybrid dense+sparse
fusion, semantic caching, a planner). Duplicating that here would make
both projects demonstrate the same skill twice instead of two different
ones.

**Tradeoff:** Retrieval quality here is good enough to prove the escalation
logic works, not necessarily good enough for a large or messy document
corpus. If retrieval quality genuinely mattered for this project's goals,
it would be a real gap, not a stylistic one.

---

## `topic_safety` was tried and abandoned for `self check input`

**Decision:** NeMo Guardrails' input rail (`app/data/guardrails_config/config.yml`)
consolidates prompt-injection detection and topic control into one
`self check input` rail, rather than using NeMo's dedicated `topic_safety`
library module.

**Alternatives considered:** `topic_safety` is the module actually built
for this — it exists specifically for topic classification. It was tried
first.

**What went wrong:** `topic_safety` requires a *separately registered
named model* under `models:` in the config (it's built for a dedicated
topic-classifier deployment, e.g. Llama Guard) — it doesn't resolve
against the already-configured "main" model the way `self_check_input`
does. Confirmed by actually building the config and hitting
`InvalidRailsConfigurationError`, not assumed from documentation.

**Tradeoff:** One consolidated LLM call instead of two, no second model
to stand up — but the injection-detection and topic-relevance prompts are
now coupled in one instruction block, which is slightly less precise than
two independently-tunable rails would be.

---

## Guardrails checks fail open, not closed

**Decision:** If the guardrails check itself errors (API timeout, network
issue), `app/services/guardrails.py` lets the message through to the
normal agent rather than blocking it.

**Alternatives considered:** Fail closed — block every message if the
safety check can't run.

**Tradeoff:** Prioritizes availability over paranoid safety. A safety-
critical system might reasonably choose the opposite. Every failure is
still logged (`logger.exception`), so an outage is visible in logs, not
silent — but a user experiencing it just sees the bot working normally,
with no visible sign guardrails were skipped for that message.

---

## Rate limiting is per-`user_id`, and deliberately not applied everywhere

**Decision:** `slowapi`, keyed by `user_id`/`thread_id` extracted from the
URL path (`app/core/rate_limit.py`), not by IP. Applied to every
state-changing or cost-bearing endpoint: `POST /chat/{user_id}` (15/min),
`POST /docs/upload` (5/min), `POST /support/resolve/{thread_id}` (20/min),
`POST /support/thread/{id}/reply` (30/min). **Not** applied to the GET
polling endpoints (`/support/pending`, `/support/thread/{id}`).

**Alternatives considered:** Per-IP limiting (the slowapi default) — but
this is multi-tenant, and many legitimate users could share an IP (NAT,
shared network). The thing actually worth protecting against is one user
hammering the LLM, not "too many requests from one network." Rate
limiting the GET polling routes was also considered and rejected: the
frontend polls `/support/thread/{id}` every 2.5 seconds, and a limit
there would break the demo's own working polling loop.

**Tradeoff:** In-memory bucket storage (slowapi's default) — fine for a
single process, but the buckets don't share state across multiple
instances. A real multi-instance deployment needs a Redis-backed `limits`
storage string instead.

---

## Auth is a stub, but enforcement is real

**Decision:** `POST /auth/token` (`app/routers/auth.py`) issues a JWT for
any `user_id`/`role` requested — no password or credential check, because
there's no real user database in this project. Everything *downstream* of
that is real: `app/core/auth.py`'s `require_matching_user` and
`require_support_agent` are genuinely enforced on every protected route,
verified by tests that assert 401/403 on missing, forged, expired, and
mismatched tokens (`tests/test_auth.py`), not just tests that assume
enforcement works.

**Alternatives considered:** Skipping auth entirely (documented as a known
limitation for a long stretch of this project's history) — simpler, but
"no auth" was the most-repeated line across every "known limitations"
section, meaning it was the first thing a careful reader would flag.
Building real credential-based login (password hashing, a user table) was
also considered and rejected as disproportionate: this project doesn't
have real user accounts to authenticate against, so a fake login step
would just be theater around real enforcement, not more real enforcement.

**Tradeoff:** Anyone can mint a token claiming to be any `user_id` or
`role` — this is *not* production security, and shouldn't be mistaken for
it. What's genuinely demonstrated is the *pattern*: token issuance,
verification, role-based access control, and websocket auth via query
param (since browsers can't set custom headers on a WS handshake). A real
system swaps the internals of `/auth/token` for actual credential
verification; nothing downstream changes.

---

## Tool-level authorization uses injected config, not LLM-supplied arguments

**Decision:** `get_user_cart` and `get_order_history`
(`app/graph/tools.py`) take zero LLM-visible arguments — they read the
current `user_id` from an injected `RunnableConfig`, bound from the
authenticated session in `chat.py`, not from anything the model outputs.
`get_order_by_id` still takes an LLM-supplied `order_id` (the user has to
say which order), but checks the order's owning `user_id` against the
injected one before returning any data.

**What this fixes, found by actually using the app:** the earlier version
had these tools take `user_id` as a normal LLM-controllable argument. Two
real problems followed: the model had no way to know the current user's
id unless the user stated it directly in the conversation (a genuine
usability bug — hit firsthand, not caught by any test beforehand), and
more seriously, nothing stopped the model from calling
`get_order_by_id(order_id=...)` for an order that belonged to a
*different* user and returning that data — an actual cross-user data leak
via tool-calling, not a hypothetical one.

**Alternatives considered:** Fixing only the usability bug by injecting
the user_id into the system prompt as text (e.g. "the current user's id is
user_001"). This would have resolved the symptom but not the underlying
issue — the model could still be convinced (accidentally or via prompt
injection) to call a tool with a different user_id, since the argument
would still be LLM-controllable either way.

**Tradeoff:** None significant on the cart/history tools — removing an
argument the model shouldn't have had control over in the first place.
`get_order_by_id`'s "not found" response is deliberately identical
whether an order doesn't exist or just isn't the current user's, so the
distinction itself doesn't leak that a given order_id is real.

---

## In-memory websocket connections do not need a message queue yet

**Decision:** `app/services/ws_manager.py` is a plain Python dict mapping
`thread_id → websocket`, held in process memory.

**Alternatives considered:** A message queue (Redis pub/sub, etc.) to
broadcast push notifications across processes.

**Tradeoff:** This works because the process that resolves an escalation
is the same process holding that user's open connection — true for a
single-process deployment, which is what this project actually runs as.
A message queue only becomes necessary once there are multiple server
instances behind a load balancer, where the instance handling
`POST /support/resolve/{id}` might not be the instance holding the user's
websocket. Building that now would be solving a scaling problem this
project doesn't have — the same reasoning behind not indexing
`/support/pending` yet (see the rate-limiting entry above for the
parallel case of in-memory state generally).

---

## A mock commerce DB instead of a real storefront

**Decision:** `app/services/mock_db.py` is a seeded SQLite database with
users, orders, and cart items — no storefront, no checkout flow, no
payment integration.

**Alternatives considered:** Building an actual e-commerce frontend for
the agent to sit in front of.

**Tradeoff:** The point of this project is agent orchestration and HITL
state management, not e-commerce — a seeded DB gives the agent realistic
structured data to query without weeks of frontend work that wouldn't
showcase anything this project is actually about. The cost is that
nothing here demonstrates real payment/checkout integration, which a
genuine production support agent would need to touch.

---

## Three users have deterministic, scripted data instead of everything being random

**Decision:** `user_001`, `user_002`, and `user_003` in `mock_db.py` have
fixed scenarios instead of random data — a duplicate charge, an order
outside the refund window, a shipment stuck processing for 18 days. Each
maps directly onto a real escalation trigger in the policy docs.

**Alternatives considered:** Fully random seed data for every user, same
as the rest of the seeded set.

**Tradeoff:** Testing "does escalation happen for the right reason"
becomes reproducible instead of requiring order data to be hand-crafted
through the API first every time — `tests/manual_eval_escalation.py`
depends on this directly. The cost: these three users behave differently
from the rest of the seeded set, which is a minor surprise if you're not
expecting it and go looking at their data.

---

## `thread_id` is just `user_id`

**Decision:** No separate session/thread concept — the LangGraph
checkpoint thread, the escalation side-channel key, and the websocket
connection key are all literally the `user_id`.

**Alternatives considered:** A separate `thread_id` per conversation,
allowing one user to have multiple concurrent support threads.

**Tradeoff:** Simpler by a wide margin, and sufficient to prove
multi-tenant thread isolation — the harder problem interrupts introduce.
The real cost: a user can only have one active conversation (and one
active escalation) at a time, which a real support system would probably
not want as a hard constraint.

---

## Money-related resolutions require server-enforced confirmation

**Decision:** `app/services/money_detection.py` (a keyword list + a
currency-amount regex) flags resolutions that look like a refund, credit,
chargeback, or currency amount. `POST /support/resolve/{thread_id}`
returns `confirmation_required` — touching nothing — unless the request
carries `confirmed: true`.

**Alternatives considered:** A frontend-only confirmation dialog (`confirm()`
before submitting). Simpler, but trivially bypassed by anyone hitting the
API directly instead of using the UI.

**Tradeoff:** `money_detection.py` is a heuristic, not an exhaustive
financial-intent classifier — a false positive costs an agent one extra
click; a false negative means a refund resolution ships without the
confirmation step, which is the actual risk worth tracking if this list
needs expanding later.

---

## Two demo frontends existed at different points; only one remains

**Decision:** `frontend/index.html` — plain HTML/JS, served same-origin by
FastAPI, holds a real persistent websocket, connects automatically the
instant a message escalates.

**Alternatives considered and actually built, then removed:** An earlier
version used Streamlit. It worked, but Streamlit reruns its entire script
on every interaction, so it can't hold a persistent connection open the
way a real client should — it had to poll instead. Once the plain-HTML
version proved the real push mechanism worked correctly (verified with
Playwright against a real browser, not just unit tests), the Streamlit
version stopped earning its place and was deleted rather than kept as a
second, weaker example.

**Tradeoff:** None significant — this was a straightforward improvement,
not a tradeoff. Worth recording anyway because "we tried something, it
worked well enough to ship, and we replaced it once something better
existed" is itself a decision worth being able to explain.


