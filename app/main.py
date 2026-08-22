"""
App assembly only. Endpoint logic lives in app/routers/*; lifespan wires up
Mongo + the compiled graph once at startup and stores them in app/deps.py
so routers can read them without a circular import back to this file.
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.graph import create_graph_chat
from app.services import mock_db
from app.services import vector_store
from app.core import deps
from app.core.rate_limit import limiter
from app.core.agent_logging import configure_logging, get_logger
from app.routers import chat, support, metrics, docs, health, auth, users

load_dotenv()
configure_logging()

MONGODB_URI = os.getenv("MONGODB_URI")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Default 30 days. Set to "0" or "" to disable expiry entirely (keep
# every thread forever) -- see _get_checkpoint_ttl_seconds's docstring
# for what actually gets deleted and why an expired thread isn't a bug.
DEFAULT_CHECKPOINT_TTL_SECONDS = 60 * 60 * 24 * 30


def _get_checkpoint_ttl_seconds() -> int | None:
    """Parses CHECKPOINT_TTL_SECONDS from the environment. Separated out
    from lifespan() (which needs a live Mongo connection and so can't be
    unit tested without one) purely so this bit of config parsing —
    "what TTL value do we actually end up passing to MongoDBSaver" — can
    be tested in isolation.

    Returns None (no expiry) if the env var is unset, empty, "0", or not
    a valid non-negative integer (falling back to the safe default of
    "keep everything" on a malformed value, rather than silently deleting
    threads based on a typo).

    What this actually controls: MongoDBSaver's own built-in `ttl`
    parameter, which -- confirmed by reading langgraph-checkpoint-mongodb's
    source, not assumed -- creates a genuine MongoDB TTL index
    (`expireAfterSeconds`) on `created_at` for BOTH the checkpoints and
    checkpoint_writes collections. This means old thread state is expired
    by MongoDB's own background TTL monitor, not by any custom cleanup
    code, cron job, or worker process this app would otherwise need to
    run and keep alive itself.

    An expired thread isn't a data-loss bug: the next message from that
    user_id simply starts a fresh conversation (get_state() returns empty
    state, same as a user_id that has never messaged before) -- exactly
    the same as if they were a new user, which is the correct behavior
    for state that's aged out.

    One real interaction worth knowing, not "fixed" by this: if a thread
    expires while genuinely escalated (an open interrupt, unresolved
    ticket), the CHECKPOINT expires but the ticket itself
    (app/services/ticket_store.py, a separate Mongo collection) does not
    -- it stays "open" forever unless a human resolves it or something
    else reaps stale tickets. Checkpoint TTL and ticket lifecycle are two
    different concerns; this only solves the first one. See README's
    Known Limitations.
    """
    raw = os.getenv("CHECKPOINT_TTL_SECONDS", str(DEFAULT_CHECKPOINT_TTL_SECONDS))
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger("support_agent.startup").warning(
            "CHECKPOINT_TTL_SECONDS=%r is not a valid integer -- disabling checkpoint "
            "expiry (threads will be kept forever) rather than guessing.", raw,
        )
        return None
    return value if value > 0 else None


def _log_langsmith_tracing_status() -> None:
    """LangSmith tracing needs ZERO code changes to actually work — every
    LangChain/LangGraph call (the chatbot node's LLM invocation, each
    tool call, the RAG generation call in policy_engine.py) is
    auto-instrumented via LangChain's own callback system the moment
    LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set in the
    environment. This function doesn't enable anything — it just logs,
    once at startup, whether tracing is actually active, so "is this
    request supposed to show up in LangSmith" isn't a silent guess.
    Uses the LANGSMITH_* names (langsmith>=0.11 canonical); the older
    LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY names still work identically
    at the library level but aren't checked here to avoid this function
    silently getting out of sync with whichever alias someone actually set.
    """
    logger = get_logger("startup")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1")
    if tracing_enabled and os.getenv("LANGSMITH_API_KEY"):
        project = os.getenv("LANGSMITH_PROJECT", "default")
        logger.info(f"LangSmith tracing is ENABLED (project={project!r})", extra={"tag": "AGENT"})
    elif tracing_enabled:
        logger.warning(
            "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set -- "
            "tracing will silently no-op. Set both or neither.",
        )
    else:
        logger.info(
            "LangSmith tracing is disabled (LANGSMITH_TRACING unset) -- "
            "set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable it, "
            "no code changes required.",
            extra={"tag": "AGENT"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set in the environment variables.")

    _log_langsmith_tracing_status()

    mongo_client = MongoClient(MONGODB_URI)
    checkpoint_ttl = _get_checkpoint_ttl_seconds()
    checkpointer_cm = MongoDBSaver.from_conn_string(MONGODB_URI, ttl=checkpoint_ttl)
    checkpointer = checkpointer_cm.__enter__()
    graph_app = create_graph_chat(checkpointer=checkpointer)

    deps.set_mongo_client(mongo_client)
    deps.set_graph_app(graph_app)

    mock_db.seed(num_users=20)  # no-op if already seeded

    try:
        vector_store.index_all_docs()  # idempotent — safe to run every startup
    except Exception:
        logging.getLogger("support_agent.startup").exception(
            "Failed to index docs into Qdrant at startup — check_policy's RAG "
            "path will return NO_ANSWER_FOUND until this succeeds."
        )

    yield

    checkpointer_cm.__exit__(None, None, None)
    mongo_client.close()


app = FastAPI(
    title="Support Agent Infra",
    description="Durable, tool-calling support agent with human-in-the-loop escalation.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(support.router)
app.include_router(users.router)
app.include_router(metrics.router)
app.include_router(docs.router)
app.include_router(health.router)

# Demo frontend only — NOT the project's real interface (that's the API
# itself, see /docs). Served same-origin so the page's fetch()/WebSocket
# calls need no CORS configuration. Mounted last so it doesn't shadow any
# API route above.
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


@app.get("/")
def root():
    return {
        "service": "support-agent-infra",
        "docs": "/docs",
        "demo_ui": "/ui/",
        "note": "Interactive Swagger UI at /docs. A demo frontend with real-time "
                "push (auto-connecting websocket, no manual steps) is also "
                "available at /ui/.",
    }


