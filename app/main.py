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
from app.routers import chat, support, metrics, docs, health, auth

load_dotenv()
logging.basicConfig(level=logging.INFO)

MONGODB_URI = os.getenv("MONGODB_URI")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set in the environment variables.")

    mongo_client = MongoClient(MONGODB_URI)
    checkpointer_cm = MongoDBSaver.from_conn_string(MONGODB_URI)
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
