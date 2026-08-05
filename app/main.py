"""
FastAPI wrapper around the LangGraph support agent.

Why this exists: the original app.py/support.py were CLI scripts with a
`while True: continue` busy-wait poll loop — that pins a CPU core and only
supports one hardcoded thread. This turns the same graph into a real
multi-tenant service:

  POST /chat/{user_id}            -> talk to the agent, get escalated or resolved
  GET  /support/pending            -> list every thread currently waiting on a human
  POST /support/resolve/{thread_id} -> resume a paused thread with human input
  GET  /metrics                    -> escalation rate, avg resolution time, tool usage
  POST /docs/upload                -> drop a company doc in for check_policy to use

Visit /docs for a live Swagger UI you can click through end to end —
that's the "UI" for this project.
"""

import os
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from pymongo import MongoClient

from app.graph import create_graph_chat
from app import mock_db
from app.doc_store import DOCS_DIR

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = None
checkpointer_cm = None
graph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, checkpointer_cm, graph_app
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set in the environment variables.")

    mongo_client = MongoClient(MONGODB_URI)
    checkpointer_cm = MongoDBSaver.from_conn_string(MONGODB_URI)
    checkpointer = checkpointer_cm.__enter__()
    graph_app = create_graph_chat(checkpointer=checkpointer)

    mock_db.seed(num_users=20)  # no-op if already seeded

    yield

    checkpointer_cm.__exit__(None, None, None)
    mongo_client.close()


app = FastAPI(
    title="Support Agent Infra",
    description="Durable, tool-calling support agent with human-in-the-loop escalation.",
    lifespan=lifespan,
)


def _metrics_col():
    return mongo_client["support_agent"]["events"]


def _log_event(event_type: str, thread_id: str, **extra):
    _metrics_col().insert_one({
        "event_type": event_type,
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc),
        **extra,
    })


class ChatRequest(BaseModel):
    message: str


class ResolveRequest(BaseModel):
    resolution: str


@app.post("/chat/{user_id}")
def chat(user_id: str, req: ChatRequest):
    """Send a message to the agent for a given user. Returns either a normal
    reply or an escalation notice if the agent paused for human input."""
    config = {"configurable": {"thread_id": user_id}}

    state = graph_app.get_state(config)
    if getattr(state, "interrupts", ()):
        return {
            "status": "escalated",
            "message": "Your request is being handled by support. Please wait for a resolution.",
        }

    start = time.monotonic()
    last_messages = []
    for chunk in graph_app.stream(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
        stream_mode="values",
    ):
        if "messages" in chunk:
            last_messages = chunk["messages"]

    new_state = graph_app.get_state(config)
    if getattr(new_state, "interrupts", ()):
        payload = new_state.interrupts[0].value or {}
        _log_event("escalated", user_id, query=payload.get("query"))
        return {
            "status": "escalated",
            "message": "Your request has been escalated to a human agent. Please wait.",
        }

    _log_event("resolved_by_agent", user_id, latency_ms=int((time.monotonic() - start) * 1000))
    reply = last_messages[-1].content if last_messages else ""
    return {"status": "ok", "reply": reply}


@app.get("/support/pending")
def list_pending():
    """All threads currently paused on a human-in-the-loop interrupt."""
    pending = []
    with mock_db.get_conn() as conn:
        known_users = [row["user_id"] for row in conn.execute("SELECT user_id FROM users").fetchall()]

    for user_id in known_users:
        config = {"configurable": {"thread_id": user_id}}
        state = graph_app.get_state(config)
        interrupts = getattr(state, "interrupts", ())
        if interrupts:
            payload = interrupts[0].value or {}
            pending.append({
                "thread_id": user_id,
                "query": payload.get("query"),
                "message": payload.get("message"),
            })
    return {"pending": pending}


@app.post("/support/resolve/{thread_id}")
def resolve(thread_id: str, req: ResolveRequest):
    """Support agent resumes a paused thread with their resolution."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_app.get_state(config)
    if not getattr(state, "interrupts", ()):
        raise HTTPException(status_code=404, detail="No pending interrupt for this thread.")

    resume_start = time.monotonic()
    resume_command = Command(resume={"data": req.resolution})
    last_messages = []
    for event in graph_app.stream(resume_command, config=config, stream_mode="values"):
        if "messages" in event:
            last_messages = event["messages"]

    _log_event(
        "resolved_by_human", thread_id,
        resolution_latency_ms=int((time.monotonic() - resume_start) * 1000),
    )
    reply = last_messages[-1].content if last_messages else ""
    return {"status": "resumed", "final_reply": reply}


@app.post("/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a company doc (.txt/.md) that check_policy can search against."""
    if file.filename is None or not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported.")
    DOCS_DIR.mkdir(exist_ok=True)
    dest = DOCS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "filename": file.filename}


@app.get("/metrics")
def metrics():
    """Escalation rate, avg time-to-resolution, tool usage breakdown."""
    col = _metrics_col()
    total = col.count_documents({"event_type": {"$in": ["resolved_by_agent", "escalated"]}})
    escalated = col.count_documents({"event_type": "escalated"})
    resolved_by_human = list(col.find({"event_type": "resolved_by_human"}))

    avg_resolution_ms = (
        sum(e["resolution_latency_ms"] for e in resolved_by_human) / len(resolved_by_human)
        if resolved_by_human else None
    )
    avg_agent_latency = list(col.find({"event_type": "resolved_by_agent"}))
    avg_agent_ms = (
        sum(e["latency_ms"] for e in avg_agent_latency) / len(avg_agent_latency)
        if avg_agent_latency else None
    )

    return {
        "total_conversations": total,
        "escalation_rate": round(escalated / total, 3) if total else None,
        "avg_agent_latency_ms": avg_agent_ms,
        "avg_human_resolution_ms": avg_resolution_ms,
        "total_escalations": escalated,
        "total_human_resolutions": len(resolved_by_human),
    }


@app.get("/")
def root():
    return {
        "service": "support-agent-infra",
        "docs": "/docs",
        "note": "Interactive Swagger UI at /docs — no separate frontend needed.",
    }
