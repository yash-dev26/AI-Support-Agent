"""
Live side-channel for messages exchanged between a user and a human
support agent WHILE a thread is escalated. Deliberately separate from the
LangGraph checkpointed conversation: LangGraph's interrupt() can only be
resumed once — you can't have an ongoing back-and-forth through repeated
interrupt/resume cycles on the same interrupt call. This collection is
where that back-and-forth actually lives.

Once the agent is done (POST /support/resolve/{thread_id}), the resolution
text they write is what gets fed back into the graph via
Command(resume=...); everything exchanged here informs the agent's
judgment but isn't part of the AI's own message history.
"""
from datetime import datetime, timezone

from app.core.deps import get_mongo_client


def _messages_col():
    return get_mongo_client()["support_agent"]["escalation_messages"]


def add_message(thread_id: str, sender: str, text: str) -> dict:
    """sender is 'user' or 'support'."""
    doc = {
        "thread_id": thread_id,
        "sender": sender,
        "text": text,
        "timestamp": datetime.now(timezone.utc),
    }
    _messages_col().insert_one(doc)
    return doc


def get_messages(thread_id: str) -> list[dict]:
    cursor = _messages_col().find({"thread_id": thread_id}).sort("timestamp", 1)
    return [
        {"sender": m["sender"], "text": m["text"], "timestamp": m["timestamp"].isoformat()}
        for m in cursor
    ]


def get_distinct_thread_ids() -> list[str]:
    """Every thread_id that has ever exchanged a message here — i.e. every
    thread that has escalated at some point, seeded mock user or not.

    This is what /support/pending actually scans (see support.py). It used
    to scan mock_db's seeded user list instead, which meant a thread for
    any user_id typed into the frontend that wasn't one of the pre-seeded
    ones (user_001-user_020) would never show up in the queue at all — not
    a polling problem, a wrong-source-of-truth problem. escalation_store
    only gets a message the moment a thread actually escalates, so this is
    the correct set to scan, and it doesn't require pre-seeding anything.
    """
    return _messages_col().distinct("thread_id")


