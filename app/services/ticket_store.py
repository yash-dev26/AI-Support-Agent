"""
Persistence for formal support tickets, created by the agent's
create_support_ticket tool (see app/graph/tools.py).

Deliberately separate from escalation_store.py: escalation_store is the
live back-and-forth transcript exchanged between a user and a human once
a thread is paused, keyed by thread_id. A ticket is a structured record
of *why* the thread was escalated in the first place — issue_type,
details, priority, status — that a support queue UI or a real ticketing
system integration would want to read, filter, and update independently
of the message transcript. Mongo, same as escalation_store, since both
are mutable operational state rather than the deterministic seeded
commerce data mock_db.py owns.
"""
from datetime import datetime, timezone

from app.core.deps import get_mongo_client

OPEN = "open"
RESOLVED = "resolved"

# Anything mentioning money, a security concern, or an already-failed
# delivery is treated as higher priority than a generic question — this
# is a simple keyword heuristic, not a claim of real triage intelligence,
# but it's enough to make "high priority" tickets show up differently in
# a demo queue instead of everything being flatly the same priority.
_HIGH_PRIORITY_HINTS = (
    "refund", "charge", "chargeback", "duplicate", "fraud", "unauthorized",
    "damaged", "broken", "lost", "missing", "security", "hacked",
)


def _tickets_col():
    return get_mongo_client()["support_agent"]["tickets"]


def _infer_priority(issue_type: str, details: str) -> str:
    haystack = f"{issue_type} {details}".lower()
    return "high" if any(hint in haystack for hint in _HIGH_PRIORITY_HINTS) else "normal"


def create_ticket(ticket_id: str, user_id: str, issue_type: str, details: str) -> dict:
    """Upsert, not insert — create_support_ticket's caller (tools.py) is a
    LangGraph node whose interrupt() call means this exact line re-runs on
    every resume ("the graph resumes from the start of the node,
    re-executing all logic" — see langgraph.types.interrupt's docstring).
    ticket_id is derived from the tool_call_id, which IS stable across
    replay (it's part of the already-checkpointed AIMessage), so a resume
    calling this again with the same ticket_id must update the existing
    doc, not insert a second one — otherwise every human reply on a ticket
    would silently spawn a duplicate open ticket with a fresh random id.
    $setOnInsert means fields set here only apply the FIRST time; a
    replay that already advanced status/resolved_at won't get clobbered
    back to "open" by this upsert running again.
    """
    _tickets_col().update_one(
        {"ticket_id": ticket_id},
        {"$setOnInsert": {
            "ticket_id": ticket_id,
            "user_id": user_id,
            "issue_type": issue_type,
            "details": details,
            "priority": _infer_priority(issue_type, details),
            "status": OPEN,
            "created_at": datetime.now(timezone.utc),
            "resolved_at": None,
        }},
        upsert=True,
    )
    return get_ticket(ticket_id)


def get_ticket(ticket_id: str) -> dict | None:
    doc = _tickets_col().find_one({"ticket_id": ticket_id}, {"_id": 0})
    return doc


def update_status(ticket_id: str, status: str) -> dict | None:
    update = {"status": status}
    if status == RESOLVED:
        update["resolved_at"] = datetime.now(timezone.utc)
    _tickets_col().update_one({"ticket_id": ticket_id}, {"$set": update})
    return get_ticket(ticket_id)


def list_tickets_for_user(user_id: str) -> list[dict]:
    cursor = _tickets_col().find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1)
    return list(cursor)


def list_open_tickets() -> list[dict]:
    cursor = _tickets_col().find({"status": OPEN}, {"_id": 0}).sort("created_at", 1)
    return list(cursor)
