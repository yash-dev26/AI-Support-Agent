"""
Integration test for a specific correctness property that unit tests on
ticket_store.py alone can't prove: that create_support_ticket, run
through a REAL LangGraph interrupt()/resume cycle (not a fake), results
in exactly one ticket — not one per resume.

Why this needed a real graph instead of mocking: langgraph.types.interrupt
documents that "the graph resumes from the start of the node,
re-executing all logic" — this is LangGraph's actual replay behavior, not
something a hand-rolled fake would reproduce unless it deliberately
modeled it. The bug this test guards is easy to introduce by accident
(e.g. reverting ticket_id to uuid4() instead of the tool_call_id derived
one) and a mocked-graph test would happily pass right through it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mongomock
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from app.core import deps
from app.graph.tools import create_support_ticket
from app.services import ticket_store


@pytest.fixture(autouse=True)
def fake_mongo():
    deps.set_mongo_client(mongomock.MongoClient())
    yield


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_single_tool_graph():
    """A minimal one-node graph that always calls create_support_ticket —
    no LLM involved, since this test only cares about replay behavior
    around the interrupt, not tool selection (that's nodes.py's system
    prompt's job, exercised elsewhere)."""
    def force_ticket_call(state):
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "create_support_ticket",
            "args": {"issue_type": "missing_order", "details": "Order never arrived"},
            "id": "call_fixed_id_123",
        }])]}

    builder = StateGraph(_State)
    builder.add_node("force_call", force_ticket_call)
    builder.add_node("tools", ToolNode([create_support_ticket]))
    builder.add_edge(START, "force_call")
    builder.add_edge("force_call", "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_create_support_ticket_survives_replay_with_exactly_one_ticket():
    graph = _build_single_tool_graph()
    config = {"configurable": {"thread_id": "user_replay_test", "user_id": "user_replay_test"}}

    # First run: pauses at interrupt() inside create_support_ticket.
    graph.invoke({"messages": [HumanMessage(content="my order never showed up")]}, config=config)

    state = graph.get_state(config)
    assert state.interrupts, "expected the graph to be paused on interrupt()"

    tickets_after_first_pause = ticket_store.list_tickets_for_user("user_replay_test")
    assert len(tickets_after_first_pause) == 1

    # Resume: this is exactly what replays create_support_ticket's body
    # from the top, per langgraph's own documented interrupt() semantics.
    graph.invoke(Command(resume={"data": "A human confirmed a reshipment is on the way."}), config=config)

    tickets_after_resume = ticket_store.list_tickets_for_user("user_replay_test")
    assert len(tickets_after_resume) == 1, (
        "create_support_ticket's replay-on-resume created a duplicate ticket — "
        "ticket_id must stay derived from tool_call_id (stable across replay), "
        "not uuid4() (regenerated every replay)."
    )
    assert tickets_after_resume[0]["status"] == ticket_store.RESOLVED
