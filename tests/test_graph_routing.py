"""
Tests for route_after_tools — the routing decision that skips a second LLM
call after a human resolves an escalation. Pure function, no LLM or Mongo
needed, so this runs in CI like the other unit tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END

from app.graph.graph import route_after_tools


def test_routes_to_end_after_human_interrupt_tool_resolves():
    state = {
        "messages": [
            HumanMessage(content="my payment was double charged"),
            AIMessage(
                content="",
                tool_calls=[{"name": "human_interrupt_tool", "args": {"query": "duplicate charge"}, "id": "call_1"}],
            ),
            ToolMessage(content="Refund issued, duplicate reversed.", tool_call_id="call_1"),
        ]
    }
    assert route_after_tools(state) == END


def test_routes_to_chatbot_after_ordinary_tool_call():
    state = {
        "messages": [
            HumanMessage(content="what's in my cart?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_cart_items", "args": {"user_id": "user_001"}, "id": "call_2"}],
            ),
            ToolMessage(content="- Mechanical Keyboard x1", tool_call_id="call_2"),
        ]
    }
    assert route_after_tools(state) == "chatbot"


def test_routes_to_chatbot_when_no_ai_message_present():
    # defensive case — shouldn't happen in practice, but should degrade
    # to the safe default rather than error
    state = {"messages": [HumanMessage(content="hi")]}
    assert route_after_tools(state) == "chatbot"


def test_routes_to_chatbot_on_mixed_tool_calls_in_one_turn():
    # if the model ever calls human_interrupt_tool alongside another tool
    # in the same turn, fall back to the safe default (let the LLM handle
    # synthesizing both results) rather than guessing which one "wins"
    state = {
        "messages": [
            HumanMessage(content="what's my order status and also escalate this"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "ord_1"}, "id": "call_3"},
                    {"name": "human_interrupt_tool", "args": {"query": "..."}, "id": "call_4"},
                ],
            ),
            ToolMessage(content="ord_1 is processing", tool_call_id="call_3"),
            ToolMessage(content="Handled manually.", tool_call_id="call_4"),
        ]
    }
    assert route_after_tools(state) == "chatbot"


def test_finds_most_recent_ai_message_not_an_earlier_one():
    # a human_interrupt_tool call earlier in the conversation shouldn't
    # affect routing for the current turn's ordinary tool call
    state = {
        "messages": [
            HumanMessage(content="earlier escalated question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "human_interrupt_tool", "args": {"query": "old"}, "id": "call_5"}],
            ),
            ToolMessage(content="resolved earlier", tool_call_id="call_5"),
            HumanMessage(content="now what's in my cart?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_cart_items", "args": {"user_id": "user_001"}, "id": "call_6"}],
            ),
            ToolMessage(content="- USB-C Hub x2", tool_call_id="call_6"),
        ]
    }
    assert route_after_tools(state) == "chatbot"
