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



# --- _trim_to_recent_turns (app/graph/nodes.py) --------------------------
#
# In a separate section (not a separate file) since this file already
# covers graph-level behavior end to end, and trimming is squarely part
# of "what happens to a thread's messages before the next LLM call" —
# the same territory as route_after_tools above.

from app.graph.nodes import _trim_to_recent_turns, SYSTEM_PROMPT


def _turn(human_text: str, *rest) -> list:
    """One HumanMessage plus whatever AI/Tool messages followed it —
    matches how _trim_to_recent_turns groups messages."""
    return [HumanMessage(content=human_text), *rest]


def test_trim_is_a_no_op_when_under_the_turn_limit():
    messages = _turn("hi") + _turn("how are you")
    assert _trim_to_recent_turns(messages, max_turns=12) == messages


def test_trim_keeps_only_the_most_recent_n_turns():
    turns = [_turn(f"message {i}") for i in range(5)]
    messages = [m for t in turns for m in t]

    trimmed = _trim_to_recent_turns(messages, max_turns=2)

    # only the last 2 turns' HumanMessages should survive
    human_texts = [m.content for m in trimmed if isinstance(m, HumanMessage)]
    assert human_texts == ["message 3", "message 4"]


def test_trim_never_splits_a_tool_call_from_its_tool_message():
    # An AIMessage with tool_calls MUST stay glued to its ToolMessage(s) —
    # splitting them would produce a message list the OpenAI API rejects
    # outright (a tool_call with no matching tool response).
    old_turn = _turn(
        "old question",
        AIMessage(content="", tool_calls=[{"name": "get_user_cart", "args": {}, "id": "call_1"}]),
        ToolMessage(content="No items in cart.", tool_call_id="call_1"),
    )
    recent_turns = [_turn(f"recent question {i}") for i in range(3)]
    messages = old_turn + [m for t in recent_turns for m in t]

    trimmed = _trim_to_recent_turns(messages, max_turns=3)

    # the old turn (with its tool call) should be dropped ENTIRELY, not
    # partially — never a dangling AIMessage(tool_calls=...) with no
    # matching ToolMessage anywhere in the trimmed result
    tool_call_ids_requested = {
        tc["id"] for m in trimmed if isinstance(m, AIMessage) for tc in m.tool_calls
    }
    tool_call_ids_answered = {
        m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)
    }
    assert tool_call_ids_requested == tool_call_ids_answered
    assert "call_1" not in tool_call_ids_requested  # confirms the old turn was actually dropped


def test_trim_keeps_a_turn_with_multiple_tool_calls_intact():
    turn_with_two_calls = _turn(
        "what's my cart and my latest order",
        AIMessage(content="", tool_calls=[
            {"name": "get_user_cart", "args": {}, "id": "call_a"},
            {"name": "get_latest_order", "args": {}, "id": "call_b"},
        ]),
        ToolMessage(content="cart data", tool_call_id="call_a"),
        ToolMessage(content="order data", tool_call_id="call_b"),
    )
    trimmed = _trim_to_recent_turns(turn_with_two_calls, max_turns=1)
    assert trimmed == turn_with_two_calls


def test_trim_handles_empty_message_list():
    assert _trim_to_recent_turns([], max_turns=5) == []


def test_system_prompt_encourages_batching_independent_tool_calls():
    # Cheap regression check that the parallel-tool-call guidance wasn't
    # accidentally removed in a future prompt edit — doesn't (and can't,
    # without a live model) verify the LLM actually complies, just that
    # the instruction is present.
    assert "same turn" in SYSTEM_PROMPT.lower()
    assert "parallel" in SYSTEM_PROMPT.lower()
