"""
Graph wiring only. Node logic lives in nodes.py, tool definitions live in
tools.py — this file's only job is "how are they connected."
"""
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

from app.graph.state import State
from app.graph.nodes import chatbot
from app.graph.tools import TOOLS

load_dotenv()


# Tool names whose return value is a human's finished, deliberately worded
# answer rather than raw data the LLM still needs to turn into a reply.
# "human_interrupt_tool" is kept here for backward compatibility with
# existing routing tests and any external caller still referencing the
# older tool name, even though it's no longer exposed to the LLM directly
# — create_support_ticket (see tools.py) is what actually escalates now,
# but it goes through the exact same interrupt()-based mechanism.
TERMINAL_TOOLS = {"human_interrupt_tool", "create_support_ticket"}


def route_after_tools(state: State) -> str:
    """Decide what happens after a tool call finishes.

    Every ordinary tool needs the LLM to turn raw data (a cart, an order
    status, a policy paragraph) into a coherent reply, so those route back
    to "chatbot" as usual. A tool in TERMINAL_TOOLS is different: its
    return value IS the human's finished, deliberately worded answer —
    routing that back through the LLM would mean another model call
    re-paraphrasing something a person already answered correctly, adding
    latency and a real risk of it subtly rewording a refund amount or
    policy commitment. So when every tool call in the most recent turn was
    a terminal one, this ends the turn immediately and the human's message
    reaches the user verbatim. A turn mixing a terminal tool with an
    ordinary one falls back to "chatbot" — the LLM still needs to
    synthesize the ordinary tool's result either way.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            call_names = {tc["name"] for tc in msg.tool_calls}
            if call_names and call_names.issubset(TERMINAL_TOOLS):
                return END
            return "chatbot"
    return "chatbot"


def build_graph() -> StateGraph:
    tools_node = ToolNode(tools=TOOLS)

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_conditional_edges("tools", route_after_tools, {"chatbot": "chatbot", END: END})
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder


graph_builder = build_graph()
graph = graph_builder.compile()


def create_graph_chat(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


