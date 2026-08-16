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


def route_after_tools(state: State) -> str:
    """Decide what happens after a tool call finishes.

    Every tool except human_interrupt_tool needs the LLM to turn raw data
    (a cart, an order status, a policy paragraph) into a coherent reply, so
    those route back to "chatbot" as usual. human_interrupt_tool is
    different: its return value IS the human's finished, deliberately
    worded answer — routing that back through the LLM would mean another
    model call re-paraphrasing something a person already answered
    correctly, adding latency and a real risk of it subtly rewording a
    refund amount or policy commitment. So when the most recent tool call
    was human_interrupt_tool, this ends the turn immediately and the
    human's message reaches the user verbatim.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            if msg.tool_calls and {tc["name"] for tc in msg.tool_calls} == {"human_interrupt_tool"}:
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
