from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

from app import mock_db
from app import doc_store

load_dotenv()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool()
def get_cart_items(user_id: str) -> str:
    """Get the current shopping cart items for a given user_id."""
    items = mock_db.get_cart(user_id)
    if not items:
        return f"No items in cart for {user_id}."
    lines = [
        f"- {i['product_name']} x{i['quantity']} (₹{i['amount_cents'] / 100:.2f} each)"
        for i in items
    ]
    return "\n".join(lines)


@tool()
def get_order_history(user_id: str) -> str:
    """Get the order history for a given user_id, most recent first."""
    orders = mock_db.get_order_history(user_id)
    if not orders:
        return f"No order history for {user_id}."
    lines = [
        f"- {o['order_id']}: {o['product_name']} (₹{o['amount_cents'] / 100:.2f}) — {o['status']} — placed {o['placed_at']}"
        for o in orders
    ]
    return "\n".join(lines)


@tool()
def get_order_status(order_id: str) -> str:
    """Look up the current status of a specific order_id."""
    order = mock_db.get_order_status(order_id)
    if not order:
        return f"No order found with id {order_id}."
    return f"{order['order_id']} ({order['product_name']}) is currently: {order['status']}"


@tool()
def check_policy(query: str) -> str:
    """Search uploaded company documents (policies, FAQs) for an answer to the query.
    Use this before escalating to a human whenever the question might be answered
    by an existing policy document."""
    return doc_store.search(query)


@tool()
def human_interrupt_tool(query: str) -> str:
    """Request human assistance when the query cannot be resolved by tools or policy lookup."""
    human_reply = interrupt({
        "query": query,
        "message": "The agent has requested human assistance. Please provide input to help the agent continue.",
    })
    return human_reply["data"]


tools = [get_cart_items, get_order_history, get_order_status, check_policy, human_interrupt_tool]

llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You are a customer support agent. You can look up a user's cart, order history, "
    "and order status, and search company policy docs. If none of these resolve the "
    "user's issue, escalate to a human via human_interrupt_tool. Do not guess at policy "
    "— check_policy first."
)


def chatbot(state: State):
    messages = state["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def create_graph_chat(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
