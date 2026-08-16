"""
All tools the agent can call, in one place. Kept separate from graph.py
so the graph builder file only has to answer "how are nodes wired," not
"what does each tool do."

get_cart_items, get_order_history, and get_order_status all take the
current user's identity from an INJECTED RunnableConfig, not an
LLM-supplied argument. Earlier versions had the model pass user_id as a
regular tool argument — which meant the model had to be told the user_id
in the conversation (real usability bug: it had no other way to know),
and worse, nothing stopped it from calling these tools with a DIFFERENT
user_id and leaking another customer's cart or order data. Injected
config parameters are invisible to the LLM's tool schema (verified:
`get_cart_items.args` shows no user_id field at all) and are bound from
the actual authenticated session in chat.py, not from anything the model
can influence.
"""
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.services import mock_db
from app.services import policy_engine


def _current_user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


@tool()
def get_cart_items(config: RunnableConfig) -> str:
    """Get the current shopping cart items for the user in this conversation."""
    user_id = _current_user_id(config)
    items = mock_db.get_cart(user_id)
    if not items:
        return "No items in cart."
    lines = [
        f"- {i['product_name']} x{i['quantity']} (₹{i['amount_cents'] / 100:.2f} each)"
        for i in items
    ]
    return "\n".join(lines)


@tool()
def get_order_history(config: RunnableConfig) -> str:
    """Get the order history for the user in this conversation, most recent first."""
    user_id = _current_user_id(config)
    orders = mock_db.get_order_history(user_id)
    if not orders:
        return "No order history."
    lines = [
        f"- {o['order_id']}: {o['product_name']} (₹{o['amount_cents'] / 100:.2f}) — {o['status']} — placed {o['placed_at']}"
        for o in orders
    ]
    return "\n".join(lines)


@tool()
def get_order_status(order_id: str, config: RunnableConfig) -> str:
    """Look up the current status of a specific order_id, for the user in this conversation."""
    user_id = _current_user_id(config)
    order = mock_db.get_order_status(order_id)
    # Same response whether the order doesn't exist or belongs to someone
    # else — distinguishing the two would confirm to a user (or a model
    # that's been prompt-injected) that a given order_id exists at all,
    # even if it's not theirs.
    if not order or order.get("user_id") != user_id:
        return f"No order found with id {order_id}."
    return f"{order['order_id']} ({order['product_name']}) is currently: {order['status']}"


@tool()
def check_policy(query: str) -> str:
    """Check company FAQ and documentation for an answer to the query. Tries
    a fast FAQ match first, then a cited retrieval search over uploaded docs
    if no FAQ hits. ALWAYS call this before escalating — if it returns a
    result starting with "NO_ANSWER_FOUND", that is your signal to escalate
    via human_interrupt_tool. A user asking to speak to a human is not
    itself a reason to escalate; check_policy returning NO_ANSWER_FOUND is."""
    return policy_engine.answer(query)


@tool()
def human_interrupt_tool(query: str) -> str:
    """Request human assistance when the query cannot be resolved by tools or policy lookup."""
    human_reply = interrupt({
        "query": query,
        "message": "The agent has requested human assistance. Please provide input to help the agent continue.",
    })
    return human_reply["data"]


TOOLS = [get_cart_items, get_order_history, get_order_status, check_policy, human_interrupt_tool]
