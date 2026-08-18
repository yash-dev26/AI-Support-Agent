"""
All tools the agent can call, in one place. Kept separate from graph.py
so the graph builder file only has to answer "how are nodes wired," not
"what does each tool do."

get_user_cart, get_order_history, get_latest_order, get_order_by_id, and
create_support_ticket all take the current user's identity from an
INJECTED RunnableConfig, not an LLM-supplied argument. Earlier versions
had the model pass user_id as a regular tool argument -- which meant the
model had to be told the user_id in the conversation (real usability
bug: it had no other way to know), and worse, nothing stopped it from
calling these tools with a DIFFERENT user_id and leaking another
customer's cart or order data. Injected config parameters are invisible
to the LLM's tool schema (verified: `get_user_cart.args` shows no
user_id field at all) and are bound from the actual authenticated
session in chat.py, not from anything the model can influence. This is
why these tools don't literally match a "get_user_cart(user_id)"
free-form spec -- the signature that LOOKS more capable (accepting
user_id as an argument) is the less secure one, so it was deliberately
not implemented that way.
"""
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from typing_extensions import Annotated

from app.core.agent_logging import get_logger, log_tool_execution
from app.services import mock_db
from app.services import policy_engine
from app.services import ticket_store

logger = get_logger("tools")


def _current_user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


def _format_order(o: dict) -> str:
    """Shared formatting for get_latest_order/get_order_by_id/get_order_history
    so all three surface the same fields (status, items, tracking) instead
    of drifting apart over time. Handles a null tracking_number gracefully
    -- orders that are still "processing" never had one assigned."""
    tracking = (
        f", tracked via {o['carrier']} ({o['tracking_number']})"
        if o.get("tracking_number") else ", no tracking number yet"
    )
    return (
        f"{o['order_id']}: {o['product_name']} (\u20b9{o['amount_cents'] / 100:.2f}) "
        f"-- {o['status']}{tracking} -- placed {o['placed_at']}"
    )


@tool()
@log_tool_execution
def get_user_cart(config: RunnableConfig) -> str:
    """Get the current shopping cart items for the user in this conversation."""
    user_id = _current_user_id(config)
    try:
        items = mock_db.get_cart(user_id)
    except Exception:
        logger.exception("get_user_cart failed for a user")
        return "Couldn't load the cart right now -- try again in a moment."
    if not items:
        return "No items in cart."
    lines = [
        f"- {i['product_name']} x{i['quantity']} (\u20b9{i['amount_cents'] / 100:.2f} each)"
        for i in items
    ]
    return "\n".join(lines)


@tool()
@log_tool_execution
def get_order_history(config: RunnableConfig) -> str:
    """Get the FULL order history for the user in this conversation, most
    recent first. Use get_latest_order instead if the user is only asking
    about their most recent order -- it's cheaper to reason over and is
    what phrases like "my order", "my latest order", or "the thing I just
    bought" almost always mean."""
    user_id = _current_user_id(config)
    try:
        orders = mock_db.get_order_history(user_id)
    except Exception:
        logger.exception("get_order_history failed for a user")
        return "Couldn't load order history right now -- try again in a moment."
    if not orders:
        return "No order history."
    return "\n".join(f"- {_format_order(o)}" for o in orders)


@tool()
@log_tool_execution
def get_latest_order(config: RunnableConfig) -> str:
    """Get the most recent order for the user in this conversation --
    status, items, and tracking info. This is the right tool for implicit
    references like "my latest order", "the order I just placed", or "I
    had an issue with my order" when no order_id has been given -- don't
    make the user repeat information you can already look up."""
    user_id = _current_user_id(config)
    try:
        order = mock_db.get_latest_order(user_id)
    except Exception:
        logger.exception("get_latest_order failed for a user")
        return "Couldn't load the latest order right now -- try again in a moment."
    if not order:
        return "This user has no orders yet."
    return _format_order(order)


@tool()
@log_tool_execution
def get_order_by_id(order_id: str, config: RunnableConfig) -> str:
    """Look up the current status, items, and tracking info of a specific
    order_id, for the user in this conversation."""
    user_id = _current_user_id(config)
    try:
        order = mock_db.get_order_status(order_id)
    except Exception:
        logger.exception("get_order_by_id failed for order_id=%s", order_id)
        return "Couldn't look up that order right now -- try again in a moment."
    # Same response whether the order doesn't exist or belongs to someone
    # else -- distinguishing the two would confirm to a user (or a model
    # that's been prompt-injected) that a given order_id exists at all,
    # even if it's not theirs.
    if not order or order.get("user_id") != user_id:
        return f"No order found with id {order_id}."
    return _format_order(order)


@tool()
@log_tool_execution
def check_policy(query: str, config: RunnableConfig) -> str:
    """Check company FAQ and documentation for an answer to the query. Tries
    a fast FAQ match first, then a cited retrieval search over uploaded docs
    if no FAQ hits. ALWAYS call this before escalating -- if it returns a
    result starting with "NO_ANSWER_FOUND", that is your signal to escalate
    via create_support_ticket. A user asking to speak to a human is not
    itself a reason to escalate -- check_policy returning NO_ANSWER_FOUND is."""
    # config isn't otherwise used here -- policy lookup isn't user-scoped,
    # unlike get_user_cart etc. It's accepted purely so log_tool_execution
    # can attribute this call to a thread_id instead of logging "unknown",
    # since check_policy is one of the two most-called tools per the
    # system prompt's "always call this before escalating" instruction.
    try:
        return policy_engine.answer(query)
    except Exception:
        logger.exception("check_policy failed for query=%r", query)
        # Unexpected schema/empty result from the retrieval layer shouldn't
        # crash the turn -- treat it the same as a genuine miss so the
        # model's existing "then escalate" instructions still apply.
        return "NO_ANSWER_FOUND"


@tool()
@log_tool_execution
def create_support_ticket(
    issue_type: str,
    details: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Open a formal support ticket and hand this conversation off to a
    human agent. Use this once check_policy has returned NO_ANSWER_FOUND,
    or the issue genuinely requires manual action no tool covers (e.g. a
    policy document explicitly says a case needs human review). A user
    asking to talk to a human is NOT itself a reason to call this -- make
    a real attempt with your other tools and check_policy first.

    issue_type should be a short category (e.g. "duplicate_charge",
    "missing_order", "account_access"). details should summarize what the
    user needs and anything you've already learned from your tool calls,
    since the human agent picking this up starts cold otherwise."""
    user_id = _current_user_id(config)
    # Deterministic from tool_call_id, NOT uuid4(). interrupt() causes
    # LangGraph to replay this entire function from the top on every
    # resume -- a random id here would mint a fresh duplicate ticket on
    # every single human reply. tool_call_id is part of the
    # already-checkpointed AIMessage, so it's identical across replays;
    # paired with ticket_store.create_ticket's upsert, that makes this
    # call idempotent instead of a silent duplicate-ticket bug.
    ticket_id = f"tkt_{tool_call_id}"

    try:
        ticket_store.create_ticket(ticket_id, user_id, issue_type, details)
    except Exception:
        # A failed write to the ticket store shouldn't block the user from
        # reaching a human -- fall back to escalating without a persisted
        # ticket record rather than silently failing the whole turn.
        logger.exception("Failed to persist ticket %s for user %s -- escalating without one", ticket_id, user_id)
        ticket_id = None

    human_reply = interrupt({
        "query": details,
        "issue_type": issue_type,
        "ticket_id": ticket_id,
        "message": "The agent has opened a support ticket and requested human assistance. "
                   "Please provide input to help the agent continue.",
    })

    if ticket_id:
        try:
            ticket_store.update_status(ticket_id, ticket_store.RESOLVED)
        except Exception:
            logger.exception("Failed to mark ticket %s resolved", ticket_id)

    # Defensive: tolerate a human-resume payload missing the expected
    # "data" key rather than raising a KeyError mid-conversation.
    if not isinstance(human_reply, dict) or "data" not in human_reply:
        logger.warning("Unexpected resume payload shape for ticket %s: %r", ticket_id, human_reply)
        return "A human agent has responded, but their reply couldn't be read. Please check back shortly."
    return human_reply["data"]


TOOLS = [
    get_user_cart,
    get_order_history,
    get_latest_order,
    get_order_by_id,
    check_policy,
    create_support_ticket,
]
