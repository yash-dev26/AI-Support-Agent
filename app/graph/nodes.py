"""
Node functions for the graph. The LLM itself is set up here since the
chatbot node is the only thing that calls it — graph.py shouldn't need
to know about model config to wire nodes together.
"""
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.core.agent_logging import get_logger, log_agent
from app.core.retry import with_retry
from app.graph.state import State
from app.graph.tools import TOOLS

logger = get_logger("nodes")

# How many of the most recent "turns" (a turn = one HumanMessage plus
# everything the agent did in response to it — AIMessages, ToolMessages,
# up to the next HumanMessage) are sent to the LLM. Bounds the prompt
# size, and therefore both latency and $ per call, for a long-running
# thread instead of letting it grow unboundedly for the life of the
# checkpointed conversation. See _trim_to_recent_turns for why the unit
# is "whole turns" and not "last N messages" or "last N tokens".
MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "12"))

SYSTEM_PROMPT = (
    "You are a customer support agent. You have tools to look up a user's cart "
    "(get_user_cart), full order history (get_order_history), their single most "
    "recent order (get_latest_order), a specific order by id (get_order_by_id), "
    "and check_policy to check company FAQ and documentation (it tries a fast FAQ "
    "match first, then a cited retrieval search).\n\n"
    "Resolve implicit context yourself instead of asking the user to repeat "
    "information you can look up. If they refer to \"my order\", \"my latest "
    "order\", or describe an issue without giving an order_id, call "
    "get_latest_order first — don't ask them which order they mean unless "
    "get_latest_order comes back empty or they clearly have multiple orders in "
    "play (e.g. they mention two different products). Only use get_order_by_id "
    "when the user has given you a specific order_id, or get_order_history when "
    "they're asking about their orders broadly (e.g. \"what have I ordered "
    "recently\") rather than one specific order.\n\n"
    "Escalation is a LAST RESORT, not a default. A user asking to talk to a human, "
    "by itself, is NOT a reason to escalate — always make a real attempt first: "
    "look up relevant data with your tools, and call check_policy before assuming "
    "you can't help. Only call create_support_ticket once check_policy's result "
    "starts with \"NO_ANSWER_FOUND\", or the issue genuinely requires manual action "
    "no tool covers (e.g. a policy document explicitly says a case needs human "
    "review). If check_policy returns an answer, use it to actually resolve the "
    "user's issue instead of escalating anyway. When you do call "
    "create_support_ticket, give it a short issue_type category and a details "
    "summary that includes what you've already learned from your other tool "
    "calls, so the human agent doesn't start cold.\n\n"
    "If you need more than one independent piece of information to answer — e.g. "
    "both their latest order AND a policy check, or their cart AND their order "
    "history — request all of those tool calls together in the SAME turn rather "
    "than one at a time across multiple turns. They don't depend on each other's "
    "results, so there's no reason to wait for one before asking for the next; "
    "requesting them together lets them run in parallel instead of one round trip "
    "per tool. Only sequence tool calls across turns when a later call genuinely "
    "needs a piece of information only the earlier one's result provides."
)

# Constructed lazily, on first actual use, rather than at import time.
# Building the OpenAI client eagerly here means importing this module (or
# anything that imports it, like graph.py) requires a real OPENAI_API_KEY
# even to test pure logic — e.g. route_after_tools has nothing to do with
# the LLM, but a naive eager import chain would still demand credentials
# just to run that test. Deferring construction until chatbot() actually
# runs keeps "can I import this" and "do I have live credentials" separate.
_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = init_chat_model(model_provider="openai", model="gpt-4.1")
        _llm_with_tools = llm.bind_tools(tools=TOOLS)
    return _llm_with_tools


def _trim_to_recent_turns(messages: list, max_turns: int = MAX_CONTEXT_TURNS) -> list:
    """Groups messages into turns (each starting with a HumanMessage) and
    keeps only the most recent max_turns turns.

    Turns, not a raw message-count or token cutoff, are the only safe unit
    to cut on here: when the model calls a tool, the resulting AIMessage
    (with tool_calls) MUST be immediately followed by a ToolMessage for
    every one of those calls, or the OpenAI API rejects the request
    outright with a 400 — "tool_calls must be followed by tool messages".
    A naive "keep the last N messages" trim could easily slice a turn in
    half and leave a dangling tool call with no response, breaking every
    subsequent call on that thread. Keeping whole turns intact makes that
    structurally impossible.
    """
    if not messages:
        return messages

    turns: list[list] = []
    current: list = []
    for m in messages:
        if isinstance(m, HumanMessage) and current:
            turns.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        turns.append(current)

    if len(turns) <= max_turns:
        return messages

    kept = turns[-max_turns:]
    return [m for turn in kept for m in turn]


@with_retry()
def _invoke_llm(llm, messages):
    """Separated out from chatbot() purely so the retry decorator has a
    single, obviously-scoped target — retrying "the network call to
    OpenAI" is exactly right; retrying chatbot() as a whole would also
    re-run the trimming/logging logic around it on every attempt, which
    is harmless but noisy and not what's actually being retried."""
    return llm.invoke(messages)


def chatbot(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("user_id", "unknown")
    messages = state["messages"]
    # `add_messages` coerces incoming dicts into LangChain message objects
    # (HumanMessage, AIMessage, ...) as soon as they land in state, so we
    # can't check messages[0]["role"] or messages[0].get("role") here —
    # those objects don't support dict-style access. Check the message
    # type instead.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    system, rest = messages[0], messages[1:]
    trimmed_rest = _trim_to_recent_turns(rest)
    if len(trimmed_rest) < len(rest):
        log_agent(
            logger, thread_id,
            f"trimmed context: {len(rest)} -> {len(trimmed_rest)} messages "
            f"(kept last {MAX_CONTEXT_TURNS} turns)",
        )
    messages = [system] + trimmed_rest

    log_agent(logger, thread_id, f"invoking LLM with {len(messages)} messages in context")
    response = _invoke_llm(_get_llm_with_tools(), messages)

    usage = getattr(response, "usage_metadata", None)
    if usage:
        # Local, always-on visibility into token usage/cost even without
        # LangSmith configured (see main.py's _log_langsmith_tracing_status) —
        # this is deliberately just a log line, not a billing system: no
        # per-model $ pricing table to hand-maintain and let go stale.
        # LangSmith (when enabled) already tracks real, current pricing
        # centrally; this is the always-available fallback for local dev.
        log_agent(
            logger, thread_id,
            f"token usage: {usage.get('input_tokens', '?')} in / "
            f"{usage.get('output_tokens', '?')} out / "
            f"{usage.get('total_tokens', '?')} total",
        )

    if getattr(response, "tool_calls", None):
        tool_names = ", ".join(tc["name"] for tc in response.tool_calls)
        log_agent(logger, thread_id, f"model requested tool call(s): {tool_names}")
    else:
        log_agent(logger, thread_id, "model produced a direct reply, no tool calls")
    return {"messages": [response]}


