"""
Node functions for the graph. The LLM itself is set up here since the
chatbot node is the only thing that calls it — graph.py shouldn't need
to know about model config to wire nodes together.
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from app.graph.state import State
from app.graph.tools import TOOLS

SYSTEM_PROMPT = (
    "You are a customer support agent. You have tools to look up a user's cart, "
    "order history, and order status, and check_policy to check company FAQ and "
    "documentation (it tries a fast FAQ match first, then a cited retrieval search).\n\n"
    "Escalation is a LAST RESORT, not a default. A user asking to talk to a human, "
    "by itself, is NOT a reason to escalate — always make a real attempt first: "
    "look up relevant data with your tools, and call check_policy before assuming "
    "you can't help. Only call human_interrupt_tool once check_policy's result "
    "starts with \"NO_ANSWER_FOUND\", or the issue genuinely requires manual action "
    "no tool covers (e.g. a policy document explicitly says a case needs human "
    "review). If check_policy returns an answer, use it to actually resolve the "
    "user's issue instead of escalating anyway."
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


def chatbot(state: State):
    messages = state["messages"]
    # `add_messages` coerces incoming dicts into LangChain message objects
    # (HumanMessage, AIMessage, ...) as soon as they land in state, so we
    # can't check messages[0]["role"] or messages[0].get("role") here —
    # those objects don't support dict-style access. Check the message
    # type instead.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = _get_llm_with_tools().invoke(messages)
    return {"messages": [response]}
