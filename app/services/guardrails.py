"""
Input guardrails via NeMo Guardrails: prompt injection detection and topic
control, checked before a user's message ever reaches the LangGraph agent
(and before it reaches a human agent too, if the thread is escalated —
see chat.py, this runs first regardless of state).

Deliberately uses ONE consolidated `self check input` rail rather than
NeMo's separate `topic_safety` library module. `topic_safety` requires a
dedicated NAMED model registered under `models:` (it's built for a
separate topic-classifier model, e.g. a Llama Guard-style deployment) —
real friction hit while building this: it wouldn't resolve against the
already-configured "main" model the way `self_check_input` does. Folding
injection detection and topic relevance into one prompt is simpler, is
one LLM call instead of two, and doesn't require standing up a second
model just for this project's scope.

Uses GenerationRailsOptions(dialog=False, output=False, ...) so only the
input rail itself runs. Without that, NeMo also generates a full
throwaway reply via its own "general" flow on every allowed message —
wasted since our own graph generates the real response.
"""
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationRailsOptions

from app.core.agent_logging import get_logger, log_guardrail

logger = get_logger("guardrails")

CONFIG_PATH = Path(__file__).parent.parent / "data" / "guardrails_config"

_INPUT_ONLY_OPTIONS = GenerationOptions(
    rails=GenerationRailsOptions(
        input=True, output=False, dialog=False, retrieval=False,
        tool_output=False, tool_input=False,
    )
)

_rails = None


def _get_rails() -> LLMRails:
    """Constructed lazily so importing this module doesn't require a real
    OPENAI_API_KEY — only the first actual check does (RailsConfig parsing
    is just YAML/Colang, no LLM call happens at construction time)."""
    global _rails
    if _rails is None:
        config = RailsConfig.from_path(str(CONFIG_PATH))
        _rails = LLMRails(config)
    return _rails


async def check_message(user_message: str, thread_id: str = "unknown") -> tuple[bool, str | None]:
    """Returns (blocked, refusal_message). If not blocked, refusal_message
    is None and the caller should proceed normally.

    thread_id is optional (defaults to "unknown") purely so existing
    callers/tests that only care about the blocked/allowed logic don't
    need to pass one — chat.py, the real caller, always supplies the
    actual user_id so the [GUARDRAIL STATUS] log line is traceable back
    to a specific conversation.

    With dialog=False, an allowed message comes back unchanged (NeMo just
    passes it through); a blocked message comes back as the rail's refusal
    text instead. Comparing the two is how we detect "was this blocked" —
    confirmed empirically against a fake LLM rather than assumed, since
    NeMo doesn't otherwise surface a clean boolean here with this option set.
    """
    rails = _get_rails()
    try:
        result = await rails.generate_async(
            messages=[{"role": "user", "content": user_message}],
            options=_INPUT_ONLY_OPTIONS,
        )
    except Exception:
        # Fail OPEN, not closed: if the guardrails check itself errors (API
        # hiccup, etc.), letting the message through to the normal agent
        # keeps the support bot available. A stricter safety-critical
        # system might choose to fail closed instead — deliberate tradeoff
        # for what this project is, not an oversight.
        logger.exception("Guardrails check failed — failing open, message will proceed normally")
        log_guardrail(logger, thread_id, "self_check_input", "fail_open", reason="rails.generate_async raised")
        return False, None

    content = result.response[0]["content"]
    if content != user_message:
        log_guardrail(logger, thread_id, "self_check_input", "blocked")
        return True, content
    log_guardrail(logger, thread_id, "self_check_input", "passed")
    return False, None


