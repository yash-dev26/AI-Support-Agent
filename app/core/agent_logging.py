"""
Structured, human-readable logging for the agent's own operations.

Distinct from log_event() in app/core/deps.py, which writes structured
events to Mongo for metrics/analytics — this module is purely about
making stdout/`docker logs` readable during development and demo
recordings: every app-level line carries a bracketed [TAG] a human can
scan or grep for, instead of the request being buried in LangChain /
LangGraph / NeMo Guardrails' own internal DEBUG/INFO chatter.

Usage:
    from app.core.agent_logging import get_logger, log_agent, log_guardrail, log_response

    logger = get_logger("nodes")
    log_agent(logger, thread_id, "invoking LLM with 3 messages in context")
"""
import functools
import logging
import time

from langgraph.errors import GraphInterrupt

TAG_AGENT = "AGENT"
TAG_TOOL_CALL = "TOOL CALL"
TAG_GUARDRAIL = "GUARDRAIL STATUS"
TAG_RESPONSE = "RESPONSE"

# These frameworks each do their own verbose INFO/DEBUG logging (HTTP
# request/response bodies, retrieval internals, rail compilation, driver
# heartbeats...) that has nothing to do with what the agent decided or
# did. Turning them down to WARNING is what "filter out raw framework
# noise" actually means in practice — the alternative (parsing/hiding
# specific message strings) is brittle and breaks on every dependency
# upgrade.
_NOISY_THIRD_PARTY_LOGGERS = (
    "nemoguardrails",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "openai",
    "httpx",
    "httpcore",
    "pymongo",
    "urllib3",
    "qdrant_client",
)


class StructuredFormatter(logging.Formatter):
    """Renders app-level log records as:

        2026-08-17 10:15:33 [TOOL CALL] get_latest_order thread=user_001 -> completed in 41ms

    Falls back to a generic [LOG] tag for any record that didn't set one
    via `extra={"tag": ...}`, so nothing silently disappears if a stray
    `logger.info(...)` is added later without a tag.
    """

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", "LOG")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{tag}] {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (see app/main.py). Gives the
    "support_agent" logger tree its own tagged, structured handler, and
    turns down the third-party frameworks' own logging so their internal
    traces don't drown out what the agent itself is doing.

    Deliberately leaves propagate=True (the default) on "support_agent"
    rather than setting it False. The tempting reason to set it False is
    "don't also print through the root logger's handler" — but this app
    never attaches a handler to the root logger (uvicorn configures its
    own "uvicorn"/"uvicorn.access"/"uvicorn.error" loggers directly, not
    root), so there's nothing to double up with. Setting propagate=False
    anyway would silently break something else that expects normal
    logging propagation to reach the root logger — notably pytest's
    caplog fixture, whose capturing handler lives on the root logger.
    Found this the hard way: it broke caplog-based tests in OTHER test
    files, not just this module's own, purely from import order (any test
    that imports app.main earlier in the same pytest session triggers
    this function and leaks the propagate setting session-wide since
    Python loggers are process-global singletons).
    """
    app_logger = logging.getLogger("support_agent")
    app_logger.setLevel(level)

    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        app_logger.addHandler(handler)

    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(component: str) -> logging.Logger:
    """component is a short name like 'tools', 'nodes', 'guardrails',
    'chat' — becomes "support_agent.<component>" so configure_logging's
    handler setup on "support_agent" covers it via normal propagation."""
    return logging.getLogger(f"support_agent.{component}")


def log_agent(logger: logging.Logger, thread_id: str, message: str) -> None:
    """For orchestration-level decisions: which node ran, why the graph
    routed where it did, how many messages are in context — the "what is
    the agent doing right now" narrative, as opposed to a specific tool's
    input/output (log_tool_call) or a rail's verdict (log_guardrail)."""
    logger.info(f"thread={thread_id} — {message}", extra={"tag": TAG_AGENT})


def log_guardrail(
    logger: logging.Logger,
    thread_id: str,
    rail: str,
    status: str,
    reason: str | None = None,
) -> None:
    msg = f"thread={thread_id} rail={rail} status={status}"
    if reason:
        msg += f" reason={reason!r}"
    logger.info(msg, extra={"tag": TAG_GUARDRAIL})


def log_response(logger: logging.Logger, thread_id: str, reply: str, status: str = "ok") -> None:
    # Truncate rather than log the full reply verbatim — this is an
    # operational trace for "did a response go out and roughly what did
    # it say", not a transcript store (escalation_store.py / the
    # checkpointer already persist the real content).
    preview = reply if len(reply) <= 160 else reply[:157] + "..."
    logger.info(
        f"thread={thread_id} status={status} chars={len(reply)} reply={preview!r}",
        extra={"tag": TAG_RESPONSE},
    )


def log_tool_execution(func):
    """Decorator for the functions inside tools.py (applied BELOW @tool(),
    i.e. closer to `def`, so LangChain's schema introspection still sees
    the original signature via functools.wraps' __wrapped__ pointer).

    Logs a [TOOL CALL] line on entry and on completion, with elapsed time
    and a coarse status (see _classify_result below):
      - "completed"  — returned normally, with actual data
      - "empty"      — returned normally but found nothing (empty cart,
                        no matching order, etc.) — NOT a failure
      - "degraded"    — an internal exception was caught inside the tool's
                        own try/except and converted to a friendly
                        fallback string (see tools.py) — worth flagging
                        distinctly from "empty" in a demo log even though
                        neither one is an unhandled crash
      - "no_answer"  — check_policy found nothing in FAQ/docs; the
                        model's own instructions treat this as the signal
                        to escalate
      - "paused"     — create_support_ticket hit interrupt() and control
                        is handing off to a human; NOT an error, so this
                        re-raises langgraph's GraphInterrupt rather than
                        swallowing it
      - "failed"     — an actual exception escaped the tool (tools.py's
                        own try/except should catch nearly everything
                        before this, so this path mostly covers bugs in
                        argument handling itself, e.g. a malformed config)

    Injected params (config, tool_call_id) are excluded from the logged
    argument list — they're not something a human reading the trace needs
    to see, and user_id specifically shouldn't be echoed next to a raw
    thread id redundantly.
    """
    logger = get_logger("tools")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = kwargs.get("config")
        thread_id = "unknown"
        if isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("user_id", "unknown")

        loggable_args = {k: v for k, v in kwargs.items() if k not in ("config", "tool_call_id")}
        arg_str = ", ".join(f"{k}={v!r}" for k, v in loggable_args.items())
        logger.info(f"{func.__name__}({arg_str}) thread={thread_id} — calling", extra={"tag": TAG_TOOL_CALL})

        start = time.monotonic()
        try:
            result = func(*args, **kwargs)
        except GraphInterrupt:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                f"{func.__name__} thread={thread_id} -> paused for human input after {elapsed_ms}ms",
                extra={"tag": TAG_TOOL_CALL},
            )
            raise
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                f"{func.__name__} thread={thread_id} -> FAILED in {elapsed_ms}ms ({e})",
                extra={"tag": TAG_TOOL_CALL},
            )
            raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        status = _classify_result(result)
        logger.info(f"{func.__name__} thread={thread_id} -> {status} in {elapsed_ms}ms", extra={"tag": TAG_TOOL_CALL})
        return result

    return wrapper


def _classify_result(result) -> str:
    """Coarse status label for a tool's return value, distinguishing a
    genuine internal failure (caught inside the tool's own try/except —
    see e.g. get_user_cart in tools.py) from a merely empty/negative
    result, which isn't a problem at all — an empty cart is a completely
    normal thing for get_user_cart to report."""
    if not isinstance(result, str):
        return "completed"
    if result.startswith("Couldn't"):
        return "degraded"
    if result == "NO_ANSWER_FOUND":
        return "no_answer"
    if result.startswith("No ") or "has no orders yet" in result:
        return "empty"
    return "completed"
