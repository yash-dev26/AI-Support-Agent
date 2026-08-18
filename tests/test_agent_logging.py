"""
Tests for app/core/agent_logging.py: the StructuredFormatter's output
shape, log_tool_execution's status classification (completed / empty /
degraded / no_answer / failed / paused), and that it correctly
re-raises GraphInterrupt rather than treating an escalation as a
failure.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langgraph.errors import GraphInterrupt

from app.core.agent_logging import (
    StructuredFormatter,
    _classify_result,
    log_agent,
    log_guardrail,
    log_response,
    log_tool_execution,
)


def _make_record(msg: str, tag: str | None = None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="support_agent.test", level=logging.INFO, pathname=__file__,
        lineno=1, msg=msg, args=(), exc_info=None,
    )
    if tag is not None:
        record.tag = tag
    return record


def test_formatter_renders_bracketed_tag():
    line = StructuredFormatter().format(_make_record("hello world", tag="TOOL CALL"))
    assert "[TOOL CALL] hello world" in line


def test_formatter_falls_back_to_log_tag_when_none_set():
    line = StructuredFormatter().format(_make_record("untagged line"))
    assert "[LOG] untagged line" in line


def test_formatter_includes_a_timestamp_prefix():
    line = StructuredFormatter().format(_make_record("msg", tag="AGENT"))
    # "YYYY-MM-DD HH:MM:SS [AGENT] msg" — just check it's not glued to the tag
    assert line.split(" [AGENT]")[0].count("-") == 2


# --- _classify_result ---------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("- USB-C Hub x1", "completed"),
    ("No items in cart.", "empty"),
    ("No order history.", "empty"),
    ("No order found with id ord_123.", "empty"),
    ("This user has no orders yet.", "empty"),
    ("Couldn't load the cart right now — try again in a moment.", "degraded"),
    ("NO_ANSWER_FOUND", "no_answer"),
])
def test_classify_result_string_cases(value, expected):
    assert _classify_result(value) == expected


def test_classify_result_defaults_to_completed_for_non_string():
    assert _classify_result(None) == "completed"
    assert _classify_result(42) == "completed"


# --- log_tool_execution decorator ---------------------------------------

def test_decorator_logs_calling_then_completed(caplog):
    @log_tool_execution
    def sample(config=None):
        return "some real data"

    with caplog.at_level("INFO", logger="support_agent.tools"):
        result = sample(config={"configurable": {"user_id": "user_001"}})

    assert result == "some real data"
    tags_and_msgs = [(getattr(r, "tag", None), r.getMessage()) for r in caplog.records]
    assert any(tag == "TOOL CALL" and "calling" in msg for tag, msg in tags_and_msgs)
    assert any(tag == "TOOL CALL" and "completed" in msg for tag, msg in tags_and_msgs)
    assert all("user_001" in msg for _, msg in tags_and_msgs)


def test_decorator_classifies_empty_result_distinctly_from_completed(caplog):
    @log_tool_execution
    def sample(config=None):
        return "No items in cart."

    with caplog.at_level("INFO", logger="support_agent.tools"):
        sample(config={"configurable": {"user_id": "user_002"}})

    assert any("-> empty" in r.getMessage() for r in caplog.records)


def test_decorator_logs_failed_and_reraises_on_exception(caplog):
    @log_tool_execution
    def sample(config=None):
        raise ValueError("boom")

    with caplog.at_level("INFO", logger="support_agent.tools"):
        with pytest.raises(ValueError, match="boom"):
            sample(config={"configurable": {"user_id": "user_003"}})

    assert any(
        getattr(r, "tag", None) == "TOOL CALL" and "FAILED" in r.getMessage()
        for r in caplog.records
    )


def test_decorator_logs_paused_and_reraises_graph_interrupt(caplog):
    @log_tool_execution
    def sample(config=None):
        raise GraphInterrupt([{"value": "waiting on a human"}])

    with caplog.at_level("INFO", logger="support_agent.tools"):
        with pytest.raises(GraphInterrupt):
            sample(config={"configurable": {"user_id": "user_004"}})

    assert any(
        getattr(r, "tag", None) == "TOOL CALL" and "paused for human input" in r.getMessage()
        for r in caplog.records
    )


def test_decorator_excludes_config_and_tool_call_id_from_logged_args(caplog):
    @log_tool_execution
    def sample(issue_type=None, config=None, tool_call_id=None):
        return "ok"

    with caplog.at_level("INFO", logger="support_agent.tools"):
        sample(
            issue_type="missing_order",
            config={"configurable": {"user_id": "user_005"}},
            tool_call_id="call_abc123",
        )

    calling_line = next(r.getMessage() for r in caplog.records if "calling" in r.getMessage())
    assert "issue_type='missing_order'" in calling_line
    assert "call_abc123" not in calling_line
    assert "configurable" not in calling_line


def test_decorator_preserves_function_signature_for_schema_introspection():
    import inspect

    @log_tool_execution
    def sample(x: str, y: int = 3):
        return x

    sig = inspect.signature(sample)
    assert list(sig.parameters) == ["x", "y"]


# --- log_agent / log_guardrail / log_response --------------------------

def test_log_agent_includes_thread_id_and_tag(caplog):
    logger = logging.getLogger("support_agent.test_helpers")
    with caplog.at_level("INFO", logger="support_agent.test_helpers"):
        log_agent(logger, "user_010", "invoking LLM")
    record = caplog.records[-1]
    assert record.tag == "AGENT"
    assert "user_010" in record.getMessage()
    assert "invoking LLM" in record.getMessage()


def test_log_guardrail_includes_rail_status_and_optional_reason(caplog):
    logger = logging.getLogger("support_agent.test_helpers")
    with caplog.at_level("INFO", logger="support_agent.test_helpers"):
        log_guardrail(logger, "user_011", "self_check_input", "blocked", reason="prompt injection detected")
    record = caplog.records[-1]
    assert record.tag == "GUARDRAIL STATUS"
    msg = record.getMessage()
    assert "rail=self_check_input" in msg
    assert "status=blocked" in msg
    assert "prompt injection detected" in msg


def test_log_response_truncates_long_replies(caplog):
    logger = logging.getLogger("support_agent.test_helpers")
    long_reply = "x" * 500
    with caplog.at_level("INFO", logger="support_agent.test_helpers"):
        log_response(logger, "user_012", long_reply, status="ok")
    record = caplog.records[-1]
    assert record.tag == "RESPONSE"
    assert "chars=500" in record.getMessage()
    # truncated preview should be much shorter than the full 500-char reply
    assert len(record.getMessage()) < 300
