"""
Tests for app/main.py's _get_checkpoint_ttl_seconds — the config parsing
for how long a thread's checkpointed state is kept before MongoDB's own
TTL index expires it. Pure function, no live Mongo needed (the actual
TTL index creation happens inside MongoDBSaver, a third-party library
already trusted to do what its own docs/source say — see the function's
docstring for what was actually verified there).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.main import _get_checkpoint_ttl_seconds, DEFAULT_CHECKPOINT_TTL_SECONDS


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("CHECKPOINT_TTL_SECONDS", raising=False)
    yield


def test_defaults_to_30_days_when_unset():
    assert _get_checkpoint_ttl_seconds() == DEFAULT_CHECKPOINT_TTL_SECONDS
    assert DEFAULT_CHECKPOINT_TTL_SECONDS == 60 * 60 * 24 * 30


def test_explicit_positive_value_is_used(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_TTL_SECONDS", "86400")
    assert _get_checkpoint_ttl_seconds() == 86400


def test_zero_disables_expiry(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_TTL_SECONDS", "0")
    assert _get_checkpoint_ttl_seconds() is None


def test_empty_string_disables_expiry(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_TTL_SECONDS", "")
    assert _get_checkpoint_ttl_seconds() is None


def test_negative_value_disables_expiry_rather_than_erroring(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_TTL_SECONDS", "-5")
    assert _get_checkpoint_ttl_seconds() is None


def test_malformed_value_falls_back_to_no_expiry_not_a_crash(monkeypatch):
    # A typo in an env var should never cause silent data loss — falling
    # back to "keep everything" (the safe direction) rather than raising
    # or guessing a number.
    monkeypatch.setenv("CHECKPOINT_TTL_SECONDS", "thirty days")
    assert _get_checkpoint_ttl_seconds() is None


# --- _log_langsmith_tracing_status --------------------------------------

from app.main import _log_langsmith_tracing_status


@pytest.fixture(autouse=True)
def clean_langsmith_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    yield


def test_logs_enabled_when_tracing_and_key_both_set(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")

    with caplog.at_level("INFO", logger="support_agent.startup"):
        _log_langsmith_tracing_status()

    assert any("ENABLED" in r.getMessage() and "my-project" in r.getMessage() for r in caplog.records)


def test_logs_disabled_when_tracing_unset(caplog):
    with caplog.at_level("INFO", logger="support_agent.startup"):
        _log_langsmith_tracing_status()

    assert any("disabled" in r.getMessage() for r in caplog.records)


def test_warns_when_tracing_true_but_key_missing(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    # deliberately no LANGSMITH_API_KEY

    with caplog.at_level("WARNING", logger="support_agent.startup"):
        _log_langsmith_tracing_status()

    assert any(r.levelname == "WARNING" and "silently no-op" in r.getMessage() for r in caplog.records)


def test_tracing_value_is_case_insensitive(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "TRUE")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")

    with caplog.at_level("INFO", logger="support_agent.startup"):
        _log_langsmith_tracing_status()

    assert any("ENABLED" in r.getMessage() for r in caplog.records)
