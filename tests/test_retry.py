"""
Tests for app/core/retry.py's with_retry decorator. Monkeypatches
time.sleep to a no-op so these run in milliseconds regardless of the
configured backoff delays — what's being tested is the retry LOGIC
(attempt counting, which exceptions are retried, eventual success vs.
eventual failure), not real wall-clock timing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
import pytest

from app.core.retry import with_retry, RETRYABLE_OPENAI_ERRORS


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    monkeypatch.setattr("app.core.retry.time.sleep", lambda seconds: None)


def _fake_rate_limit_error():
    return openai.RateLimitError(
        message="rate limited", response=_fake_response(429), body=None
    )


def _fake_connection_error():
    return openai.APIConnectionError(request=_fake_request())


def _fake_response(status_code):
    import httpx
    return httpx.Response(status_code=status_code, request=_fake_request())


def _fake_request():
    import httpx
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def test_succeeds_on_first_attempt_without_retrying():
    calls = {"n": 0}

    @with_retry()
    def flaky():
        calls["n"] += 1
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 1


def test_retries_a_retryable_error_and_eventually_succeeds():
    calls = {"n": 0}

    @with_retry(max_attempts=3)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _fake_rate_limit_error()
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 3


def test_gives_up_and_reraises_after_max_attempts():
    calls = {"n": 0}

    @with_retry(max_attempts=3)
    def always_fails():
        calls["n"] += 1
        raise _fake_connection_error()

    with pytest.raises(openai.APIConnectionError):
        always_fails()
    assert calls["n"] == 3, "should have tried exactly max_attempts times, not more or fewer"


def test_non_retryable_error_propagates_immediately_without_retrying():
    calls = {"n": 0}

    @with_retry(max_attempts=5)
    def bad_request():
        calls["n"] += 1
        raise openai.BadRequestError(
            message="bad request", response=_fake_response(400), body=None,
        )

    with pytest.raises(openai.BadRequestError):
        bad_request()
    assert calls["n"] == 1, "a non-retryable error must fail fast, not consume retry attempts"


def test_default_retryable_set_includes_connection_rate_limit_and_server_errors():
    assert openai.APIConnectionError in RETRYABLE_OPENAI_ERRORS
    assert openai.RateLimitError in RETRYABLE_OPENAI_ERRORS
    assert openai.InternalServerError in RETRYABLE_OPENAI_ERRORS


def test_preserves_function_name_and_docstring():
    @with_retry()
    def my_documented_function():
        """A docstring."""
        return None

    assert my_documented_function.__name__ == "my_documented_function"
    assert my_documented_function.__doc__ == "A docstring."


def test_custom_retryable_tuple_is_respected():
    calls = {"n": 0}

    @with_retry(max_attempts=2, retryable=(ValueError,))
    def only_retries_value_errors():
        calls["n"] += 1
        raise TypeError("not in the retryable set")

    with pytest.raises(TypeError):
        only_retries_value_errors()
    assert calls["n"] == 1
