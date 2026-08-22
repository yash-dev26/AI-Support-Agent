"""
Shared pytest fixtures, autoused across the whole test session.

Currently just one: resetting slowapi's rate limiter state between every
test. `app/core/rate_limit.py`'s `limiter` is a module-level singleton —
imported by reference into chat.py and support.py — so its in-memory
bucket state is shared across EVERY test in the same pytest process,
regardless of which `FastAPI()` app instance a given test builds or which
test file it lives in. Without this fixture, any test hitting
`/chat/user_001` contributes to the exact same cumulative 15/minute
bucket as every other test that happens to reuse "user_001" as its test
subject — which is most of them. Enough tests accumulating enough calls
against the same handful of conventional test user_ids will eventually
trip the real 429 rate limit and fail tests that have nothing to do with
rate limiting at all, purely based on how many tests happened to run
before them in that process. `tests/test_rate_limit.py` worked around
this for ITS OWN tests by using distinctly-scoped ids (rl_user_a,
rl_user_b, ...) — this fixture fixes it properly, for every test, so new
tests don't each need to remember that same workaround.
"""
import pytest

from app.core.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter._storage.reset()
    yield
