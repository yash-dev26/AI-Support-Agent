"""
Retry/backoff decorator for external API calls that can fail transiently
— rate limits, connection resets, upstream 5xxs — used around the LLM
calls in nodes.py's chatbot() and policy_engine.py's RAG generation.

Hand-rolled rather than pulling in `tenacity`: it isn't in
requirements.txt, and a bounded retry loop with exponential backoff is
about 20 lines of stdlib code — not enough complexity to justify a new
dependency and its own supply-chain surface for this codebase.
"""
import functools
import logging
import random
import time

import openai

logger = logging.getLogger("support_agent.retry")

# Only these are worth retrying. A connection reset, a rate limit, or an
# upstream 5xx (InternalServerError) can plausibly succeed a moment
# later with no other change. AuthenticationError, BadRequestError (a
# malformed request), PermissionDeniedError, NotFoundError etc. will
# fail again identically on retry — retrying those just delays reporting
# a failure that was never going to succeed, so they're deliberately
# excluded rather than caught by a blanket `except openai.OpenAIError`.
RETRYABLE_OPENAI_ERRORS = (
    openai.APIConnectionError,   # includes APITimeoutError (subclass)
    openai.RateLimitError,
    openai.InternalServerError,
)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: tuple = RETRYABLE_OPENAI_ERRORS,
):
    """Wraps a function to retry on the given exception types with
    exponential backoff plus jitter. NEVER swallows a failure silently —
    if every attempt fails, the last exception propagates exactly as it
    would without this decorator; retrying only delays reporting a
    failure in exchange for a chance to recover from a transient blip.

    Jitter (a randomized delay within the backoff window, not a fixed
    one) matters at real scale, not just in theory: without it, many
    concurrent requests that all failed at the same moment — e.g. a
    brief upstream blip hitting every in-flight call at once — would all
    retry at EXACTLY the same moment too, turning a brief blip into a
    synchronized retry storm that can look like a self-inflicted DDoS on
    the upstream service just as it's starting to recover.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retryable as e:
                    if attempt >= max_attempts:
                        logger.warning(
                            "%s failed after %d attempt(s), giving up: %s",
                            func.__name__, attempt, e,
                        )
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.25)
                    logger.info(
                        "%s failed (attempt %d/%d): %s -- retrying in %.2fs",
                        func.__name__, attempt, max_attempts, e, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
