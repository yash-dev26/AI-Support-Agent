"""
Rate limiting via slowapi (in-memory backend — fine for a single-process
deployment; swap for a Redis-backed limits storage string if this ever
runs as multiple instances, since in-memory buckets don't share state
across processes).

Keyed by user_id where the route has one (chat/support endpoints), not by
IP — this is a multi-tenant app where many legitimate users could share
an IP (e.g. behind a NAT or in this sandboxed demo), and the thing worth
protecting against is one user hammering the LLM, not "too many requests
from one network."
"""
from fastapi import Request
from slowapi import Limiter


def _rate_limit_key(request: Request) -> str:
    user_id = request.path_params.get("user_id") or request.path_params.get("thread_id")
    if user_id:
        return f"user:{user_id}"
    # fall back to remote address for routes with no user/thread in the path
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_rate_limit_key)


