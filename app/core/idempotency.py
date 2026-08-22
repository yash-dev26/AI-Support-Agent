"""
A small, generic TTL-bounded cache for idempotency keys.

Used by POST /chat/{user_id} and /chat/{user_id}/stream: a client that
retries a request after a network timeout (never seeing whether the
first attempt actually succeeded) would otherwise risk sending the SAME
user message into the graph twice — appending two HumanMessages to the
thread, burning two LLM calls, and potentially double-triggering
create_support_ticket. An idempotency_key lets the client mark "this is
a retry of the exact same logical request," and a cache hit replays the
first attempt's result instead of reprocessing it.

Separate from policy_engine.py's answer cache — same TTL/lock/eviction
shape, deliberately not shared code, because the two caches have
different eviction needs (policy answers are worth keeping around
indefinitely until a doc changes; idempotency entries are only useful
for the few seconds/minutes a client might plausibly retry in, and
MUST expire — an idempotency key is not a promise that a client can
resend the "same" logical request forever and always get the ORIGINAL
answer regardless of what changed since).
"""
import threading
import time
from typing import Any, Callable


class IdempotencyCache:
    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1024):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._store_lock = threading.Lock()
        # A separate lock PER KEY, not one global lock around compute_fn —
        # a global lock would serialize every idempotent request in the
        # whole process, even ones for completely unrelated keys/users.
        # Only concurrent callers for the SAME key should ever block on
        # each other.
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Returns the cached value, or None on a miss OR an expired
        entry (expired entries are evicted lazily on the next access
        that would have hit them, rather than needing a background
        sweep thread for a cache this small)."""
        with self._store_lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            timestamp, value = entry
            if time.monotonic() - timestamp > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._store_lock:
            if len(self._store) >= self._max_size and key not in self._store:
                # Simple FIFO eviction (dicts preserve insertion order) —
                # good enough at this size.
                self._store.pop(next(iter(self._store)))
            self._store[key] = (time.monotonic(), value)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._key_locks_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                if len(self._key_locks) >= self._max_size and key not in self._key_locks:
                    self._key_locks.pop(next(iter(self._key_locks)))
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Ensures compute_fn runs AT MOST ONCE per key, even under
        genuinely concurrent callers — not just safe against a slow
        sequential retry (call, wait, retry), but also safe against a
        client that fires a retry so fast the original request hasn't
        finished yet, which a plain "check cache, else compute, then
        cache" sequence would NOT protect against: two threads could both
        see a miss and both call compute_fn before either had a chance to
        populate the cache.

        The first caller for a given key acquires that key's lock,
        computes, and caches. Anyone else calling with the SAME key
        concurrently blocks on the same lock, then — after re-checking
        the cache, since the first caller will have populated it by the
        time the lock releases — returns that result instead of ALSO
        running compute_fn.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        lock = self._lock_for(key)
        with lock:
            # Re-check: another thread may have completed and populated
            # the cache while this one was waiting for the lock.
            cached = self.get(key)
            if cached is not None:
                return cached
            result = compute_fn()
            self.set(key, result)
            return result

    def size(self) -> int:
        with self._store_lock:
            return len(self._store)
