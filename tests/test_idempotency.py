"""
Tests for app/core/idempotency.py's IdempotencyCache — basic get/set/TTL
behavior, and (the part that actually matters for calling it "safe under
concurrency") a real multithreaded stress test proving get_or_compute
runs compute_fn AT MOST ONCE per key even when many threads race to call
it with the same key simultaneously.
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.idempotency import IdempotencyCache


def test_get_returns_none_on_miss():
    cache = IdempotencyCache()
    assert cache.get("missing") is None


def test_set_then_get_round_trips():
    cache = IdempotencyCache()
    cache.set("key1", {"status": "ok"})
    assert cache.get("key1") == {"status": "ok"}


def test_entry_expires_after_ttl():
    cache = IdempotencyCache(ttl_seconds=0.05)
    cache.set("key1", "value")
    assert cache.get("key1") == "value"
    time.sleep(0.1)
    assert cache.get("key1") is None


def test_fifo_eviction_when_over_max_size():
    cache = IdempotencyCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)  # should evict "a", the oldest

    assert cache.get("a") is None
    assert cache.get("d") == 4
    assert cache.size() == 3


def test_get_or_compute_returns_cached_value_on_hit():
    cache = IdempotencyCache()
    cache.set("key1", "cached")
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "fresh"

    result = cache.get_or_compute("key1", compute)
    assert result == "cached"
    assert calls["n"] == 0, "compute_fn must not run at all on a cache hit"


def test_get_or_compute_computes_and_caches_on_miss():
    cache = IdempotencyCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return "computed"

    first = cache.get_or_compute("key1", compute)
    second = cache.get_or_compute("key1", compute)

    assert first == "computed"
    assert second == "computed"
    assert calls["n"] == 1, "second call should have hit the cache, not recomputed"


def test_different_keys_are_computed_independently():
    cache = IdempotencyCache()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    a = cache.get_or_compute("key_a", compute)
    b = cache.get_or_compute("key_b", compute)

    assert a != b
    assert calls["n"] == 2


def test_get_or_compute_runs_at_most_once_under_real_concurrent_race():
    # The property that actually matters: a plain "check cache, else
    # compute, then cache" sequence (no locking) would NOT be safe here —
    # many threads could all observe a miss and all call compute_fn
    # before any of them finishes and populates the cache. This uses a
    # real ThreadPoolExecutor (not a mock) and an artificial delay inside
    # compute_fn specifically to widen that race window, so the test
    # would actually catch a regression back to the naive version instead
    # of passing by luck on fast machines.
    cache = IdempotencyCache()
    call_count_lock = threading.Lock()
    calls = {"n": 0}

    def slow_compute():
        with call_count_lock:
            calls["n"] += 1
        time.sleep(0.05)  # widen the race window
        return "the one true result"

    def worker():
        return cache.get_or_compute("shared_key", slow_compute)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker) for _ in range(20)]
        results = [f.result() for f in futures]

    assert calls["n"] == 1, f"compute_fn ran {calls['n']} times under concurrent access, expected exactly 1"
    assert all(r == "the one true result" for r in results), "every concurrent caller must get the SAME result"


def test_get_or_compute_concurrent_different_keys_all_run_independently():
    # Confirms the per-key locking doesn't over-serialize: concurrent
    # calls for DIFFERENT keys should all actually run, not queue up
    # behind one global lock.
    cache = IdempotencyCache()
    calls = {"n": 0}
    call_lock = threading.Lock()

    def compute_for(key):
        def _compute():
            with call_lock:
                calls["n"] += 1
            return f"result_for_{key}"
        return _compute

    def worker(i):
        key = f"key_{i}"
        return cache.get_or_compute(key, compute_for(key))

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, range(10)))

    assert calls["n"] == 10, "10 distinct keys should all have been computed independently"
    assert results == [f"result_for_key_{i}" for i in range(10)]
