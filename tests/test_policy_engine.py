"""
Tests for policy_engine.answer() — the routing decision between FAQ,
RAG, and the NO_ANSWER_FOUND escalation signal. Monkeypatches faq.match_faq,
vector_store.search, and the LLM generation step, so this runs without
live credentials in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.services import policy_engine


@pytest.fixture(autouse=True)
def clear_answer_cache():
    """_cached_answer is an lru_cache — a module-level, process-global
    cache that would otherwise leak a result computed under one test's
    monkeypatched vector_store.search/​_generate_cited_answer into a LATER
    test that queries the same text with different monkeypatches. Every
    test in this file gets a clean cache before AND after it runs."""
    policy_engine.clear_cache()
    yield
    policy_engine.clear_cache()


def test_faq_hit_short_circuits_before_vector_search(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: "Refunds take 5-7 days.")

    def fail_if_called(*a, **kw):
        raise AssertionError("vector_store.search should not be called on an FAQ hit")

    monkeypatch.setattr(policy_engine.vector_store, "search", fail_if_called)

    result = policy_engine.answer("how do refunds work")
    assert result == "[FAQ] Refunds take 5-7 days."


def test_no_faq_hit_falls_through_to_rag_with_relevant_hits(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(
        policy_engine.vector_store, "search",
        lambda q, top_k=3: [{"doc_name": "refund_policy.md", "text": "Refunds within 14 days.", "score": 0.9}],
    )
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] Refunds within 14 days. [refund_policy.md]")

    result = policy_engine.answer("what's the refund window")
    assert result.startswith("[RAG]")
    assert "refund_policy.md" in result


def test_no_faq_hit_and_no_relevant_vector_hits_returns_no_answer_signal(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(policy_engine.vector_store, "search", lambda q, top_k=3: [])

    result = policy_engine.answer("do you ship to the moon")
    assert result.startswith("NO_ANSWER_FOUND")
    assert "create_support_ticket" in result


def test_low_score_vector_hits_are_filtered_out_below_floor(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(
        policy_engine.vector_store, "search",
        lambda q, top_k=3: [{"doc_name": "x.md", "text": "irrelevant", "score": 0.05}],
    )

    def fail_if_called(*a, **kw):
        raise AssertionError("_generate_cited_answer should not be called when nothing clears the score floor")

    monkeypatch.setattr(policy_engine, "_generate_cited_answer", fail_if_called)

    result = policy_engine.answer("something totally unrelated")
    assert result.startswith("NO_ANSWER_FOUND")


def test_vector_search_failure_degrades_to_no_answer_signal_not_a_crash(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)

    def broken_search(q, top_k=3):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(policy_engine.vector_store, "search", broken_search)

    result = policy_engine.answer("anything")
    assert result.startswith("NO_ANSWER_FOUND")
    assert "create_support_ticket" in result


def test_generation_returning_no_answer_found_is_normalized_to_the_standard_signal(monkeypatch):
    # if the LLM itself says it can't answer from the retrieved context,
    # that should come back as the same NO_ANSWER_FOUND signal, not a
    # partial/garbled [RAG] response
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(
        policy_engine.vector_store, "search",
        lambda q, top_k=3: [{"doc_name": "x.md", "text": "unrelated content", "score": 0.9}],
    )
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: policy_engine.NO_ANSWER_SIGNAL)

    result = policy_engine.answer("something the docs don't actually cover")
    assert result == policy_engine.NO_ANSWER_SIGNAL


# --- caching ---------------------------------------------------------

def test_repeated_rag_query_hits_the_cache_not_vector_search_again(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    call_count = {"n": 0}

    def counting_search(q, top_k=3):
        call_count["n"] += 1
        return [{"doc_name": "warranty.md", "text": "1 year warranty.", "score": 0.9}]

    monkeypatch.setattr(policy_engine.vector_store, "search", counting_search)
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] 1 year warranty. [warranty.md]")

    first = policy_engine.answer("how long is the warranty")
    second = policy_engine.answer("how long is the warranty")

    assert first == second
    assert call_count["n"] == 1, "second call should have been served from cache, not re-run vector_store.search"


def test_cache_key_is_normalized_for_case_and_whitespace(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    call_count = {"n": 0}

    def counting_search(q, top_k=3):
        call_count["n"] += 1
        return [{"doc_name": "warranty.md", "text": "1 year warranty.", "score": 0.9}]

    monkeypatch.setattr(policy_engine.vector_store, "search", counting_search)
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] answer")

    policy_engine.answer("  How Long Is The Warranty   ")
    policy_engine.answer("how long is the warranty")

    assert call_count["n"] == 1, "differently-cased/whitespaced same question should share one cache entry"


def test_different_queries_are_cached_separately(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(
        policy_engine.vector_store, "search",
        lambda q, top_k=3: [{"doc_name": "x.md", "text": "content", "score": 0.9}],
    )
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: f"[RAG] answer for {q}")

    a = policy_engine.answer("question one")
    b = policy_engine.answer("question two")
    assert a != b


def test_retrieval_failure_is_not_cached(monkeypatch):
    # a transient outage shouldn't get baked in as this query's permanent
    # answer — the next call, once the backend recovers, should try again
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    call_count = {"n": 0}

    def flaky_search(q, top_k=3):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("qdrant unavailable")
        return [{"doc_name": "x.md", "text": "content", "score": 0.9}]

    monkeypatch.setattr(policy_engine.vector_store, "search", flaky_search)
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] recovered answer")

    first = policy_engine.answer("is the service up")
    second = policy_engine.answer("is the service up")

    assert first.startswith("NO_ANSWER_FOUND")
    assert second == "[RAG] recovered answer"
    assert call_count["n"] == 2, "the failed first attempt must not have been cached"


def test_clear_cache_forces_recomputation(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    call_count = {"n": 0}

    def counting_search(q, top_k=3):
        call_count["n"] += 1
        return [{"doc_name": "x.md", "text": "content", "score": 0.9}]

    monkeypatch.setattr(policy_engine.vector_store, "search", counting_search)
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] answer")

    policy_engine.answer("some cached question")
    policy_engine.clear_cache()
    policy_engine.answer("some cached question")

    assert call_count["n"] == 2, "clear_cache() (called after a new doc upload) should force a fresh lookup"


def test_faq_hits_are_not_routed_through_the_cache_at_all(monkeypatch):
    # an FAQ hit never touches _cached_answer, so editing FAQ_ENTRIES
    # takes effect immediately without a stale cache entry masking it
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: "first answer")
    first = policy_engine.answer("what's your refund policy")

    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: "updated answer")
    second = policy_engine.answer("what's your refund policy")

    assert first == "[FAQ] first answer"
    assert second == "[FAQ] updated answer"


def test_cache_info_reports_hits_misses_and_size(monkeypatch):
    monkeypatch.setattr(policy_engine.faq, "match_faq", lambda q: None)
    monkeypatch.setattr(
        policy_engine.vector_store, "search",
        lambda q, top_k=3: [{"doc_name": "x.md", "text": "content", "score": 0.9}],
    )
    monkeypatch.setattr(policy_engine, "_generate_cited_answer", lambda q, hits: "[RAG] answer")

    policy_engine.answer("question a")  # miss
    policy_engine.answer("question a")  # hit
    policy_engine.answer("question b")  # miss

    info = policy_engine.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 2
    assert info["size"] == 2


