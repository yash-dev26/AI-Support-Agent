"""
Tests for policy_engine.answer() — the routing decision between FAQ,
RAG, and the NO_ANSWER_FOUND escalation signal. Monkeypatches faq.match_faq,
vector_store.search, and the LLM generation step, so this runs without
live credentials in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import policy_engine


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
    assert "human_interrupt_tool" in result


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
    assert "human_interrupt_tool" in result


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
