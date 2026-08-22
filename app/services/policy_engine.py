"""
The intent router check_policy actually uses:

  1. FAQ keyword match (faq.py) — fast, free, deterministic, for the small
     set of genuinely common questions.
  2. RAG over uploaded docs (vector_store.py + an LLM call that must
     answer only from retrieved context, with citations) — for anything
     that doesn't hit an FAQ.
  3. If neither produces a confident answer, returns a clear NO_ANSWER_FOUND
     signal. nodes.py's SYSTEM_PROMPT is written to treat that signal —
     not the user simply asking for a human — as the actual trigger to
     escalate. That's what makes escalation a last resort rather than
     something that happens on request: the model is instructed to try
     check_policy first, and only escalate once it gets this signal back.

Cached (see _cached_answer): the RAG path is one embedding call plus one
LLM call per query, and check_policy is called on essentially every
non-trivial support message per the system prompt's "always call this
before escalating" instruction — so the same handful of common
non-FAQ questions ("what's your warranty on X", "can I return a Y")
get asked constantly across different users/threads. An exact-match
in-memory cache turns every repeat of the SAME question into a dict
lookup instead of a network round trip. Cleared automatically on doc
upload (see clear_cache / docs.py) so a newly-indexed doc's answer to a
question that previously came back NO_ANSWER_FOUND isn't served stale.
"""
import threading

from app.services import faq, vector_store

NO_ANSWER_SIGNAL = (
    "NO_ANSWER_FOUND: No FAQ or documented policy addresses this question. "
    "Escalate to a human now via create_support_ticket."
)

_SCORE_FLOOR = 0.35  # below this, treat retrieval as "found nothing relevant" —
                       # a starting heuristic, tune against real docs/queries

_CACHE_SIZE = 256
_answer_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0


def answer(query: str) -> str:
    faq_answer = faq.match_faq(query)
    if faq_answer:
        # Not cached — a dict/set lookup over ~10 entries is already
        # sub-millisecond, and skipping the cache here means match_faq's
        # own logic (which can change if FAQ_ENTRIES is edited) never goes
        # stale behind a cached miss.
        return f"[FAQ] {faq_answer}"
    return _cached_answer(_normalize(query))


def _normalize(query: str) -> str:
    """Exact-match cache key, not semantic — "What's your refund policy?"
    and "what's your refund policy" hit the same entry, but "how do
    refunds work" does not. That's a deliberate, conservative tradeoff:
    a semantic/embedding-based cache key risks serving a WRONG cached
    answer for a similar-but-different question, which is worse than the
    lower hit rate of an exact match."""
    return " ".join(query.strip().lower().split())


def _cached_answer(normalized_query: str) -> str:
    """Deliberately a hand-rolled cache rather than functools.lru_cache:
    lru_cache would cache ANY return value, including the retrieval
    -unavailable fallback string built in the except block below — that's
    wrong, a transient outage shouldn't become this query's PERMANENT
    answer for the rest of the process's lifetime. Only a genuine result
    (a real RAG answer, or a legitimate "nothing relevant indexed"
    NO_ANSWER_FOUND) gets cached; a caught exception does not.
    """
    global _cache_hits, _cache_misses
    with _cache_lock:
        cached = _answer_cache.get(normalized_query)
    if cached is not None:
        with _cache_lock:
            _cache_hits += 1
        return cached

    with _cache_lock:
        _cache_misses += 1

    try:
        hits = vector_store.search(normalized_query, top_k=3)
    except Exception as e:
        # NOT cached — see docstring above.
        return (
            f"NO_ANSWER_FOUND: retrieval unavailable ({e}). "
            "Escalate to a human now via create_support_ticket."
        )

    relevant = [h for h in hits if h["score"] >= _SCORE_FLOOR]
    result = NO_ANSWER_SIGNAL if not relevant else _generate_cited_answer(normalized_query, relevant)

    with _cache_lock:
        if len(_answer_cache) >= _CACHE_SIZE:
            # Simple FIFO eviction (Python dicts preserve insertion
            # order) rather than true LRU — good enough at this cache
            # size; avoids pulling in a dependency or hand-rolling an
            # ordered-move-to-end LRU for a demo-scale cache.
            _answer_cache.pop(next(iter(_answer_cache)))
        _answer_cache[normalized_query] = result
    return result


def clear_cache() -> None:
    """Called after a new doc is indexed (see docs.py's /docs/upload) so a
    question that previously came back NO_ANSWER_FOUND — or was answered
    from now-outdated context — gets re-evaluated against the updated
    index instead of serving a stale cached response for the rest of the
    process's lifetime."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _answer_cache.clear()
        _cache_hits = 0
        _cache_misses = 0


def cache_info() -> dict:
    """Exposed for /metrics or manual debugging — hits/misses/current size,
    not user-facing."""
    with _cache_lock:
        return {"hits": _cache_hits, "misses": _cache_misses, "size": len(_answer_cache), "max_size": _CACHE_SIZE}


def _generate_cited_answer(query: str, hits: list[dict]) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain.chat_models import init_chat_model

    from app.core.retry import with_retry

    context = "\n\n".join(f"[{h['doc_name']}] {h['text']}" for h in hits)
    system = (
        "Answer the question using ONLY the provided context. Cite the source "
        "document name(s) in brackets, e.g. [refund_policy.md]. If the context "
        "does not actually answer the question, respond with exactly: NO_ANSWER_FOUND"
    )
    llm = init_chat_model(model_provider="openai", model="gpt-4.1")

    @with_retry()
    def _invoke():
        return llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ])

    response = _invoke()
    text = (response.content or "").strip()
    if "NO_ANSWER_FOUND" in text:
        return NO_ANSWER_SIGNAL
    return f"[RAG] {text}"


