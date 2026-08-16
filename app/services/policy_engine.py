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
"""
from app.services import faq, vector_store

NO_ANSWER_SIGNAL = (
    "NO_ANSWER_FOUND: No FAQ or documented policy addresses this question. "
    "Escalate to a human now via human_interrupt_tool."
)

_SCORE_FLOOR = 0.35  # below this, treat retrieval as "found nothing relevant" —
                       # a starting heuristic, tune against real docs/queries


def answer(query: str) -> str:
    faq_answer = faq.match_faq(query)
    if faq_answer:
        return f"[FAQ] {faq_answer}"

    try:
        hits = vector_store.search(query, top_k=3)
    except Exception as e:
        return (
            f"NO_ANSWER_FOUND: retrieval unavailable ({e}). "
            "Escalate to a human now via human_interrupt_tool."
        )

    relevant = [h for h in hits if h["score"] >= _SCORE_FLOOR]
    if not relevant:
        return NO_ANSWER_SIGNAL

    return _generate_cited_answer(query, relevant)


def _generate_cited_answer(query: str, hits: list[dict]) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain.chat_models import init_chat_model

    context = "\n\n".join(f"[{h['doc_name']}] {h['text']}" for h in hits)
    system = (
        "Answer the question using ONLY the provided context. Cite the source "
        "document name(s) in brackets, e.g. [refund_policy.md]. If the context "
        "does not actually answer the question, respond with exactly: NO_ANSWER_FOUND"
    )
    llm = init_chat_model(model_provider="openai", model="gpt-4.1")
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    text = (response.content or "").strip()
    if "NO_ANSWER_FOUND" in text:
        return NO_ANSWER_SIGNAL
    return f"[RAG] {text}"
