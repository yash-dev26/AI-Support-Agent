"""
Small curated FAQ list matched via fast keyword overlap — no embedding
call, no LLM call. Exists so obviously common questions ("how do refunds
work", "how long does shipping take") get an instant, free, deterministic
answer instead of paying for a RAG round-trip on every single message.
Anything that doesn't clear the match threshold falls through to the RAG
path in policy_engine.py.

Deliberately kept to GENERIC, non-decision questions only. Anything with
a real escalation nuance (a fraudulent charge, a damaged-on-arrival item,
a restocking fee dispute, a lost package) is intentionally left OUT of
this list, even though it's common — those need to go through RAG so the
actual policy text (and its "escalate this" instruction) gets surfaced to
the model, instead of a canned FAQ answer masking the fact that a human
needs to be involved.
"""
import re

FAQ_ENTRIES = [
    {
        "question": "what is your refund policy how do refunds work",
        "answer": "Orders can be refunded within 14 days of delivery if unopened and in "
                  "original packaging. Refunds go to the original payment method within "
                  "5-7 business days. Digital goods and gift cards are non-refundable.",
    },
    {
        "question": "how long does shipping take when will my order arrive",
        "answer": "Standard shipping typically takes 3-5 business days after your order ships. "
                  "Express shipping, where available, takes 1-2 business days.",
    },
    {
        "question": "how much does shipping cost is shipping free",
        "answer": "Shipping cost is calculated at checkout based on weight and destination. "
                  "Orders over ₹5000 qualify for free standard shipping.",
    },
    {
        "question": "do you ship internationally what countries do you ship to",
        "answer": "We currently ship to all 50 US states. We don't offer international "
                  "shipping at this time.",
    },
    {
        "question": "how do i track my order where is my order",
        "answer": "Ask me for your order status directly and I can look it up, or check "
                  "the tracking link in your shipping confirmation email.",
    },
    {
        "question": "what is your warranty period how long is the warranty",
        "answer": "Most electronics accessories carry a 1-year manufacturer warranty against "
                  "defects, starting from the delivery date. It covers manufacturing defects, "
                  "not accidental or liquid damage.",
    },
    {
        "question": "how do i exchange an item can i exchange for a different size or color",
        "answer": "We don't currently support direct one-step exchanges — return the original "
                  "item under the standard 14-day return policy, then place a new order for "
                  "the replacement.",
    },
    {
        "question": "how do i reset my password forgot password",
        "answer": "Use the \"Forgot Password\" link on the sign-in page — a reset link is sent "
                  "to your registered email and expires after 1 hour.",
    },
    {
        "question": "how do i contact support talk to a human agent",
        "answer": "If I can't resolve your issue myself, I'll connect you with a human "
                  "support agent automatically — just describe the problem you're having.",
    },
]

_MIN_OVERLAP = 2  # at least this many shared keywords to count as a confident match

# Common filler words excluded from matching — without this, a query like
# "I see a charge on my account, is this fraud?" can spuriously hit the
# _MIN_OVERLAP threshold against an unrelated FAQ entry purely through
# words like "i", "my", "is" rather than any real topic overlap. Caught by
# test_fraud_question_does_not_match_any_faq_entry actually failing before
# this existed — not a hypothetical concern.
_STOPWORDS = {
    "i", "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "to", "my", "me", "you", "your", "it", "its", "this", "that", "and",
    "or", "on", "in", "at", "of", "for", "with", "how", "what", "where",
    "when", "why", "can", "will", "would", "if",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower())) - _STOPWORDS


def match_faq(query: str) -> str | None:
    """Returns the best FAQ answer if the query confidently matches one,
    else None (meaning: fall through to RAG)."""
    query_terms = _tokenize(query)
    best_score, best_answer = 0, None
    for entry in FAQ_ENTRIES:
        score = len(query_terms & _tokenize(entry["question"]))
        if score > best_score:
            best_score, best_answer = score, entry["answer"]
    if best_score >= _MIN_OVERLAP:
        return best_answer
    return None
