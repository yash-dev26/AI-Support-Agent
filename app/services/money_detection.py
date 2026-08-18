"""
Detects whether a support agent's resolution text is money-related —
mentions a refund, credit, chargeback, or a currency amount. If so,
POST /support/resolve/{thread_id} requires an explicit confirmed=true
before it actually resumes the graph, instead of letting a single text
box submit a real refund with no second look.

Deliberately simple: a keyword list plus a currency-amount regex, not a
model call. This only needs to catch obvious cases reliably enough to put
a real confirmation step in front of an agent — it doesn't need to be a
general-purpose financial-intent classifier, and a false positive here
just costs the agent one extra click, not a wrong answer to the customer.
"""
import re

_CURRENCY_PATTERN = re.compile(
    r"[$₹€£]\s?\d[\d,]*(\.\d{1,2})?"          # $50, ₹1,499.00, € 20
    r"|\b\d[\d,]*(\.\d{1,2})?\s?(usd|inr|rs\.?|dollars|rupees)\b",
    re.IGNORECASE,
)

_MONEY_KEYWORDS = [
    "refund", "reimburse", "reimbursement", "credit back", "credited",
    "chargeback", "charge back", "waive", "waived", "compensation",
    "money back", "reversed", "reverse the charge",
]


def looks_money_related(text: str) -> bool:
    if _CURRENCY_PATTERN.search(text):
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in _MONEY_KEYWORDS)


