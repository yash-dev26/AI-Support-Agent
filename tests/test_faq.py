import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import faq


def test_confident_match_returns_answer():
    result = faq.match_faq("how do refunds work")
    assert result is not None
    assert "14 days" in result


def test_no_match_returns_none():
    result = faq.match_faq("zzz completely unrelated xyz query about spaceships")
    assert result is None


def test_weak_single_keyword_overlap_does_not_count_as_a_match():
    # shares only one word ("order") with an FAQ entry — below the
    # confidence threshold, should fall through to RAG rather than
    # confidently answering off a single word
    result = faq.match_faq("order")
    assert result is None


def test_shipping_question_matches_shipping_entry_not_refund_entry():
    result = faq.match_faq("how long does shipping take")
    assert result is not None
    assert "3-5 business days" in result


def test_warranty_question_matches_warranty_entry():
    result = faq.match_faq("what is your warranty period")
    assert result is not None
    assert "1-year" in result


def test_password_reset_question_matches():
    result = faq.match_faq("how do i reset my password")
    assert result is not None
    assert "Forgot Password" in result


def test_fraud_question_does_not_match_any_faq_entry():
    # a fraud/unauthorized-charge question must NOT get a canned FAQ
    # answer — it needs to fall through to RAG so the real policy text
    # (which says to escalate) actually gets surfaced
    result = faq.match_faq("I see a charge on my account I don't recognize, is this fraud?")
    assert result is None


def test_damaged_on_arrival_question_does_not_match_any_faq_entry():
    result = faq.match_faq("my item arrived broken and damaged out of the box")
    assert result is None


def test_account_closure_question_does_not_match_any_faq_entry():
    result = faq.match_faq("I want to permanently delete and close my account")
    assert result is None


