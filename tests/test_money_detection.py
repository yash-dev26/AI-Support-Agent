import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import money_detection


def test_refund_keyword_detected():
    assert money_detection.looks_money_related("Refund issued for the duplicate charge.")


def test_currency_amount_detected():
    assert money_detection.looks_money_related("We've credited $49.99 back to your card.")


def test_rupee_symbol_detected():
    assert money_detection.looks_money_related("Refund of ₹1,499 processed.")


def test_chargeback_detected():
    assert money_detection.looks_money_related("This will be handled as a chargeback.")


def test_waived_fee_detected():
    assert money_detection.looks_money_related("We've waived the restocking fee for this order.")


def test_non_money_resolution_not_flagged():
    assert not money_detection.looks_money_related(
        "Your package was found and redelivered this morning, no further action needed."
    )


def test_order_number_alone_not_flagged_as_currency():
    # order numbers/ids shouldn't trip the currency regex just for having digits
    assert not money_detection.looks_money_related("Your order ORD-12345 has shipped.")


def test_case_insensitive_keyword_match():
    assert money_detection.looks_money_related("REFUND has been processed.")


def test_dollar_amount_without_symbol_but_with_word_detected():
    assert money_detection.looks_money_related("Issued a credit of 50 USD to your account.")
