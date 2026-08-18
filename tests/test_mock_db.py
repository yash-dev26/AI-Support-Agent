"""
Unit tests for the mock commerce DB. No server, no live Mongo/OpenAI creds
needed — these run in CI on every push.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.services import mock_db


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    """Point the DB at a throwaway file per test so tests don't share state."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_commerce.db")
    mock_db.seed(num_users=5, force=True)
    yield


def test_seed_creates_expected_number_of_users():
    with mock_db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    assert count == 5


def test_every_seeded_user_has_at_least_one_order():
    with mock_db.get_conn() as conn:
        user_ids = [r["user_id"] for r in conn.execute("SELECT user_id FROM users").fetchall()]
    for user_id in user_ids:
        assert len(mock_db.get_order_history(user_id)) >= 1


def test_get_cart_returns_empty_list_for_unknown_user():
    assert mock_db.get_cart("nonexistent_user") == []


def test_get_order_status_returns_none_for_unknown_order():
    assert mock_db.get_order_status("nonexistent_order") is None


def test_get_order_status_matches_get_order_history():
    with mock_db.get_conn() as conn:
        user_id = conn.execute("SELECT user_id FROM users LIMIT 1").fetchone()["user_id"]
    orders = mock_db.get_order_history(user_id)
    assert orders, "expected at least one seeded order"
    first_order = orders[0]
    status = mock_db.get_order_status(first_order["order_id"])
    assert status is not None
    assert status["status"] == first_order["status"]


def test_reseed_with_force_replaces_existing_data():
    mock_db.seed(num_users=3, force=True)
    with mock_db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    assert count == 3


def test_reseed_without_force_is_a_noop():
    with mock_db.get_conn() as conn:
        before = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    mock_db.seed(num_users=999, force=False)
    with mock_db.get_conn() as conn:
        after = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    assert before == after


def test_user_001_has_a_scripted_duplicate_charge():
    orders = mock_db.get_order_history("user_001")
    assert len(orders) == 2
    assert orders[0]["product_name"] == orders[1]["product_name"] == "Mechanical Keyboard"
    assert orders[0]["amount_cents"] == orders[1]["amount_cents"] == 349900


def test_user_002_has_an_order_outside_the_refund_window():
    orders = mock_db.get_order_history("user_002")
    assert len(orders) == 1
    assert orders[0]["status"] == "delivered"
    assert orders[0]["product_name"] == "Noise Cancelling Headphones"


def test_user_003_has_a_stuck_processing_order():
    orders = mock_db.get_order_history("user_003")
    assert len(orders) == 1
    assert orders[0]["status"] == "processing"
    assert orders[0]["product_name"] == "27-inch Monitor"


def test_scripted_users_are_skipped_when_num_users_below_three():
    mock_db.seed(num_users=2, force=True)
    with mock_db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    assert count == 2
    # with no scripted scenarios, user_001 should NOT have the duplicate charge
    orders = mock_db.get_order_history("user_001")
    assert not (len(orders) == 2 and orders[0]["product_name"] == "Mechanical Keyboard" == orders[1]["product_name"])


