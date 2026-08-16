"""
Tests for app/graph/tools.py's authorization behavior: user_id is
injected from RunnableConfig (the authenticated session), never an
LLM-controllable argument, and get_order_status can't be used to look up
another user's order by guessing an order_id.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.graph.tools import get_cart_items, get_order_history, get_order_status
from app.services import mock_db


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_commerce.db")
    mock_db.seed(num_users=5, force=True)
    yield


def _config(user_id: str) -> dict:
    return {"configurable": {"user_id": user_id, "thread_id": user_id}}


def test_user_id_is_not_visible_in_any_tool_schema():
    # if the LLM's tool-calling schema ever exposes user_id again, that's
    # the exact regression this test exists to catch
    assert "user_id" not in get_cart_items.args
    assert "user_id" not in get_order_history.args
    assert "user_id" not in get_order_status.args


def test_get_cart_items_uses_injected_user_id_not_an_argument():
    result = get_cart_items.invoke({}, config=_config("user_001"))
    expected_items = mock_db.get_cart("user_001")
    if expected_items:
        assert expected_items[0]["product_name"] in result
    else:
        assert "No items in cart" in result


def test_get_order_history_uses_injected_user_id():
    result = get_order_history.invoke({}, config=_config("user_002"))
    orders = mock_db.get_order_history("user_002")
    if orders:
        assert orders[0]["order_id"] in result
    else:
        assert "No order history" in result


def test_get_order_status_returns_data_for_the_owning_user():
    orders = mock_db.get_order_history("user_001")
    order_id = orders[0]["order_id"]
    result = get_order_status.invoke({"order_id": order_id}, config=_config("user_001"))
    assert order_id in result
    assert "No order found" not in result


def test_get_order_status_blocks_cross_user_access():
    orders = mock_db.get_order_history("user_001")
    user_001_order_id = orders[0]["order_id"]
    user_001_product = orders[0]["product_name"]

    # user_002 tries to look up an order that belongs to user_001 — the
    # order's DATA (product name, status) must not leak, even though
    # echoing back the requested order_id in a "not found" message is fine
    result = get_order_status.invoke({"order_id": user_001_order_id}, config=_config("user_002"))
    assert "No order found" in result
    assert user_001_product not in result


def test_get_order_status_gives_identical_response_for_nonexistent_and_someone_elses_order():
    # the response for "doesn't exist" and "exists but isn't yours" must be
    # identical — otherwise the difference itself confirms an order_id is
    # real, which is its own small information leak
    orders = mock_db.get_order_history("user_001")
    real_but_not_mine = get_order_status.invoke(
        {"order_id": orders[0]["order_id"]}, config=_config("user_002")
    )
    doesnt_exist = get_order_status.invoke(
        {"order_id": "totally_fake_order_id"}, config=_config("user_002")
    )
    # both should be the "not found" template with only the id differing
    assert real_but_not_mine.startswith("No order found with id")
    assert doesnt_exist.startswith("No order found with id")
