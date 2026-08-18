"""
Tests for app/routers/users.py (the sidebar context endpoint) and
app/routers/chat.py's _extract_tool_trace helper (the tool-execution
badge data). Uses a real SQLite-backed mock_db like test_mock_db.py, and
a real FastAPI TestClient like test_escalation_flow.py, since neither
needs a fake graph_app or Mongo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.auth import create_access_token
from app.routers import users
from app.routers.chat import _extract_tool_trace
from app.services import mock_db


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_commerce.db")
    mock_db.seed(num_users=5, force=True)
    yield


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(users.router)
    return TestClient(app)


def _user_headers(user_id: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id, role='user')}"}


def _agent_headers() -> dict:
    return {"Authorization": f"Bearer {create_access_token('agent_smith', role='support_agent')}"}


def test_get_context_returns_profile_orders_and_cart(client):
    resp = client.get("/users/user_001/context", headers=_user_headers("user_001"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["profile"]["user_id"] == "user_001"
    assert len(data["orders"]) >= 1
    assert "total_cents" in data["cart"]


def test_cart_total_is_sum_of_quantity_times_price(client):
    resp = client.get("/users/user_001/context", headers=_user_headers("user_001"))
    data = resp.json()
    expected = sum(i["amount_cents"] * i["quantity"] for i in data["cart"]["items"])
    assert data["cart"]["total_cents"] == expected


def test_customer_token_cannot_view_another_users_context(client):
    resp = client.get("/users/user_001/context", headers=_user_headers("user_002"))
    assert resp.status_code == 403


def test_support_agent_token_can_view_any_users_context(client):
    resp = client.get("/users/user_001/context", headers=_agent_headers())
    assert resp.status_code == 200


def test_context_404s_for_unknown_user(client):
    resp = client.get("/users/nonexistent_user/context", headers=_agent_headers())
    assert resp.status_code == 404


def test_list_users_requires_support_agent_token(client):
    resp = client.get("/users", headers=_user_headers("user_001"))
    assert resp.status_code == 403


def test_list_users_returns_seeded_ids_for_support_agent(client):
    resp = client.get("/users", headers=_agent_headers())
    assert resp.status_code == 200
    assert "user_001" in resp.json()["user_ids"]


# --- _extract_tool_trace ---------------------------------------------

def test_extract_tool_trace_marks_ordinary_tool_completed_when_result_present():
    messages = [
        HumanMessage(content="what's in my cart?"),
        AIMessage(content="", tool_calls=[{"name": "get_user_cart", "args": {}, "id": "call_1"}]),
        ToolMessage(content="- USB-C Hub x1", tool_call_id="call_1"),
        AIMessage(content="You have a USB-C Hub in your cart."),
    ]
    trace = _extract_tool_trace(messages)
    assert trace == [{"tool": "get_user_cart", "status": "completed"}]


def test_extract_tool_trace_marks_create_support_ticket_escalated_even_without_a_result():
    # create_support_ticket pauses INSIDE the tool on interrupt() — no
    # ToolMessage exists yet when this turn's messages are read back.
    messages = [
        HumanMessage(content="my card was charged twice"),
        AIMessage(content="", tool_calls=[{"name": "create_support_ticket", "args": {}, "id": "call_2"}]),
    ]
    trace = _extract_tool_trace(messages)
    assert trace == [{"tool": "create_support_ticket", "status": "escalated"}]


def test_extract_tool_trace_covers_multiple_calls_in_order():
    messages = [
        HumanMessage(content="what's my latest order and is it policy to refund late orders?"),
        AIMessage(content="", tool_calls=[
            {"name": "get_latest_order", "args": {}, "id": "call_3"},
            {"name": "check_policy", "args": {"query": "late refund"}, "id": "call_4"},
        ]),
        ToolMessage(content="ord_1: ... -- shipped", tool_call_id="call_3"),
        ToolMessage(content="[FAQ] ...", tool_call_id="call_4"),
    ]
    trace = _extract_tool_trace(messages)
    assert trace == [
        {"tool": "get_latest_order", "status": "completed"},
        {"tool": "check_policy", "status": "completed"},
    ]


def test_extract_tool_trace_is_empty_when_no_tools_were_called():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello!")]
    assert _extract_tool_trace(messages) == []
