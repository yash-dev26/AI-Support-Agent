"""
Integration test for the escalation side-channel across chat.py and
support.py. Uses a fake graph_app that mimics LangGraph's get_state()/
stream() contract exactly (interrupts, values, Command-based resume) so
the router logic is exercised for real — no live LLM/Mongo needed, since
this is deliberately testing routing behavior, not the graph itself
(which has its own tests in test_graph_routing.py).

Uses real JWTs (via create_access_token, the same function the actual
/auth/token endpoint calls) rather than bypassing auth — these tests are
what actually prove enforcement works, same as test_auth.py does more
narrowly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mongomock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.core import deps
from app.core.auth import create_access_token
from app.services import mock_db
from app.routers import chat, support


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeState:
    def __init__(self, values, interrupts=()):
        self.values = values
        self.interrupts = interrupts


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeGraphApp:
    """Simulates just enough of the compiled graph's interface for router
    tests: escalates when the message contains "human", otherwise echoes;
    resumes verbatim on Command (matching what route_after_tools produces
    in the real graph, tested separately)."""

    def __init__(self):
        self._state: dict[str, FakeState] = {}

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self._state.get(thread_id, FakeState({"messages": []}))

    def stream(self, input_data, config, stream_mode):
        thread_id = config["configurable"]["thread_id"]
        if isinstance(input_data, Command):
            resolution = input_data.resume["data"]
            msg = FakeMessage(resolution)
            self._state[thread_id] = FakeState({"messages": [msg]}, interrupts=())
            yield {"messages": [msg]}
        else:
            text = input_data["messages"][0]["content"]
            if "human" in text.lower():
                self._state[thread_id] = FakeState({"messages": []}, interrupts=(FakeInterrupt({"query": text}),))
                yield {"messages": []}
            else:
                msg = FakeMessage(f"echo: {text}")
                self._state[thread_id] = FakeState({"messages": [msg]}, interrupts=())
                yield {"messages": [msg]}


def user_headers(user_id: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id, role='user')}"}


def agent_headers() -> dict:
    return {"Authorization": f"Bearer {create_access_token('agent_smith', role='support_agent')}"}


@pytest.fixture
def client():
    deps.set_mongo_client(mongomock.MongoClient())
    deps.set_graph_app(FakeGraphApp())
    mock_db.seed(num_users=3, force=True)

    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(support.router)
    return TestClient(app)


def test_normal_message_goes_to_the_llm_path(client):
    resp = client.post("/chat/user_001", json={"message": "hi there"}, headers=user_headers("user_001"))
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "reply": "echo: hi there"}


def test_chat_rejects_missing_token(client):
    resp = client.post("/chat/user_001", json={"message": "hi there"})
    assert resp.status_code == 401


def test_chat_rejects_token_for_a_different_user(client):
    resp = client.post("/chat/user_001", json={"message": "hi"}, headers=user_headers("user_002"))
    assert resp.status_code == 403


def test_chat_allows_support_agent_token_on_any_user(client):
    resp = client.post("/chat/user_001", json={"message": "hi"}, headers=agent_headers())
    assert resp.status_code == 200


def test_escalating_message_triggers_escalation(client):
    resp = client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"


def test_message_sent_after_escalation_goes_to_side_channel_not_the_llm(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))
    resp = client.post("/chat/user_001", json={"message": "are you there?"}, headers=user_headers("user_001"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "escalated"
    assert "support" in data["message"].lower()

    thread = client.get("/support/thread/user_001", headers=agent_headers()).json()
    texts = [m["text"] for m in thread["messages"]]
    assert "let me talk to a human" in texts
    assert "are you there?" in texts
    assert thread["pending"] is True


def test_pending_list_includes_message_count(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))
    client.post("/chat/user_001", json={"message": "hello?"}, headers=user_headers("user_001"))

    pending = client.get("/support/pending", headers=agent_headers()).json()["pending"]
    entry = next(p for p in pending if p["thread_id"] == "user_001")
    assert entry["message_count"] == 2


def test_pending_list_includes_a_non_seeded_user_id(client):
    # regression test: /support/pending used to scan mock_db's seeded user
    # list, so a thread for any user_id NOT in that pre-seeded set (e.g.
    # someone typing a custom value into the frontend's user id field)
    # would escalate correctly but never actually appear in the queue.
    custom_user_id = "totally_unseeded_walk_in_user"
    client.post(
        f"/chat/{custom_user_id}",
        json={"message": "let me talk to a human"},
        headers=user_headers(custom_user_id),
    )

    pending = client.get("/support/pending", headers=agent_headers()).json()["pending"]
    thread_ids = [p["thread_id"] for p in pending]
    assert custom_user_id in thread_ids


def test_support_can_reply_without_resolving(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))

    resp = client.post("/support/thread/user_001/reply", json={"text": "checking on this now"}, headers=agent_headers())
    assert resp.status_code == 200
    assert resp.json()["status"] == "sent"

    thread = client.get("/support/thread/user_001", headers=agent_headers()).json()
    assert thread["pending"] is True  # replying doesn't resolve
    support_messages = [m for m in thread["messages"] if m["sender"] == "support"]
    assert any(m["text"] == "checking on this now" for m in support_messages)


def test_reply_to_non_escalated_thread_is_rejected(client):
    resp = client.post("/support/thread/user_002/reply", json={"text": "hi"}, headers=agent_headers())
    assert resp.status_code == 404


def test_resolve_delivers_verbatim_and_reopens_llm_path(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))

    resp = client.post(
        "/support/resolve/user_001",
        json={"resolution": "Refund issued.", "confirmed": True},
        headers=agent_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_reply"] == "Refund issued."

    thread = client.get("/support/thread/user_001", headers=agent_headers()).json()
    assert thread["pending"] is False
    assert any(m["text"] == "Refund issued." and m["sender"] == "support" for m in thread["messages"])

    # user can talk to the LLM again after resolution
    resp2 = client.post("/chat/user_001", json={"message": "thanks, one more question"}, headers=user_headers("user_001"))
    assert resp2.json() == {"status": "ok", "reply": "echo: thanks, one more question"}


def test_money_related_resolution_requires_confirmation(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))

    resp = client.post(
        "/support/resolve/user_001",
        json={"resolution": "Refund issued for $50."},
        headers=agent_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "confirmation_required"

    # nothing should have actually happened — thread still pending, no
    # graph resume, no support message logged yet
    thread = client.get("/support/thread/user_001", headers=agent_headers()).json()
    assert thread["pending"] is True
    assert not any(m["sender"] == "support" for m in thread["messages"])


def test_money_related_resolution_proceeds_once_confirmed(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))
    client.post("/support/resolve/user_001", json={"resolution": "Refund issued for $50."}, headers=agent_headers())

    resp = client.post(
        "/support/resolve/user_001",
        json={"resolution": "Refund issued for $50.", "confirmed": True},
        headers=agent_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resumed"
    assert data["final_reply"] == "Refund issued for $50."

    thread = client.get("/support/thread/user_001", headers=agent_headers()).json()
    assert thread["pending"] is False


def test_non_money_resolution_does_not_require_confirmation(client):
    client.post("/chat/user_001", json={"message": "let me talk to a human"}, headers=user_headers("user_001"))

    resp = client.post(
        "/support/resolve/user_001",
        json={"resolution": "Your package was found and redelivered this morning."},
        headers=agent_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resumed"  # no confirmation needed, resolved immediately
