"""
Confirms rate limiting is actually wired into the real app (middleware +
exception handler + app.state.limiter), not just that the decorated
route functions don't crash. Uses a fake graph + mongomock so this runs
without live credentials.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-for-tests")
os.environ.setdefault("MONGODB_URI", "mongodb://fake:27017")

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import deps
from app.core.auth import create_access_token


class FakeState:
    def __init__(self):
        self.interrupts = ()
        self.values = {"messages": []}


class FakeGraphApp:
    def get_state(self, config):
        return FakeState()

    def stream(self, *args, **kwargs):
        yield {"messages": []}


def user_headers(user_id: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user_id, role='user')}"}


@pytest.fixture
def client():
    deps.set_graph_app(FakeGraphApp())
    deps.set_mongo_client(mongomock.MongoClient())
    return TestClient(app)


def test_chat_endpoint_rate_limits_after_15_requests_per_user(client):
    headers = user_headers("rl_user_a")
    statuses = [client.post("/chat/rl_user_a", json={"message": "hi"}, headers=headers).status_code for _ in range(17)]
    assert statuses[:15] == [200] * 15
    assert all(s == 429 for s in statuses[15:])


def test_rate_limit_is_scoped_per_user_not_global(client):
    headers_b = user_headers("rl_user_b")
    # exhaust one user's limit
    for _ in range(15):
        client.post("/chat/rl_user_b", json={"message": "hi"}, headers=headers_b)
    exhausted = client.post("/chat/rl_user_b", json={"message": "hi"}, headers=headers_b)
    assert exhausted.status_code == 429

    # a different user should be unaffected
    headers_c = user_headers("rl_user_c")
    fresh = client.post("/chat/rl_user_c", json={"message": "hi"}, headers=headers_c)
    assert fresh.status_code == 200
