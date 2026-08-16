"""
Tests for escalation_store using mongomock (an in-memory pymongo-compatible
fake), so no live Mongo connection is needed in CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mongomock
import pytest

from app.core import deps
from app.services import escalation_store


@pytest.fixture(autouse=True)
def fake_mongo():
    client = mongomock.MongoClient()
    deps.set_mongo_client(client)
    yield client


def test_add_and_get_messages_round_trip():
    escalation_store.add_message("user_001", "user", "my order never arrived")
    escalation_store.add_message("user_001", "support", "let me check that for you")

    messages = escalation_store.get_messages("user_001")
    assert len(messages) == 2
    assert messages[0]["sender"] == "user"
    assert messages[0]["text"] == "my order never arrived"
    assert messages[1]["sender"] == "support"


def test_messages_are_returned_in_chronological_order():
    for i in range(5):
        escalation_store.add_message("user_002", "user", f"message {i}")

    messages = escalation_store.get_messages("user_002")
    texts = [m["text"] for m in messages]
    assert texts == [f"message {i}" for i in range(5)]


def test_messages_are_isolated_per_thread():
    escalation_store.add_message("user_001", "user", "thread A message")
    escalation_store.add_message("user_002", "user", "thread B message")

    thread_a = escalation_store.get_messages("user_001")
    thread_b = escalation_store.get_messages("user_002")

    assert len(thread_a) == 1
    assert len(thread_b) == 1
    assert thread_a[0]["text"] == "thread A message"
    assert thread_b[0]["text"] == "thread B message"


def test_empty_thread_returns_empty_list():
    assert escalation_store.get_messages("never_escalated_user") == []
