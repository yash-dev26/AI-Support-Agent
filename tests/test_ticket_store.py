"""
Tests for ticket_store using mongomock (an in-memory pymongo-compatible
fake), so no live Mongo connection is needed in CI. Mirrors the pattern
in test_escalation_store.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mongomock
import pytest

from app.core import deps
from app.services import ticket_store


@pytest.fixture(autouse=True)
def fake_mongo():
    client = mongomock.MongoClient()
    deps.set_mongo_client(client)
    yield client


def test_create_ticket_round_trips_with_open_status():
    doc = ticket_store.create_ticket("tkt_1", "user_001", "missing_order", "Order never arrived")
    assert doc["status"] == ticket_store.OPEN
    assert doc["resolved_at"] is None

    fetched = ticket_store.get_ticket("tkt_1")
    assert fetched["ticket_id"] == "tkt_1"
    assert fetched["user_id"] == "user_001"
    assert fetched["issue_type"] == "missing_order"


def test_get_ticket_returns_none_for_unknown_id():
    assert ticket_store.get_ticket("does_not_exist") is None


def test_update_status_to_resolved_sets_resolved_at():
    ticket_store.create_ticket("tkt_2", "user_002", "account_access", "Locked out")
    updated = ticket_store.update_status("tkt_2", ticket_store.RESOLVED)
    assert updated["status"] == ticket_store.RESOLVED
    assert updated["resolved_at"] is not None


def test_list_open_tickets_excludes_resolved():
    ticket_store.create_ticket("tkt_3", "user_003", "duplicate_charge", "Charged twice")
    ticket_store.create_ticket("tkt_4", "user_003", "missing_order", "Never arrived")
    ticket_store.update_status("tkt_3", ticket_store.RESOLVED)

    open_tickets = ticket_store.list_open_tickets()
    open_ids = {t["ticket_id"] for t in open_tickets}
    assert "tkt_4" in open_ids
    assert "tkt_3" not in open_ids


def test_list_tickets_for_user_is_isolated_per_user():
    ticket_store.create_ticket("tkt_5", "user_004", "missing_order", "...")
    ticket_store.create_ticket("tkt_6", "user_005", "missing_order", "...")

    user_004_tickets = ticket_store.list_tickets_for_user("user_004")
    assert len(user_004_tickets) == 1
    assert user_004_tickets[0]["ticket_id"] == "tkt_5"


def test_priority_is_inferred_high_for_money_related_issues():
    doc = ticket_store.create_ticket("tkt_7", "user_006", "duplicate_charge", "I was charged twice for the same order")
    assert doc["priority"] == "high"


def test_priority_defaults_to_normal_for_generic_issues():
    doc = ticket_store.create_ticket("tkt_8", "user_007", "general_question", "How do I change my email address?")
    assert doc["priority"] == "normal"


def test_create_ticket_is_idempotent_under_the_same_ticket_id():
    # Regression test for a real bug: langgraph replays a node's entire
    # function body from the top on every interrupt() resume (see
    # langgraph.types.interrupt's docstring: "The graph resumes from the
    # start of the node, re-executing all logic"). create_support_ticket
    # in tools.py calls ticket_store.create_ticket() BEFORE interrupt(),
    # so that call re-runs on every human reply. If create_ticket weren't
    # an idempotent upsert, every reply would silently mint a brand new
    # duplicate ticket. This test simulates exactly that replay by
    # calling create_ticket twice with the same ticket_id (which is what
    # tools.py now derives deterministically from tool_call_id, not
    # uuid4()) and asserts only one ticket document exists.
    ticket_store.create_ticket("tkt_replay_1", "user_008", "missing_order", "first pass details")
    ticket_store.create_ticket("tkt_replay_1", "user_008", "missing_order", "first pass details")

    all_for_user = ticket_store.list_tickets_for_user("user_008")
    assert len(all_for_user) == 1


def test_create_ticket_replay_does_not_reset_an_already_resolved_ticket():
    # A second replay of the same node (e.g. a support agent's reply
    # triggering another resume) must not stomp a status that already
    # advanced past "open" back to "open" — $setOnInsert protects this.
    ticket_store.create_ticket("tkt_replay_2", "user_009", "account_access", "...")
    ticket_store.update_status("tkt_replay_2", ticket_store.RESOLVED)

    ticket_store.create_ticket("tkt_replay_2", "user_009", "account_access", "...")  # replay

    ticket = ticket_store.get_ticket("tkt_replay_2")
    assert ticket["status"] == ticket_store.RESOLVED
