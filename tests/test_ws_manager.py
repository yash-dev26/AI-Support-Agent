"""
Tests for the in-memory WebSocket connection manager. Uses a lightweight
fake WebSocket rather than spinning up a real FastAPI app, so these run
fast and don't need any live credentials.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import pytest
from app.services.ws_manager import ConnectionManager


class FakeWebSocket:
    def __init__(self, fail_on_send: bool = False):
        self.accepted = False
        self.sent: list[dict] = []
        self.fail_on_send = fail_on_send

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload: dict):
        if self.fail_on_send:
            raise RuntimeError("simulated broken connection")
        self.sent.append(payload)


def test_connect_registers_and_accepts():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    asyncio.run(manager.connect("user_001", ws))
    assert ws.accepted is True
    assert "user_001" in manager._connections


def test_notify_delivers_to_connected_user():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    asyncio.run(manager.connect("user_001", ws))

    delivered = asyncio.run(manager.notify("user_001", {"status": "resolved", "reply": "done"}))
    assert delivered is True
    assert ws.sent == [{"status": "resolved", "reply": "done"}]


def test_notify_returns_false_for_unconnected_user():
    manager = ConnectionManager()
    delivered = asyncio.run(manager.notify("nobody_connected", {"status": "resolved"}))
    assert delivered is False


def test_disconnect_removes_user():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    asyncio.run(manager.connect("user_001", ws))
    manager.disconnect("user_001")
    assert "user_001" not in manager._connections


def test_notify_drops_connection_on_send_failure():
    manager = ConnectionManager()
    ws = FakeWebSocket(fail_on_send=True)
    asyncio.run(manager.connect("user_001", ws))

    delivered = asyncio.run(manager.notify("user_001", {"status": "resolved"}))
    assert delivered is False
    # a broken connection should be cleaned up, not left dangling
    assert "user_001" not in manager._connections


