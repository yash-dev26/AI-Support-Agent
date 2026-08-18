"""
In-memory WebSocket connection manager, keyed by thread_id (== user_id).

This is enough for a single-process deployment: the same process that
resolves an escalation (POST /support/resolve/{thread_id}) is the same
process holding that user's websocket connection, so there's nothing to
bridge between processes.

A message queue (Redis pub/sub, etc.) becomes necessary once you run
multiple server instances behind a load balancer — the instance that
handles the resolve request might not be the instance holding the user's
open connection, and you'd need something to broadcast the resolution
across instances. That's a real, deliberate scope cut for this project,
not an oversight — call it out if asked.
"""
import logging

from fastapi import WebSocket

logger = logging.getLogger("support_agent.ws")


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, thread_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[thread_id] = websocket

    def disconnect(self, thread_id: str):
        self._connections.pop(thread_id, None)

    async def notify(self, thread_id: str, payload: dict) -> bool:
        """Push a message to the user's open connection, if any.

        Returns True if delivered, False if the user isn't currently
        connected — in which case they'll pick up the resolution via
        GET /chat/{user_id}/status instead.
        """
        ws = self._connections.get(thread_id)
        if ws is None:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            logger.warning("Failed to push to thread %s, dropping connection", thread_id, exc_info=True)
            self.disconnect(thread_id)
            return False


manager = ConnectionManager()


