"""
Tiny shared-state module. FastAPI lifespan sets these once at startup;
routers read them via the getters below instead of importing globals
directly from main.py (which would create a circular import once main.py
needs to import the routers).
"""
from datetime import datetime, timezone

_state = {
    "graph_app": None,
    "mongo_client": None,
}


def set_graph_app(graph_app):
    _state["graph_app"] = graph_app


def get_graph_app():
    if _state["graph_app"] is None:
        raise RuntimeError("graph_app not initialized — lifespan hasn't run yet.")
    return _state["graph_app"]


def set_mongo_client(client):
    _state["mongo_client"] = client


def get_mongo_client():
    if _state["mongo_client"] is None:
        raise RuntimeError("mongo_client not initialized — lifespan hasn't run yet.")
    return _state["mongo_client"]


def get_metrics_col():
    return get_mongo_client()["support_agent"]["events"]


def log_event(event_type: str, thread_id: str, **extra):
    get_metrics_col().insert_one({
        "event_type": event_type,
        "thread_id": thread_id,
        "timestamp": datetime.now(timezone.utc),
        **extra,
    })
