"""
Tests for app/core/auth.py directly — token creation, decoding, expiry,
and the require_matching_user / require_support_agent enforcement logic.
Complements test_escalation_flow.py, which proves this is actually wired
into real endpoints; these tests isolate the auth logic itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import jwt
import pytest
from fastapi import HTTPException

from app.core import auth


def test_create_and_decode_round_trip():
    token = auth.create_access_token("user_001", role="user")
    payload = auth._decode(token)
    assert payload.sub == "user_001"
    assert payload.role == "user"


def test_default_role_is_user():
    token = auth.create_access_token("user_002")
    payload = auth._decode(token)
    assert payload.role == "user"


def test_expired_token_is_rejected():
    expired_payload = {
        "sub": "user_001",
        "role": "user",
        "exp": int(time.time()) - 3600,
    }
    expired_token = jwt.encode(expired_payload, auth._SECRET_KEY, algorithm=auth.ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        auth._decode(expired_token)
    assert exc_info.value.status_code == 401


def test_token_signed_with_wrong_secret_is_rejected():
    forged = jwt.encode(
        {"sub": "user_001", "role": "support_agent", "exp": int(time.time()) + 3600},
        "wrong-secret-entirely",
        algorithm=auth.ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc_info:
        auth._decode(forged)
    assert exc_info.value.status_code == 401


def test_garbage_token_is_rejected():
    with pytest.raises(HTTPException) as exc_info:
        auth._decode("not-a-real-jwt-at-all")
    assert exc_info.value.status_code == 401


def test_require_matching_user_allows_own_token():
    token = auth._decode(auth.create_access_token("user_001", role="user"))
    auth.require_matching_user("user_001", token)  # should not raise


def test_require_matching_user_rejects_different_user():
    token = auth._decode(auth.create_access_token("user_001", role="user"))
    with pytest.raises(HTTPException) as exc_info:
        auth.require_matching_user("user_002", token)
    assert exc_info.value.status_code == 403


def test_require_matching_user_allows_support_agent_for_any_user():
    token = auth._decode(auth.create_access_token("agent_1", role="support_agent"))
    auth.require_matching_user("literally_any_user_id", token)  # should not raise


def test_require_support_agent_allows_agent_role():
    token = auth._decode(auth.create_access_token("agent_1", role="support_agent"))
    auth.require_support_agent(token)  # should not raise


def test_require_support_agent_rejects_user_role():
    token = auth._decode(auth.create_access_token("user_001", role="user"))
    with pytest.raises(HTTPException) as exc_info:
        auth.require_support_agent(token)
    assert exc_info.value.status_code == 403


def _protected_test_client():
    """get_current_token now depends on HTTPBearer, which does its own
    validation during FastAPI's dependency resolution (before our function
    body ever runs) — so testing "missing/malformed header" behavior needs
    a real HTTP call through a minimal app, not a direct function call."""
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/protected")
    def protected(current: auth.TokenPayload = Depends(auth.get_current_token)):
        return {"sub": current.sub}

    return TestClient(app)


def test_get_current_token_rejects_missing_header():
    client = _protected_test_client()
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_get_current_token_rejects_malformed_header():
    client = _protected_test_client()
    resp = client.get("/protected", headers={"Authorization": "NotBearer sometoken"})
    assert resp.status_code == 401


def test_get_current_token_accepts_valid_bearer_header():
    client = _protected_test_client()
    token = auth.create_access_token("user_001", role="user")
    resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["sub"] == "user_001"


def test_get_current_token_from_query_rejects_missing_token():
    with pytest.raises(HTTPException) as exc_info:
        auth.get_current_token_from_query(token=None)
    assert exc_info.value.status_code == 401


def test_get_current_token_from_query_accepts_valid_token():
    token = auth.create_access_token("user_001", role="user")
    payload = auth.get_current_token_from_query(token=token)
    assert payload.sub == "user_001"


def test_auth_token_endpoint_issues_a_working_token():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import auth as auth_router

    app = FastAPI()
    app.include_router(auth_router.router)
    client = TestClient(app)

    resp = client.post("/auth/token", json={"user_id": "user_001", "role": "user"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0

    payload = auth._decode(data["access_token"])
    assert payload.sub == "user_001"
    assert payload.role == "user"


def test_auth_token_endpoint_rejects_invalid_role():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import auth as auth_router

    app = FastAPI()
    app.include_router(auth_router.router)
    client = TestClient(app)

    resp = client.post("/auth/token", json={"user_id": "user_001", "role": "superadmin"})
    assert resp.status_code == 400


