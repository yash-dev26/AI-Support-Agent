"""
Minimal auth stub: JWT issuance + verification, with a "user" vs
"support_agent" role distinction.

This is deliberately a STUB, not real authentication — POST /auth/token
mints a token for whatever user_id/role is asked for, with no password or
credential check, because there's no real user database with credentials
in this project. What it DOES demonstrate honestly: once a token exists,
it's actually enforced — a customer token can only act as its own
user_id, a support-agent token is required for the support endpoints, and
neither can be forged without the signing secret. That enforcement is the
part worth the exercise; the login step itself is intentionally fake.

A real system would replace POST /auth/token with actual credential
verification (password hash check, OAuth, SSO, whatever) — everything
downstream of "we now have a valid token for this user_id/role" stays
the same.
"""
import os
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("support_agent.auth")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
if not _SECRET_KEY:
    _SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "AUTH_SECRET_KEY not set — generated a random one for this process. "
        "Tokens issued now will stop working after a restart, and won't be "
        "valid across multiple instances. Set AUTH_SECRET_KEY in .env for "
        "anything beyond local single-process use."
    )


class TokenPayload(BaseModel):
    sub: str  # user_id
    role: str  # "user" | "support_agent"


def create_access_token(user_id: str, role: str = "user") -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "role": role, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=ALGORITHM)


def _decode(token: str) -> TokenPayload:
    try:
        raw = jwt.decode(token, _SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.") from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.") from None
    return TokenPayload(sub=raw["sub"], role=raw.get("role", "user"))


def get_current_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> TokenPayload:
    """For normal HTTP routes. Uses FastAPI's HTTPBearer security scheme
    (rather than a plain Header dependency) specifically so Swagger UI
    renders a real "Authorize" button and padlock icons on protected
    routes, instead of just an ordinary text field — get a token from
    POST /auth/token, paste it into that dialog once, and it's attached
    to every subsequent request Swagger makes for you."""
    return _decode(credentials.credentials)


def get_current_token_from_query(token: Optional[str] = Query(None)) -> TokenPayload:
    """For the websocket route — browsers can't set custom headers on a
    WebSocket handshake, so the token travels as a query param instead
    (?token=...). Same validation either way."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing token query parameter.") from None
    return _decode(token)


def require_matching_user(user_id: str, current: TokenPayload) -> None:
    """A customer token may only act as its own user_id. A support_agent
    token may act on behalf of any user_id (an agent debugging a customer's
    chat, for instance) — raises 403 otherwise."""
    if current.role == "support_agent":
        return
    if current.sub != user_id:
        raise HTTPException(status_code=403, detail="Token does not grant access to this user_id.") from None


def require_support_agent(current: TokenPayload) -> None:
    if current.role != "support_agent":
        raise HTTPException(status_code=403, detail="This endpoint requires a support_agent token.") from None


