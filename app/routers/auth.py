from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str
    role: str = "user"  # "user" | "support_agent"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=TokenResponse)
def issue_token(req: TokenRequest):
    """Issues a JWT for the given user_id/role.

    STUB — no password or credential check. Anyone can mint a token for
    any user_id or role, because there's no real user database with
    credentials in this project. What's real is everything downstream:
    once issued, this token is genuinely required and genuinely enforced
    on every protected endpoint (see app/core/auth.py). A real system
    swaps this endpoint's internals for actual credential verification;
    nothing downstream changes.
    """
    if req.role not in ("user", "support_agent"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'support_agent'.")
    token = create_access_token(req.user_id, req.role)
    return TokenResponse(access_token=token, expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
