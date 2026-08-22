"""
Read-only user context for the demo frontend's sidebar: profile, recent
orders, and cart in one call, instead of the frontend making three
separate round trips (and three separate auth checks) every time someone
switches which user they're viewing.

Deliberately thin — this wraps mock_db, it doesn't add any business
logic. If this project ever grew a real storefront, this is the endpoint
that would start talking to an actual commerce service instead.
"""
import logging

from fastapi import APIRouter, HTTPException, Depends

from app.core.auth import get_current_token, TokenPayload, require_matching_user, require_support_agent
from app.services import mock_db

logger = logging.getLogger("support_agent.users")
router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}/context")
def get_user_context(user_id: str, current: TokenPayload = Depends(get_current_token)):
    """Profile + recent order history + cart for one user, bundled for the
    demo sidebar. Same access rule as /chat/{user_id}: a customer token can
    only view its own context, a support_agent token can view anyone's —
    an agent looking at a customer's cart while handling their ticket is
    the whole point of a support tool.
    """
    require_matching_user(user_id, current)
    try:
        profile = mock_db.get_profile(user_id)
        orders = mock_db.get_order_history(user_id)
        cart = mock_db.get_cart(user_id)
    except Exception:
        logger.exception("Failed to load context for user %s", user_id)
        raise HTTPException(status_code=503, detail="Could not reach the commerce store. Try again shortly.") from None

    if not profile:
        raise HTTPException(status_code=404, detail=f"No user found with id {user_id}.") from None

    cart_total_cents = sum(item["amount_cents"] * item["quantity"] for item in cart)

    return {
        "profile": profile,
        "orders": orders[:5],
        "cart": {
            "items": cart,
            "total_cents": cart_total_cents,
        },
    }


@router.get("")
def list_users(current: TokenPayload = Depends(get_current_token)):
    """All seeded user_ids — powers the demo frontend's user picker.
    Support-agent only: a customer token has no reason to enumerate other
    customers, even just their ids."""
    require_support_agent(current)
    try:
        return {"user_ids": mock_db.list_user_ids(limit=30)}
    except Exception:
        logger.exception("Failed to list users")
        raise HTTPException(status_code=503, detail="Could not reach the commerce store. Try again shortly.") from None
