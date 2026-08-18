"""
Fires several simulated users at the API concurrently, escalates a subset,
and verifies the support queue and resume path don't clobber each other's
state. Requires the server running (uvicorn app.main:app) and MONGODB_URI set.
This is an integration test, not run in CI (needs live Mongo + OpenAI creds
and a running server) — run it manually.

Run: python tests/test_concurrent_escalation.py
"""
import concurrent.futures
import requests

BASE_URL = "http://localhost:8000"

USERS = [f"user_{i:03d}" for i in range(1, 6)]  # 5 simulated users
ESCALATE_MESSAGE = "I want to speak to a human, my payment was double charged and I need this fixed manually."
NORMAL_MESSAGE = "What's in my cart?"
RESOLUTION_TEXT = "Refund issued, duplicate charge reversed."  # money-related on purpose —
                                                                 # exercises the confirmation gate too


def get_token(user_id: str, role: str = "user") -> str:
    resp = requests.post(f"{BASE_URL}/auth/token", json={"user_id": user_id, "role": role}, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]


def run_user(user_id: str, should_escalate: bool):
    token = get_token(user_id)
    message = ESCALATE_MESSAGE if should_escalate else NORMAL_MESSAGE
    resp = requests.post(
        f"{BASE_URL}/chat/{user_id}",
        json={"message": message},
        headers={"Authorization": f"Bearer {token}"},
    )
    return user_id, should_escalate, resp.status_code, resp.json()


def main():
    agent_token = get_token("concurrency_test_agent", role="support_agent")
    agent_headers = {"Authorization": f"Bearer {agent_token}"}

    escalate_flags = [True, True, False, True, False]  # 3 escalations, 2 normal
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_user, u, f) for u, f in zip(USERS, escalate_flags)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    print("=== Initial chat results ===")
    for user_id, should_escalate, status, body in results:
        print(f"{user_id} (escalate={should_escalate}): HTTP {status} -> {body}")

    pending = requests.get(f"{BASE_URL}/support/pending", headers=agent_headers).json()["pending"]
    print(f"\n=== Pending escalations: {len(pending)} ===")
    for p in pending:
        print(p)

    escalated_ids = {r[0] for r in results if r[1]}
    pending_ids = {p["thread_id"] for p in pending}
    assert escalated_ids == pending_ids, (
        f"Mismatch: expected escalated {escalated_ids}, got pending {pending_ids}"
    )

    print("\n=== Resolving all pending in parallel ===")

    def resolve(thread_id):
        # RESOLUTION_TEXT is money-related, so confirmed=True is required
        # or every one of these would just come back confirmation_required
        # and never actually resolve — see money_detection.py.
        resp = requests.post(
            f"{BASE_URL}/support/resolve/{thread_id}",
            json={"resolution": RESOLUTION_TEXT, "confirmed": True},
            headers=agent_headers,
        )
        return thread_id, resp.status_code, resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        resolve_results = list(pool.map(resolve, pending_ids))

    for thread_id, status, body in resolve_results:
        print(f"{thread_id}: HTTP {status} -> {body}")

    remaining = requests.get(f"{BASE_URL}/support/pending", headers=agent_headers).json()["pending"]
    assert len(remaining) == 0, f"Expected no pending threads left, found {remaining}"

    print("\nAll escalations resolved independently with no cross-thread interference.")


if __name__ == "__main__":
    main()


