"""
Scripted eval for "escalation is a last resort, not a default" — the one
piece of this project's behavior that genuinely can't be verified without
a live LLM call. Everything else (routing logic, the escalation
side-channel, the graph wiring) has unit/integration test coverage that
runs without credentials; whether GPT-4.1 actually *obeys* the system
prompt is a different kind of question, answerable only by actually
asking it.

Requires the server running (uvicorn app.main:app) with real
MONGODB_URI/OPENAI_API_KEY, and the seeded mock data (user_001/002/003
have scripted scenarios — see mock_db.py). Not run in CI.

Run: python tests/manual_eval_escalation.py
"""
import sys
import uuid
import requests

BASE_URL = "http://localhost:8000"


def fresh_user(prefix: str) -> str:
    # unique thread per scenario so scenarios don't interfere with each
    # other's escalation state across runs
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


SCENARIOS = [
    {
        "name": "Simple FAQ — should NOT escalate",
        "user_id": fresh_user("eval"),
        "message": "What's your refund policy?",
        "expect_escalated": False,
    },
    {
        "name": "Bare request for a human, no real problem — should NOT escalate immediately",
        "user_id": fresh_user("eval"),
        "message": "I want to talk to a human.",
        "expect_escalated": False,
        "soft": True,  # informational — model might reasonably ask a clarifying
                        # question instead of either answering or escalating,
                        # so this is worth eyeballing rather than hard-failing on
    },
    {
        "name": "Duplicate charge (real policy-driven escalation) — SHOULD escalate",
        "user_id": "user_001",  # scripted duplicate-charge scenario from mock_db.py
        "message": "I was charged twice for the same order, can you check my order history?",
        "expect_escalated": True,
    },
    {
        "name": "Refund request outside the 14-day window — should NOT escalate, policy says no",
        "user_id": "user_002",  # scripted 20-day-old delivered order
        "message": "I'd like a refund for my order, can you check when it was delivered?",
        "expect_escalated": False,
    },
    {
        "name": "Order stuck processing for 18 days — SHOULD escalate per shipping_policy.md",
        "user_id": "user_003",  # scripted stuck-processing scenario
        "message": "My order has been stuck processing for weeks, what's going on?",
        "expect_escalated": True,
    },
    {
        "name": "Fraudulent charge — SHOULD escalate per account_and_billing.md",
        "user_id": fresh_user("eval"),
        "message": "There's a charge on my account I don't recognize at all, I think it's fraud.",
        "expect_escalated": True,
    },
    {
        "name": "Completely out-of-scope question — SHOULD escalate (NO_ANSWER_FOUND)",
        "user_id": fresh_user("eval"),
        "message": "Do you ship to the International Space Station?",
        "expect_escalated": True,
    },
    {
        "name": "Ordinary tool use — should NOT escalate",
        "user_id": fresh_user("eval"),
        "message": "What's in my cart right now?",
        "expect_escalated": False,
    },
]


def get_token(user_id: str) -> str:
    resp = requests.post(f"{BASE_URL}/auth/token", json={"user_id": user_id, "role": "user"}, timeout=10)
    resp.raise_for_status()
    return resp.json()["access_token"]


def run_scenario(scenario: dict) -> tuple[bool, str]:
    try:
        token = get_token(scenario["user_id"])
        resp = requests.post(
            f"{BASE_URL}/chat/{scenario['user_id']}",
            json={"message": scenario["message"]},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return False, f"request failed: {e}"

    escalated = data.get("status") == "escalated"
    passed = escalated == scenario["expect_escalated"]
    detail = f"escalated={escalated}, reply={data.get('reply', data.get('message', ''))[:100]!r}"
    return passed, detail


def main():
    print(f"Running {len(SCENARIOS)} escalation-behavior scenarios against {BASE_URL}\n")
    results = []
    for scenario in SCENARIOS:
        passed, detail = run_scenario(scenario)
        soft = scenario.get("soft", False)
        status = "PASS" if passed else ("INFO" if soft else "FAIL")
        results.append((status, scenario["name"], detail))
        print(f"[{status}] {scenario['name']}")
        print(f"       {detail}\n")

    hard_failures = [r for r in results if r[0] == "FAIL"]
    print("=" * 60)
    print(f"{len(results) - len(hard_failures)}/{len(results)} scenarios behaved as expected "
          f"(soft/INFO scenarios don't count as failures either way).")
    if hard_failures:
        print(f"\n{len(hard_failures)} FAILED — escalation behavior doesn't match expectations:")
        for _, name, detail in hard_failures:
            print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
