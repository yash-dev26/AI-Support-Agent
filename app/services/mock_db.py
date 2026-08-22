"""
Mock commerce layer. No storefront, no frontend — just enough structured data
for tools to query against, so the agent can answer "what's in my cart" /
"where's my order" style questions realistically.

SQLite so there's zero setup cost. Swap for Postgres later if you want to
show connection pooling, but it's not the point of this project.

A few users get deterministic, policy-relevant scenarios instead of pure
random data (see SCRIPTED_SCENARIOS below) — e.g. user_001 always has a
duplicate charge, which is the exact case refund_policy.md says must be
escalated to a human. That makes the escalation-as-last-resort behavior
reproducible to test manually, instead of having to hand-craft a message
every time.
"""
import sqlite3
import random
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "commerce.db"

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Isha", "Kabir", "Meera",
    "Arjun", "Diya", "Karan", "Neha", "Rahul", "Sanya", "Aditya", "Riya",
    "Dev", "Tara", "Nikhil", "Simran",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Kapoor", "Menon",
    "Joshi", "Rao", "Chatterjee", "Malhotra", "Bose", "Pillai", "Desai",
]
PRODUCTS = [
    ("Wireless Mouse", 799), ("Mechanical Keyboard", 3499), ("USB-C Hub", 1299),
    ("Laptop Stand", 1899), ("Noise Cancelling Headphones", 6999), ("Webcam 1080p", 2199),
    ("Portable SSD 1TB", 5499), ("Monitor Arm", 2799), ("Wireless Charger Pad", 1499),
    ("Bluetooth Speaker", 2999), ("Ergonomic Desk Mat", 1199), ("27-inch Monitor", 15999),
    ("Mechanical Numpad", 1799), ("Cable Organizer Kit", 499), ("Laptop Sleeve 14-inch", 1099),
    ("Smart LED Desk Lamp", 2299), ("USB Microphone", 3999), ("Graphics Tablet", 8999),
]
ORDER_STATUSES = ["delivered", "shipped", "processing", "cancelled", "refunded"]
# Tracking numbers only make sense once an order has actually left the
# warehouse — "processing"/"cancelled"/"refunded" orders get NULL, not a
# fake number, so get_latest_order/get_order_by_id can tell a real reader
# "not shipped yet" instead of printing a bogus tracking code.
TRACKING_ELIGIBLE_STATUSES = {"shipped", "delivered"}
CARRIERS = ["BlueDart", "Delhivery", "DTDC", "India Post"]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            status TEXT NOT NULL,
            placed_at TEXT NOT NULL,
            tracking_number TEXT,
            carrier TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            amount_cents INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        """)
        conn.commit()


def _insert_order(conn, order_id, user_id, product, price_cents, status, days_ago):
    tracking_number, carrier = (None, None)
    if status in TRACKING_ELIGIBLE_STATUSES:
        # Deterministic from order_id (not random) so re-running seed with
        # the same scripted order_ids doesn't produce a different tracking
        # number every time — useful when a demo video needs a stable value.
        tracking_number = f"TRK{abs(hash(order_id)) % 10**9:09d}"
        carrier = CARRIERS[abs(hash(order_id)) % len(CARRIERS)]
    conn.execute(
        """INSERT INTO orders (order_id, user_id, product_name, amount_cents, status, placed_at, tracking_number, carrier)
           VALUES (?, ?, ?, ?, ?, datetime('now', ?), ?, ?)""",
        (order_id, user_id, product, price_cents, status, f"-{days_ago} days", tracking_number, carrier),
    )


def _seed_scripted_scenarios(conn):
    """Deterministic, policy-relevant data for the first few users — makes
    manual/demo testing of specific escalation paths reproducible instead
    of relying on random luck or a hand-typed message every time."""

    # user_001: a genuine duplicate charge — refund_policy.md explicitly
    # says this "requires manual verification ... escalated to a human
    # support agent." Asking about this should make check_policy surface
    # that clause and the model should escalate because of it, not
    # because the user merely asked to.
    conn.execute(
        "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)",
        ("user_001", "Aarav Sharma", "user_001@example.com"),
    )
    _insert_order(conn, "ord_user_001_0", "user_001", "Mechanical Keyboard", 349900, "processing", 1)
    _insert_order(conn, "ord_user_001_1", "user_001", "Mechanical Keyboard", 349900, "processing", 1)
    conn.execute(
        "INSERT INTO cart_items (user_id, product_name, quantity, amount_cents) VALUES (?, ?, ?, ?)",
        ("user_001", "USB-C Hub", 1, 129900),
    )

    # user_002: an order delivered 20 days ago — outside refund_policy.md's
    # 14-day window. A refund request here should get a real policy-backed
    # "no" from check_policy, not an escalation.
    conn.execute(
        "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)",
        ("user_002", "Priya Verma", "user_002@example.com"),
    )
    _insert_order(conn, "ord_user_002_0", "user_002", "Noise Cancelling Headphones", 699900, "delivered", 20)

    # user_003: an order stuck "processing" for 18 days — shipping_policy.md
    # says packages missing past a threshold need investigation, another
    # real (non-user-requested) escalation trigger.
    conn.execute(
        "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)",
        ("user_003", "Rohan Iyer", "user_003@example.com"),
    )
    _insert_order(conn, "ord_user_003_0", "user_003", "27-inch Monitor", 1599900, "processing", 18)


def seed(num_users: int = 20, force: bool = False):
    init_schema()
    with get_conn() as conn:
        if not force:
            existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            if existing > 0:
                print(f"DB already seeded ({existing} users). Use force=True to reseed.")
                return
        conn.executescript("DELETE FROM cart_items; DELETE FROM orders; DELETE FROM users;")

        scripted_ids = {"user_001", "user_002", "user_003"}
        if num_users >= 3:
            _seed_scripted_scenarios(conn)

        start = len(scripted_ids) + 1 if num_users >= 3 else 1
        for i in range(start, num_users + 1):
            user_id = f"user_{i:03d}"
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            conn.execute(
                "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)",
                (user_id, name, f"{user_id}@example.com"),
            )

            for o in range(random.randint(1, 4)):
                product, price = random.choice(PRODUCTS)
                order_id = f"ord_{user_id}_{o}"
                _insert_order(conn, order_id, user_id, product, price * 100,
                               random.choice(ORDER_STATUSES), random.randint(1, 60))

            for _ in range(random.randint(0, 3)):
                product, price = random.choice(PRODUCTS)
                conn.execute(
                    """INSERT INTO cart_items (user_id, product_name, quantity, amount_cents)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, product, random.randint(1, 2), price * 100),
                )
        conn.commit()
    print(f"Seeded {num_users} users with orders and cart items "
          f"({'including' if num_users >= 3 else 'excluding'} scripted escalation scenarios).")


def get_cart(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT product_name, quantity, amount_cents FROM cart_items WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_order_history(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT order_id, product_name, amount_cents, status, placed_at, tracking_number, carrier
               FROM orders WHERE user_id = ? ORDER BY placed_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_order(user_id: str) -> dict | None:
    """The single most recent order for a user, or None if they have never
    ordered anything. Separate from get_order_history (which returns the
    whole list) so a tool/caller that only cares about "my latest order"
    doesn't have to fetch everything and pick index 0 itself."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT order_id, product_name, amount_cents, status, placed_at, tracking_number, carrier
               FROM orders WHERE user_id = ? ORDER BY placed_at DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def get_order_status(order_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT order_id, status, product_name, user_id, amount_cents,
                      placed_at, tracking_number, carrier
               FROM orders WHERE order_id = ?""",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None


def get_profile(user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id, name, email FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def list_user_ids(limit: int = 30) -> list[str]:
    """Every seeded user_id, lowest-numbered first — powers the demo
    frontend's user picker so a viewer can browse real users instead of
    typing an id blind."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT user_id FROM users ORDER BY user_id LIMIT ?", (limit,)
        ).fetchall()
        return [r["user_id"] for r in rows]


if __name__ == "__main__":
    seed(num_users=20)


