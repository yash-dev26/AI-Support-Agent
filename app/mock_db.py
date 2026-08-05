"""
Mock commerce layer. No storefront, no frontend — just enough structured data
for tools to query against, so the agent can answer "what's in my cart" /
"where's my order" style questions realistically.

SQLite so there's zero setup cost. Swap for Postgres later if you want to
show connection pooling, but it's not the point of this project.
"""
import sqlite3
import random
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "commerce.db"

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Isha", "Kabir", "Meera"]
PRODUCTS = [
    ("Wireless Mouse", 799), ("Mechanical Keyboard", 3499), ("USB-C Hub", 1299),
    ("Laptop Stand", 1899), ("Noise Cancelling Headphones", 6999), ("Webcam 1080p", 2199),
    ("Portable SSD 1TB", 5499), ("Monitor Arm", 2799),
]
ORDER_STATUSES = ["delivered", "shipped", "processing", "cancelled", "refunded"]


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


def seed(num_users: int = 20, force: bool = False):
    init_schema()
    with get_conn() as conn:
        if not force:
            existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            if existing > 0:
                print(f"DB already seeded ({existing} users). Use force=True to reseed.")
                return
        conn.executescript("DELETE FROM cart_items; DELETE FROM orders; DELETE FROM users;")

        for i in range(1, num_users + 1):
            user_id = f"user_{i:03d}"
            name = f"{random.choice(FIRST_NAMES)} {i}"
            conn.execute(
                "INSERT INTO users (user_id, name, email) VALUES (?, ?, ?)",
                (user_id, name, f"{user_id}@example.com"),
            )

            for o in range(random.randint(1, 4)):
                product, price = random.choice(PRODUCTS)
                order_id = f"ord_{user_id}_{o}"
                conn.execute(
                    """INSERT INTO orders (order_id, user_id, product_name, amount_cents, status, placed_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now', ?))""",
                    (order_id, user_id, product, price * 100, random.choice(ORDER_STATUSES),
                     f"-{random.randint(1, 60)} days"),
                )

            for c in range(random.randint(0, 3)):
                product, price = random.choice(PRODUCTS)
                conn.execute(
                    """INSERT INTO cart_items (user_id, product_name, quantity, amount_cents)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, product, random.randint(1, 2), price * 100),
                )
        conn.commit()
    print(f"Seeded {num_users} users with orders and cart items.")


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
            """SELECT order_id, product_name, amount_cents, status, placed_at
               FROM orders WHERE user_id = ? ORDER BY placed_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_order_status(order_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT order_id, status, product_name FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        return dict(row) if row else None


if __name__ == "__main__":
    seed(num_users=20)
