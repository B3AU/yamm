import sqlite3
from pathlib import Path

DB_PATH = Path("data/earnings_trades.db")

def migrate_db():
    print(f"Migrating database at {DB_PATH}...")

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Create order_events table
        print("Creating order_events table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Added simple PK
                trade_id TEXT,
                ib_order_id INTEGER,
                ts TEXT, -- Timestamp
                event TEXT, -- placed, status, filled, cancelled, replaced
                status TEXT,
                filled REAL,
                remaining REAL,
                avg_fill_price REAL,
                last_fill_price REAL,
                last_fill_qty REAL,
                limit_price REAL,
                details TEXT -- JSON or free text for extra info
            )
        """)

        # Index for efficient querying by trade_id
        conn.execute("CREATE INDEX IF NOT EXISTS idx_order_events_trade_id ON order_events(trade_id)")

        # 2. Add entry_combo_bid/ask to trades table
        print("Adding combo bid/ask columns to trades table...")
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN entry_combo_bid REAL")
            print("- Added entry_combo_bid")
        except sqlite3.OperationalError:
            print("- entry_combo_bid already exists")

        try:
            conn.execute("ALTER TABLE trades ADD COLUMN entry_combo_ask REAL")
            print("- Added entry_combo_ask")
        except sqlite3.OperationalError:
            print("- entry_combo_ask already exists")

        conn.commit()

    print("Migration complete.")

if __name__ == "__main__":
    migrate_db()
