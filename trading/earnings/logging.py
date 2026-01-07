"""Trade and non-trade logging for earnings volatility strategy.

Implements the logging requirements from V1 plan:
- Every trade: full execution details, model predictions, outcomes
- Every non-trade: rejection reason, counterfactual P&L
- Prevents survivorship bias by logging what we passed on
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


@dataclass
class TradeLog:
    """Complete trade record per V1 plan requirements."""
    # Identification
    trade_id: str
    ticker: str
    earnings_date: str  # YYYY-MM-DD
    earnings_timing: str  # "BMO" or "AMC"

    # Entry execution
    entry_datetime: str
    entry_quoted_bid: float
    entry_quoted_ask: float
    entry_quoted_mid: float
    entry_limit_price: float
    entry_fill_price: Optional[float] = None
    entry_fill_time: Optional[str] = None
    entry_slippage: Optional[float] = None  # fill - mid

    # Structure
    structure: str = "straddle"  # "straddle" or "strangle"
    strikes: str = ""  # JSON list of strikes
    expiration: str = ""  # YYYY-MM-DD
    contracts: int = 0
    premium_paid: float = 0.0
    max_loss: float = 0.0  # = premium_paid for long vol

    # Model predictions
    predicted_q50: Optional[float] = None
    predicted_q75: Optional[float] = None
    predicted_q90: Optional[float] = None
    predicted_q95: Optional[float] = None
    implied_move: Optional[float] = None
    edge_q75: Optional[float] = None  # predicted_q75 - implied_move
    edge_q90: Optional[float] = None

    # Exit execution
    exit_datetime: Optional[str] = None
    exit_quoted_bid: Optional[float] = None
    exit_quoted_ask: Optional[float] = None
    exit_quoted_mid: Optional[float] = None
    exit_limit_price: Optional[float] = None
    exit_fill_price: Optional[float] = None
    exit_slippage: Optional[float] = None

    # Outcomes
    exit_pnl: Optional[float] = None
    exit_pnl_pct: Optional[float] = None  # pnl / premium_paid
    realized_move: Optional[float] = None  # actual |stock move|
    realized_move_pct: Optional[float] = None
    spot_at_entry: Optional[float] = None
    spot_at_exit: Optional[float] = None

    # Status
    status: str = "pending"  # pending, filled, partial, cancelled, exited
    notes: str = ""

    # IBKR order tracking (for recovery after restart)
    call_order_id: Optional[int] = None
    put_order_id: Optional[int] = None

    # Counterfactuals (logged for analysis)
    counterfactual_exit_open_pnl: Optional[float] = None  # if exited at next open
    counterfactual_strangle_pnl: Optional[float] = None  # if used strangle instead


@dataclass
class NonTradeLog:
    """Record of candidates we passed on - critical for survivorship bias."""
    # Identification
    log_id: str
    ticker: str
    earnings_date: str
    earnings_timing: str
    log_datetime: str

    # Why rejected
    rejection_reason: str  # spread_too_wide, oi_too_low, edge_insufficient, position_limit, etc.

    # What we saw
    quoted_bid: Optional[float] = None
    quoted_ask: Optional[float] = None
    quoted_spread_pct: Optional[float] = None
    quoted_oi: Optional[int] = None
    spot_price: Optional[float] = None
    implied_move: Optional[float] = None
    straddle_premium: Optional[float] = None  # Total cost per contract in dollars

    # What model said (if we got that far)
    predicted_q75: Optional[float] = None
    predicted_edge: Optional[float] = None

    # Counterfactual - what would have happened
    counterfactual_realized_move: Optional[float] = None
    counterfactual_pnl: Optional[float] = None  # assuming mid fills
    counterfactual_pnl_with_spread: Optional[float] = None  # realistic fills

    notes: str = ""


@dataclass
class ExecutionMetrics:
    """Aggregated execution quality metrics for Phase 0 validation."""
    # Fill statistics
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    cancelled_orders: int = 0
    fill_rate: float = 0.0

    # Slippage
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0

    # Time to fill
    avg_fill_time_seconds: float = 0.0
    median_fill_time_seconds: float = 0.0

    # By liquidity bucket
    fill_rate_by_oi_bucket: dict = field(default_factory=dict)
    slippage_by_spread_bucket: dict = field(default_factory=dict)


class TradeLogger:
    """SQLite-backed trade logger."""

    def __init__(self, db_path: Path | str = "data/earnings_trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    earnings_date TEXT NOT NULL,
                    earnings_timing TEXT,
                    entry_datetime TEXT,
                    entry_quoted_bid REAL,
                    entry_quoted_ask REAL,
                    entry_quoted_mid REAL,
                    entry_limit_price REAL,
                    entry_fill_price REAL,
                    entry_fill_time TEXT,
                    entry_slippage REAL,
                    structure TEXT,
                    strikes TEXT,
                    expiration TEXT,
                    contracts INTEGER,
                    premium_paid REAL,
                    max_loss REAL,
                    predicted_q50 REAL,
                    predicted_q75 REAL,
                    predicted_q90 REAL,
                    predicted_q95 REAL,
                    implied_move REAL,
                    edge_q75 REAL,
                    edge_q90 REAL,
                    exit_datetime TEXT,
                    exit_quoted_bid REAL,
                    exit_quoted_ask REAL,
                    exit_quoted_mid REAL,
                    exit_limit_price REAL,
                    exit_fill_price REAL,
                    exit_slippage REAL,
                    exit_pnl REAL,
                    exit_pnl_pct REAL,
                    realized_move REAL,
                    realized_move_pct REAL,
                    spot_at_entry REAL,
                    spot_at_exit REAL,
                    status TEXT DEFAULT 'pending',
                    notes TEXT,
                    call_order_id INTEGER,
                    put_order_id INTEGER,
                    counterfactual_exit_open_pnl REAL,
                    counterfactual_strangle_pnl REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add columns if they don't exist (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN call_order_id INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN put_order_id INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Non-trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS non_trades (
                    log_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    earnings_date TEXT NOT NULL,
                    earnings_timing TEXT,
                    log_datetime TEXT,
                    rejection_reason TEXT NOT NULL,
                    quoted_bid REAL,
                    quoted_ask REAL,
                    quoted_spread_pct REAL,
                    quoted_oi INTEGER,
                    spot_price REAL,
                    implied_move REAL,
                    straddle_premium REAL,
                    predicted_q75 REAL,
                    predicted_edge REAL,
                    counterfactual_realized_move REAL,
                    counterfactual_pnl REAL,
                    counterfactual_pnl_with_spread REAL,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add straddle_premium column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE non_trades ADD COLUMN straddle_premium REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(earnings_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nontrades_ticker ON non_trades(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nontrades_reason ON non_trades(rejection_reason)")

            conn.commit()

    def log_trade(self, trade: TradeLog) -> str:
        """Log a trade entry. Returns trade_id."""
        with sqlite3.connect(self.db_path) as conn:
            data = asdict(trade)
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            conn.execute(
                f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            conn.commit()
        return trade.trade_id

    def update_trade(self, trade_id: str, **updates) -> bool:
        """Update specific fields of a trade."""
        if not updates:
            return False

        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
                list(updates.values()) + [trade_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    def log_non_trade(self, non_trade: NonTradeLog) -> str:
        """Log a rejected candidate. Returns log_id."""
        with sqlite3.connect(self.db_path) as conn:
            data = asdict(non_trade)
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            conn.execute(
                f"INSERT INTO non_trades ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            conn.commit()
        return non_trade.log_id

    def update_non_trade(self, log_id: str, **updates) -> bool:
        """Update specific fields of a non-trade record."""
        if not updates:
            return False

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE non_trades SET {set_clause} WHERE log_id = ?",
                list(updates.values()) + [log_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_non_trades_pending_counterfactual(
        self,
        earnings_date: str,
    ) -> list[NonTradeLog]:
        """Get non-trades for a specific earnings date that need counterfactual backfill.

        Returns non-trades where counterfactual_realized_move is NULL.
        """
        query = """
            SELECT * FROM non_trades
            WHERE earnings_date = ?
            AND counterfactual_realized_move IS NULL
            ORDER BY log_datetime DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (earnings_date,)).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                row_dict.pop('created_at', None)
                results.append(NonTradeLog(**row_dict))
            return results

    def get_trade(self, trade_id: str) -> Optional[TradeLog]:
        """Get a trade by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if row:
                return TradeLog(**dict(row))
        return None

    def get_trades(
        self,
        status: Optional[str] = None,
        ticker: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[TradeLog]:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if from_date:
            query += " AND earnings_date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND earnings_date <= ?"
            params.append(to_date)

        query += " ORDER BY entry_datetime DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                # Remove DB-only columns not in dataclass
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)
                results.append(TradeLog(**row_dict))
            return results

    def get_non_trades(
        self,
        reason: Optional[str] = None,
        ticker: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[NonTradeLog]:
        """Query non-trades with optional filters."""
        query = "SELECT * FROM non_trades WHERE 1=1"
        params = []

        if reason:
            query += " AND rejection_reason LIKE ?"
            params.append(f"%{reason}%")
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if from_date:
            query += " AND earnings_date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND earnings_date <= ?"
            params.append(to_date)

        query += " ORDER BY log_datetime DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                # Remove DB-only columns not in dataclass
                row_dict.pop('created_at', None)
                results.append(NonTradeLog(**row_dict))
            return results

    def get_pending_trades_with_orders(self) -> list[TradeLog]:
        """Get trades with pending/partial status that have IBKR order IDs.

        Used for order recovery after daemon restart.
        """
        query = """
            SELECT * FROM trades
            WHERE status IN ('pending', 'partial', 'filled')
            AND (call_order_id IS NOT NULL OR put_order_id IS NOT NULL)
            ORDER BY entry_datetime DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)
                results.append(TradeLog(**row_dict))
            return results

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Compute aggregated execution metrics for Phase 0 validation."""
        trades = self.get_trades()

        if not trades:
            return ExecutionMetrics()

        filled = [t for t in trades if t.status in ("filled", "exited")]
        partial = [t for t in trades if t.status == "partial"]
        cancelled = [t for t in trades if t.status == "cancelled"]

        # Slippage calculations (in basis points)
        slippages = []
        for t in filled:
            if t.entry_slippage is not None and t.entry_quoted_mid:
                slippage_bps = (t.entry_slippage / t.entry_quoted_mid) * 10000
                slippages.append(slippage_bps)

        metrics = ExecutionMetrics(
            total_orders=len(trades),
            filled_orders=len(filled),
            partial_fills=len(partial),
            cancelled_orders=len(cancelled),
            fill_rate=len(filled) / len(trades) if trades else 0,
        )

        if slippages:
            import statistics
            metrics.avg_slippage_bps = statistics.mean(slippages)
            metrics.median_slippage_bps = statistics.median(slippages)
            metrics.max_slippage_bps = max(slippages)

        return metrics

    def get_summary_stats(self) -> dict:
        """Get summary statistics for dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            trades_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            non_trades_count = conn.execute("SELECT COUNT(*) FROM non_trades").fetchone()[0]

            filled = conn.execute(
                "SELECT COUNT(*), SUM(exit_pnl), AVG(exit_pnl_pct) FROM trades WHERE status = 'exited'"
            ).fetchone()

            rejection_reasons = conn.execute(
                "SELECT rejection_reason, COUNT(*) as cnt FROM non_trades GROUP BY rejection_reason ORDER BY cnt DESC"
            ).fetchall()

        return {
            "total_trades": trades_count,
            "total_non_trades": non_trades_count,
            "completed_trades": filled[0] or 0,
            "total_pnl": filled[1] or 0,
            "avg_pnl_pct": filled[2] or 0,
            "rejection_breakdown": dict(rejection_reasons),
        }


def generate_trade_id(ticker: str, earnings_date: str) -> str:
    """Generate unique trade ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{ticker}_{earnings_date}_{timestamp}"


def generate_log_id(ticker: str) -> str:
    """Generate unique log ID for non-trades."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"NT_{ticker}_{timestamp}"
