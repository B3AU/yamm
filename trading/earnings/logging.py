"""Trade and non-trade logging for earnings volatility strategy.

Implements the logging requirements from V1 plan:
- Every trade: full execution details, model predictions, outcomes
- Every non-trade: rejection reason, counterfactual P&L
- Prevents survivorship bias by logging what we passed on
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
    entry_combo_bid: Optional[float] = None
    entry_combo_ask: Optional[float] = None
    entry_fill_price: Optional[float] = None
    entry_fill_time: Optional[str] = None
    entry_slippage: Optional[float] = None  # fill - mid

    # Advanced Execution Metrics (New)
    decision_latency_ms: Optional[float] = None  # ms between quote snapshot and order submit
    fill_latency_seconds: Optional[float] = None # Time from submit to full fill
    spread_at_fill: Optional[float] = None # live spread when filled (if available)

    # Post-fill Markouts (price behavior after entry)
    markout_1min: Optional[float] = None  # Price 1 min after fill
    markout_5min: Optional[float] = None  # Price 5 min after fill
    markout_30min: Optional[float] = None # Price 30 min after fill

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
    exit_call_order_id: Optional[int] = None
    exit_put_order_id: Optional[int] = None

    # Counterfactuals (logged for analysis)
    counterfactual_exit_open_pnl: Optional[float] = None  # if exited at next open
    counterfactual_strangle_pnl: Optional[float] = None  # if used strangle instead

    # News data tracking
    news_count: Optional[int] = None  # number of FMP news articles found


# Allowed column names for SQL updates (security: prevent injection via **kwargs)
TRADE_COLUMNS = {
    'trade_id', 'ticker', 'earnings_date', 'earnings_timing',
    'entry_datetime', 'entry_quoted_bid', 'entry_quoted_ask', 'entry_quoted_mid',
    'entry_limit_price', 'entry_combo_bid', 'entry_combo_ask', 'entry_fill_price',
    'entry_fill_time', 'entry_slippage', 'decision_latency_ms', 'fill_latency_seconds',
    'spread_at_fill', 'markout_1min', 'markout_5min', 'markout_30min',
    'structure', 'strikes', 'expiration', 'contracts', 'premium_paid', 'max_loss',
    'predicted_q50', 'predicted_q75', 'predicted_q90', 'predicted_q95',
    'implied_move', 'edge_q75', 'edge_q90',
    'exit_datetime', 'exit_quoted_bid', 'exit_quoted_ask', 'exit_quoted_mid',
    'exit_limit_price', 'exit_fill_price', 'exit_slippage',
    'exit_pnl', 'exit_pnl_pct', 'realized_move', 'realized_move_pct',
    'spot_at_entry', 'spot_at_exit', 'status', 'notes',
    'call_order_id', 'put_order_id', 'exit_call_order_id', 'exit_put_order_id',
    'counterfactual_exit_open_pnl', 'counterfactual_strangle_pnl',
    'news_count', 'updated_at',
}

NON_TRADE_COLUMNS = {
    'log_id', 'ticker', 'earnings_date', 'earnings_timing', 'log_datetime',
    'rejection_reason', 'quoted_bid', 'quoted_ask', 'quoted_spread_pct',
    'quoted_oi', 'spot_price', 'implied_move', 'straddle_premium',
    'predicted_q75', 'predicted_edge',
    'counterfactual_realized_move', 'counterfactual_pnl', 'counterfactual_pnl_with_spread',
    'notes',
}


@dataclass
class SnapshotLog:
    """Intraday price snapshot for an open position."""
    trade_id: str
    ts: str  # ISO timestamp
    minutes_since_open: int  # Minutes since 9:30 ET market open
    straddle_mid: float  # Call mid + Put mid
    call_mid: Optional[float] = None
    put_mid: Optional[float] = None
    spot_price: Optional[float] = None  # Underlying stock price
    unrealized_pnl: Optional[float] = None  # (straddle_mid - entry_fill_price) * contracts * 100
    unrealized_pnl_pct: Optional[float] = None  # straddle_mid / entry_fill_price - 1


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

    # Default timeout for SQLite connections (seconds)
    # Higher than default 5s to handle concurrent access from daemon
    DB_TIMEOUT = 30.0

    def __init__(self, db_path: Path | str = "data/earnings_trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
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
                    markout_1min REAL,
                    markout_5min REAL,
                    markout_30min REAL,
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
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN exit_call_order_id INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN exit_put_order_id INTEGER")
            except sqlite3.OperationalError:
                pass

            # Add markout columns (migration)
            for col in ['markout_1min', 'markout_5min', 'markout_30min']:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL")
                except sqlite3.OperationalError:
                    pass

            # Add advanced execution metrics (migration)
            for col in ['decision_latency_ms', 'fill_latency_seconds', 'spread_at_fill']:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL")
                except sqlite3.OperationalError:
                    pass

            # Add news_count column to track trades with/without news data (migration)
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN news_count INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add combo bid/ask columns for tracking combined straddle pricing (migration)
            for col in ['entry_combo_bid', 'entry_combo_ask']:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL")
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

            # Price snapshots table (intraday position tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    minutes_since_open INTEGER,
                    straddle_mid REAL,
                    call_mid REAL,
                    put_mid REAL,
                    spot_price REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_trade ON price_snapshots(trade_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON price_snapshots(ts)")

            # Order events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    ib_order_id INTEGER,
                    ts TEXT,
                    event TEXT,
                    status TEXT,
                    filled REAL,
                    remaining REAL,
                    avg_fill_price REAL,
                    last_fill_price REAL,
                    last_fill_qty REAL,
                    limit_price REAL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_events_trade_id ON order_events(trade_id)")

            # LLM sanity checks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    trade_id TEXT,
                    decision TEXT NOT NULL,
                    risk_flags TEXT,
                    reasons TEXT,
                    search_queries TEXT,
                    search_results TEXT,
                    packet_json TEXT,
                    response_json TEXT,
                    latency_ms INTEGER,
                    model TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_checks_ticker ON llm_checks(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_checks_decision ON llm_checks(decision)")

            # Earnings calendar table (multi-source)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    earnings_date TEXT NOT NULL,
                    timing TEXT,
                    eps_estimate REAL,
                    revenue_estimate REAL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, earnings_date, source)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_date ON earnings_calendar(earnings_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_symbol ON earnings_calendar(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calendar_source ON earnings_calendar(source)")

            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(earnings_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nontrades_ticker ON non_trades(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nontrades_reason ON non_trades(rejection_reason)")

            conn.commit()

    def log_trade(self, trade: TradeLog) -> str:
        """Log a trade entry. Returns trade_id."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            data = asdict(trade)
            # Validate column names against whitelist (security)
            invalid_cols = set(data.keys()) - TRADE_COLUMNS
            if invalid_cols:
                raise ValueError(f"Invalid column names for trades: {invalid_cols}")
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

        # Validate column names against whitelist (security)
        invalid_cols = set(updates.keys()) - TRADE_COLUMNS
        if invalid_cols:
            raise ValueError(f"Invalid column names for trades: {invalid_cols}")

        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            cursor = conn.execute(
                f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
                list(updates.values()) + [trade_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    def log_non_trade(self, non_trade: NonTradeLog) -> str:
        """Log a rejected candidate. Returns log_id."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            data = asdict(non_trade)
            # Validate column names against whitelist (security)
            invalid_cols = set(data.keys()) - NON_TRADE_COLUMNS
            if invalid_cols:
                raise ValueError(f"Invalid column names for non_trades: {invalid_cols}")
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

        # Validate column names against whitelist (security)
        invalid_cols = set(updates.keys()) - NON_TRADE_COLUMNS
        if invalid_cols:
            raise ValueError(f"Invalid column names for non_trades: {invalid_cols}")

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            cursor = conn.execute(
                f"UPDATE non_trades SET {set_clause} WHERE log_id = ?",
                list(updates.values()) + [log_id]
            )
            conn.commit()
            return cursor.rowcount > 0

    def log_order_event(
        self,
        trade_id: str,
        ib_order_id: int,
        event: str,
        status: str,
        filled: float,
        remaining: float,
        avg_fill_price: float,
        limit_price: float,
        last_fill_price: float = 0.0,
        last_fill_qty: float = 0.0,
        details: dict = None
    ):
        """Log a granular order event (placed, status update, fill, etc)."""
        ts = datetime.now().isoformat()
        details_json = json.dumps(details) if details else None

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.execute("""
                INSERT INTO order_events (
                    trade_id, ib_order_id, ts, event, status,
                    filled, remaining, avg_fill_price,
                    last_fill_price, last_fill_qty, limit_price, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, ib_order_id, ts, event, status,
                filled, remaining, avg_fill_price,
                last_fill_price, last_fill_qty, limit_price, details_json
            ))
            conn.commit()

    def log_snapshot(self, snapshot: SnapshotLog) -> None:
        """Log an intraday price snapshot for a position."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.execute("""
                INSERT INTO price_snapshots (
                    trade_id, ts, minutes_since_open, straddle_mid,
                    call_mid, put_mid, spot_price,
                    unrealized_pnl, unrealized_pnl_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.trade_id, snapshot.ts, snapshot.minutes_since_open,
                snapshot.straddle_mid, snapshot.call_mid, snapshot.put_mid,
                snapshot.spot_price, snapshot.unrealized_pnl, snapshot.unrealized_pnl_pct
            ))
            conn.commit()

    def log_llm_check(
        self,
        ticker: str,
        decision: str,
        risk_flags: list,
        reasons: list,
        search_queries: list,
        search_results: list,
        packet: dict,
        response: dict,
        latency_ms: int,
        model: str,
        trade_id: str = None,
    ) -> None:
        """Log an LLM sanity check result."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.execute("""
                INSERT INTO llm_checks (
                    ts, ticker, trade_id, decision, risk_flags, reasons,
                    search_queries, search_results, packet_json, response_json,
                    latency_ms, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                ticker,
                trade_id,
                decision,
                json.dumps(risk_flags),
                json.dumps(reasons),
                json.dumps(search_queries),
                json.dumps(search_results[:5]),  # Limit stored results
                json.dumps(packet),
                json.dumps(response),
                latency_ms,
                model,
            ))
            conn.commit()

    def log_earnings_calendar(
        self,
        symbol: str,
        earnings_date,  # date or str
        timing: str,
        source: str,
        eps_estimate: float = None,
        revenue_estimate: float = None,
    ) -> None:
        """Log an earnings calendar entry from a specific source."""
        # Convert date to string if needed
        if hasattr(earnings_date, 'isoformat'):
            earnings_date = earnings_date.isoformat()

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO earnings_calendar (
                    symbol, earnings_date, timing, eps_estimate, revenue_estimate,
                    source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                earnings_date,
                timing,
                eps_estimate,
                revenue_estimate,
                source,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def get_earnings_calendar(self, from_date=None, to_date=None, source: str = None) -> list[dict]:
        """Get earnings calendar entries, optionally filtered by date range and source."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM earnings_calendar WHERE 1=1"
            params = []

            if from_date:
                if hasattr(from_date, 'isoformat'):
                    from_date = from_date.isoformat()
                query += " AND earnings_date >= ?"
                params.append(from_date)

            if to_date:
                if hasattr(to_date, 'isoformat'):
                    to_date = to_date.isoformat()
                query += " AND earnings_date <= ?"
                params.append(to_date)

            if source:
                query += " AND source = ?"
                params.append(source)

            query += " ORDER BY earnings_date, symbol"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_snapshots(self, trade_id: str) -> list[SnapshotLog]:
        """Get all snapshots for a trade, ordered by time."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM price_snapshots WHERE trade_id = ? ORDER BY ts",
                (trade_id,)
            ).fetchall()
            results = []
            for row in rows:
                results.append(SnapshotLog(
                    trade_id=row['trade_id'],
                    ts=row['ts'],
                    minutes_since_open=row['minutes_since_open'],
                    straddle_mid=row['straddle_mid'],
                    call_mid=row['call_mid'],
                    put_mid=row['put_mid'],
                    spot_price=row['spot_price'],
                    unrealized_pnl=row['unrealized_pnl'],
                    unrealized_pnl_pct=row['unrealized_pnl_pct'],
                ))
            return results

    def get_latest_order_event(self, trade_id: str) -> Optional[dict]:
        """Get the latest order event for a trade."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM order_events WHERE trade_id = ? ORDER BY event_id DESC LIMIT 1",
                (trade_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_llm_check_for_trade(self, ticker: str, entry_datetime: str) -> Optional[dict]:
        """Get most recent LLM check for a ticker before entry time."""
        # Convert entry_datetime to UTC for proper comparison with llm_checks.ts (stored in UTC)
        entry_datetime_utc = entry_datetime
        if entry_datetime:
            try:
                dt = datetime.fromisoformat(entry_datetime)
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc)
                entry_datetime_utc = dt.isoformat()
            except (ValueError, TypeError):
                pass  # Keep original if parsing fails

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM llm_checks
                WHERE ticker = ? AND ts <= ?
                ORDER BY ts DESC LIMIT 1
            """, (ticker, entry_datetime_utc)).fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                result['risk_flags'] = json.loads(result['risk_flags'] or '[]')
                result['reasons'] = json.loads(result['reasons'] or '[]')
                result['search_queries'] = json.loads(result['search_queries'] or '[]')
                return result
        return None

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
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
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
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if row:
                row_dict = dict(row)
                # Remove DB-only columns not in dataclass
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)
                # Filter to only include keys that match TradeLog fields
                valid_fields = set(TradeLog.__dataclass_fields__.keys())
                filtered_dict = {k: v for k, v in row_dict.items() if k in valid_fields}
                return TradeLog(**filtered_dict)
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

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                # Remove DB-only columns not in dataclass
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)

                # IMPORTANT: Handle dynamically added columns (like entry_combo_bid/ask)
                # that might exist in DB but aren't in TradeLog dataclass yet if old code runs
                # Or vice versa: if dataclass has fields but DB row doesn't (handled by dataclass defaults)

                # Filter row_dict to only include keys that match TradeLog fields
                # This makes the logger robust to schema drifts
                valid_fields = set(TradeLog.__dataclass_fields__.keys())
                filtered_dict = {k: v for k, v in row_dict.items() if k in valid_fields}

                results.append(TradeLog(**filtered_dict))
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

        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
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
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
            results = []
            for row in rows:
                row_dict = dict(row)
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)

                valid_fields = set(TradeLog.__dataclass_fields__.keys())
                filtered_dict = {k: v for k, v in row_dict.items() if k in valid_fields}

                results.append(TradeLog(**filtered_dict))
            return results

    def cleanup_stale_orders(self, max_age_hours: int = 24) -> list[str]:
        """Cancel trades that are stale (pending/submitted for too long without fills).

        This prevents orphan trades from polluting metrics. Trades are marked as
        'cancelled' with a note explaining they were auto-cleaned.

        Args:
            max_age_hours: Orders older than this (since entry_datetime) are cleaned up

        Returns:
            List of trade_ids that were cancelled
        """
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        # Find stale trades: pending/submitted status, no fill, older than cutoff
        query = """
            SELECT trade_id, ticker, entry_datetime, status
            FROM trades
            WHERE status IN ('pending', 'submitted')
            AND entry_fill_price IS NULL
            AND entry_datetime < ?
        """

        cancelled_ids = []
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (cutoff,)).fetchall()

            for row in rows:
                trade_id = row['trade_id']
                ticker = row['ticker']
                entry_dt = row['entry_datetime']

                # Mark as cancelled
                conn.execute(
                    """UPDATE trades
                    SET status = 'cancelled',
                        notes = COALESCE(notes || '; ', '') || 'Auto-cancelled: stale unfilled order',
                        updated_at = ?
                    WHERE trade_id = ?""",
                    (datetime.now().isoformat(), trade_id)
                )
                cancelled_ids.append(trade_id)

            conn.commit()

        return cancelled_ids

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Compute aggregated execution metrics for Phase 0 validation."""
        trades = self.get_trades()

        if not trades:
            return ExecutionMetrics()

        filled = [t for t in trades if t.status in ("filled", "exited")]
        partial = [t for t in trades if t.status == "partial"]
        cancelled = [t for t in trades if t.status == "cancelled"]
        # Exclude cancelled from fill rate calculation (they weren't real order attempts)
        attempted = [t for t in trades if t.status != "cancelled"]

        # Slippage calculations (in basis points)
        slippages = []
        for t in filled:
            if t.entry_slippage is not None and t.entry_quoted_mid:
                slippage_bps = (t.entry_slippage / t.entry_quoted_mid) * 10000
                slippages.append(slippage_bps)

        metrics = ExecutionMetrics(
            total_orders=len(attempted),
            filled_orders=len(filled),
            partial_fills=len(partial),
            cancelled_orders=len(cancelled),
            fill_rate=len(filled) / len(attempted) if attempted else 0,
        )

        if slippages:
            import statistics
            metrics.avg_slippage_bps = statistics.mean(slippages)
            metrics.median_slippage_bps = statistics.median(slippages)
            metrics.max_slippage_bps = max(slippages)

        return metrics

    def get_summary_stats(self) -> dict:
        """Get summary statistics for dashboard."""
        with sqlite3.connect(self.db_path, timeout=self.DB_TIMEOUT) as conn:
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
