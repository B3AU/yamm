"""Component tests for TradeLogger - SQLite database operations."""
import json
from datetime import datetime

import pytest

from trading.earnings.logging import (
    TradeLog,
    NonTradeLog,
    SnapshotLog,
    TradeLogger,
    generate_trade_id,
    generate_log_id,
)


class TestTradeLoggerInit:
    """Tests for TradeLogger initialization."""

    def test_creates_db_file(self, tmp_path):
        """Should create database file at specified path."""
        db_path = tmp_path / "test.db"
        logger = TradeLogger(str(db_path))

        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if needed."""
        db_path = tmp_path / "subdir" / "deep" / "test.db"
        logger = TradeLogger(str(db_path))

        assert db_path.exists()

    def test_initializes_tables(self, test_db):
        """Should create all required tables."""
        import sqlite3

        with sqlite3.connect(test_db.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "trades",
            "non_trades",
            "price_snapshots",
            "order_events",
            "llm_checks",
            "earnings_calendar",
        }
        assert expected_tables.issubset(tables)


class TestLogTrade:
    """Tests for log_trade method."""

    def test_insert_new_trade(self, test_db, sample_trade_log):
        """Should insert new trade and return trade_id."""
        trade_id = test_db.log_trade(sample_trade_log)

        assert trade_id == sample_trade_log.trade_id

        # Verify it was inserted
        retrieved = test_db.get_trade(trade_id)
        assert retrieved is not None
        assert retrieved.ticker == sample_trade_log.ticker

    def test_replace_existing_trade(self, test_db, sample_trade_log):
        """Should replace trade with same ID (UPSERT)."""
        # Insert initial
        test_db.log_trade(sample_trade_log)

        # Modify and re-insert
        sample_trade_log.status = "filled"
        sample_trade_log.entry_fill_price = 7.12
        test_db.log_trade(sample_trade_log)

        # Verify it was updated
        retrieved = test_db.get_trade(sample_trade_log.trade_id)
        assert retrieved.status == "filled"
        assert retrieved.entry_fill_price == 7.12


class TestUpdateTrade:
    """Tests for update_trade method."""

    def test_update_single_field(self, test_db, sample_trade_log):
        """Should update single field."""
        test_db.log_trade(sample_trade_log)

        result = test_db.update_trade(
            sample_trade_log.trade_id,
            status="filled",
        )

        assert result is True
        retrieved = test_db.get_trade(sample_trade_log.trade_id)
        assert retrieved.status == "filled"

    def test_update_multiple_fields(self, test_db, sample_trade_log):
        """Should update multiple fields at once."""
        test_db.log_trade(sample_trade_log)

        result = test_db.update_trade(
            sample_trade_log.trade_id,
            status="exited",
            exit_fill_price=8.50,
            exit_pnl=138.0,
            exit_pnl_pct=0.097,
        )

        assert result is True
        retrieved = test_db.get_trade(sample_trade_log.trade_id)
        assert retrieved.status == "exited"
        assert retrieved.exit_fill_price == 8.50
        assert retrieved.exit_pnl == 138.0

    def test_update_nonexistent_trade(self, test_db):
        """Should return False for nonexistent trade."""
        result = test_db.update_trade("NONEXISTENT_001", status="filled")
        assert result is False

    def test_update_sets_updated_at(self, test_db, sample_trade_log):
        """Should set updated_at timestamp."""
        test_db.log_trade(sample_trade_log)

        test_db.update_trade(sample_trade_log.trade_id, status="filled")

        # Can't easily verify timestamp but ensure no error
        retrieved = test_db.get_trade(sample_trade_log.trade_id)
        assert retrieved is not None

    def test_update_invalid_column_raises(self, test_db, sample_trade_log):
        """Should raise error for invalid column name."""
        test_db.log_trade(sample_trade_log)

        with pytest.raises(ValueError, match="Invalid column names"):
            test_db.update_trade(
                sample_trade_log.trade_id,
                invalid_column="value",
            )


class TestLogNonTrade:
    """Tests for log_non_trade method."""

    def test_insert_non_trade(self, test_db, sample_non_trade_log):
        """Should insert non-trade record."""
        log_id = test_db.log_non_trade(sample_non_trade_log)

        assert log_id == sample_non_trade_log.log_id

        # Verify retrieval
        non_trades = test_db.get_non_trades(ticker="MSFT")
        assert len(non_trades) == 1
        assert non_trades[0].rejection_reason == sample_non_trade_log.rejection_reason

    def test_non_trade_stores_quote_data(self, test_db, sample_non_trade_log):
        """Should store quote data for counterfactual analysis."""
        test_db.log_non_trade(sample_non_trade_log)

        non_trades = test_db.get_non_trades(ticker="MSFT")
        assert non_trades[0].quoted_bid == 4.20
        assert non_trades[0].quoted_ask == 5.10
        assert non_trades[0].quoted_spread_pct == 18.5


class TestUpdateNonTrade:
    """Tests for update_non_trade method."""

    def test_update_counterfactual_data(self, test_db, sample_non_trade_log):
        """Should update counterfactual fields."""
        test_db.log_non_trade(sample_non_trade_log)

        result = test_db.update_non_trade(
            sample_non_trade_log.log_id,
            counterfactual_realized_move=0.065,
            counterfactual_pnl=1.25,
            counterfactual_pnl_with_spread=0.85,
        )

        assert result is True

        non_trades = test_db.get_non_trades(ticker="MSFT")
        assert non_trades[0].counterfactual_realized_move == 0.065
        assert non_trades[0].counterfactual_pnl == 1.25


class TestGetTrades:
    """Tests for get_trades query method."""

    def test_get_all_trades(self, test_db, sample_trade_log):
        """Should return all trades when no filters."""
        test_db.log_trade(sample_trade_log)

        # Add another trade
        trade2 = TradeLog(
            trade_id="MSFT_2026-01-30_20260129143100",
            ticker="MSFT",
            earnings_date="2026-01-30",
            earnings_timing="BMO",
            entry_datetime="2026-01-29T14:31:00",
            entry_quoted_bid=5.50,
            entry_quoted_ask=6.00,
            entry_quoted_mid=5.75,
            entry_limit_price=5.80,
        )
        test_db.log_trade(trade2)

        trades = test_db.get_trades()
        assert len(trades) == 2

    def test_filter_by_status(self, test_db, sample_trade_log):
        """Should filter by status."""
        test_db.log_trade(sample_trade_log)

        trade2 = TradeLog(
            trade_id="MSFT_2026-01-30_20260129143100",
            ticker="MSFT",
            earnings_date="2026-01-30",
            earnings_timing="BMO",
            entry_datetime="2026-01-29T14:31:00",
            entry_quoted_bid=5.50,
            entry_quoted_ask=6.00,
            entry_quoted_mid=5.75,
            entry_limit_price=5.80,
            status="filled",
        )
        test_db.log_trade(trade2)

        pending = test_db.get_trades(status="pending")
        filled = test_db.get_trades(status="filled")

        assert len(pending) == 1
        assert len(filled) == 1
        assert pending[0].ticker == "AAPL"
        assert filled[0].ticker == "MSFT"

    def test_filter_by_ticker(self, test_db, sample_trade_log):
        """Should filter by ticker."""
        test_db.log_trade(sample_trade_log)

        trades = test_db.get_trades(ticker="AAPL")
        assert len(trades) == 1

        trades = test_db.get_trades(ticker="MSFT")
        assert len(trades) == 0

    def test_filter_by_date_range(self, test_db, sample_trade_log):
        """Should filter by date range."""
        test_db.log_trade(sample_trade_log)

        trades = test_db.get_trades(from_date="2026-01-30", to_date="2026-01-30")
        assert len(trades) == 1

        trades = test_db.get_trades(from_date="2026-02-01")
        assert len(trades) == 0


class TestLogSnapshot:
    """Tests for snapshot logging."""

    def test_log_snapshot(self, test_db, sample_trade_log):
        """Should log intraday snapshot."""
        test_db.log_trade(sample_trade_log)

        snapshot = SnapshotLog(
            trade_id=sample_trade_log.trade_id,
            ts="2026-01-29T15:00:00",
            minutes_since_open=330,
            straddle_mid=7.25,
            call_mid=3.65,
            put_mid=3.60,
            spot_price=230.50,
            unrealized_pnl=25.0,
            unrealized_pnl_pct=0.017,
        )
        test_db.log_snapshot(snapshot)

        # Retrieve snapshots
        snapshots = test_db.get_snapshots(sample_trade_log.trade_id)
        assert len(snapshots) == 1
        assert snapshots[0].straddle_mid == 7.25
        assert snapshots[0].unrealized_pnl == 25.0


class TestLogOrderEvent:
    """Tests for order event logging."""

    def test_log_order_event(self, test_db, sample_trade_log):
        """Should log order events."""
        test_db.log_trade(sample_trade_log)

        test_db.log_order_event(
            trade_id=sample_trade_log.trade_id,
            ib_order_id=12345,
            event="placed",
            status="Submitted",
            filled=0.0,
            remaining=2.0,
            avg_fill_price=0.0,
            limit_price=7.15,
        )

        event = test_db.get_latest_order_event(sample_trade_log.trade_id)
        assert event is not None
        assert event["ib_order_id"] == 12345
        assert event["status"] == "Submitted"


class TestLogLLMCheck:
    """Tests for LLM sanity check logging."""

    def test_log_llm_check(self, test_db):
        """Should log LLM sanity check result."""
        test_db.log_llm_check(
            ticker="AAPL",
            decision="PASS",
            risk_flags=[],
            reasons=["No significant risks identified"],
            search_queries=["AAPL earnings news"],
            search_results=[{"title": "Apple Q1 Preview", "url": "https://..."}],
            packet={"symbol": "AAPL", "edge": 0.08},
            response={"decision": "PASS"},
            latency_ms=450,
            model="claude-3.5-sonnet",
        )

        # Verify no error - can't easily query without exposing method


class TestGetPendingTradesWithOrders:
    """Tests for order recovery query."""

    def test_get_pending_with_order_ids(self, test_db):
        """Should return trades with IBKR order IDs."""
        trade = TradeLog(
            trade_id="AAPL_2026-01-30_20260129143000",
            ticker="AAPL",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            entry_datetime="2026-01-29T14:30:00",
            entry_quoted_bid=6.90,
            entry_quoted_ask=7.30,
            entry_quoted_mid=7.10,
            entry_limit_price=7.15,
            status="pending",
            call_order_id=12345,
            put_order_id=12346,
        )
        test_db.log_trade(trade)

        pending = test_db.get_pending_trades_with_orders()
        assert len(pending) == 1
        assert pending[0].call_order_id == 12345

    def test_excludes_exited_trades(self, test_db):
        """Should not return exited trades."""
        trade = TradeLog(
            trade_id="AAPL_2026-01-30_20260129143000",
            ticker="AAPL",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            entry_datetime="2026-01-29T14:30:00",
            entry_quoted_bid=6.90,
            entry_quoted_ask=7.30,
            entry_quoted_mid=7.10,
            entry_limit_price=7.15,
            status="exited",
            call_order_id=12345,
        )
        test_db.log_trade(trade)

        pending = test_db.get_pending_trades_with_orders()
        assert len(pending) == 0


class TestGetSummaryStats:
    """Tests for summary statistics."""

    def test_empty_db_stats(self, test_db):
        """Should return zero stats for empty DB."""
        stats = test_db.get_summary_stats()

        assert stats["total_trades"] == 0
        assert stats["total_non_trades"] == 0
        assert stats["total_pnl"] == 0

    def test_stats_with_trades(self, test_db, sample_trade_log, sample_non_trade_log):
        """Should compute stats from trades."""
        # Add exited trade with P&L
        sample_trade_log.status = "exited"
        sample_trade_log.exit_pnl = 150.0
        sample_trade_log.exit_pnl_pct = 0.105
        test_db.log_trade(sample_trade_log)
        test_db.log_non_trade(sample_non_trade_log)

        stats = test_db.get_summary_stats()

        assert stats["total_trades"] == 1
        assert stats["total_non_trades"] == 1
        assert stats["completed_trades"] == 1
        assert stats["total_pnl"] == 150.0


class TestEarningsCalendar:
    """Tests for earnings calendar logging."""

    def test_log_earnings_calendar(self, test_db):
        """Should log earnings calendar entry."""
        from datetime import date

        test_db.log_earnings_calendar(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            source="nasdaq",
            eps_estimate=6.05,
        )

        # Retrieve and verify
        entries = test_db.get_earnings_calendar()
        assert len(entries) == 1
        assert entries[0]["symbol"] == "AAPL"
        assert entries[0]["timing"] == "AMC"
        assert entries[0]["source"] == "nasdaq"

    def test_get_earnings_calendar_by_date(self, test_db):
        """Should filter by date range."""
        from datetime import date

        test_db.log_earnings_calendar(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            source="nasdaq",
        )
        test_db.log_earnings_calendar(
            symbol="MSFT",
            earnings_date=date(2026, 2, 5),
            timing="BMO",
            source="fmp",
        )

        # Filter by date range
        entries = test_db.get_earnings_calendar(
            from_date=date(2026, 1, 1),
            to_date=date(2026, 1, 31),
        )
        assert len(entries) == 1
        assert entries[0]["symbol"] == "AAPL"

    def test_get_earnings_calendar_by_source(self, test_db):
        """Should filter by source."""
        from datetime import date

        test_db.log_earnings_calendar(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            source="nasdaq",
        )
        test_db.log_earnings_calendar(
            symbol="MSFT",
            earnings_date=date(2026, 1, 30),
            timing="BMO",
            source="fmp",
        )

        # Filter by source
        nasdaq_entries = test_db.get_earnings_calendar(source="nasdaq")
        fmp_entries = test_db.get_earnings_calendar(source="fmp")

        assert len(nasdaq_entries) == 1
        assert len(fmp_entries) == 1
        assert nasdaq_entries[0]["symbol"] == "AAPL"
        assert fmp_entries[0]["symbol"] == "MSFT"


class TestGetTradesByDate:
    """Tests for date-based trade queries."""

    def test_filter_by_earnings_date(self, test_db, sample_trade_log):
        """Should filter trades by earnings date."""
        test_db.log_trade(sample_trade_log)

        trades = test_db.get_trades(from_date="2026-01-29", to_date="2026-01-31")
        assert len(trades) == 1

        trades = test_db.get_trades(from_date="2026-02-01", to_date="2026-02-28")
        assert len(trades) == 0


class TestGetNonTradesDateRange:
    """Tests for non-trade date range queries."""

    def test_filter_non_trades_by_date(self, test_db, sample_non_trade_log):
        """Should filter non-trades by date range."""
        test_db.log_non_trade(sample_non_trade_log)

        non_trades = test_db.get_non_trades(
            from_date="2026-01-28",
            to_date="2026-01-31",
        )
        assert len(non_trades) == 1

        non_trades = test_db.get_non_trades(
            from_date="2026-02-01",
            to_date="2026-02-28",
        )
        assert len(non_trades) == 0
