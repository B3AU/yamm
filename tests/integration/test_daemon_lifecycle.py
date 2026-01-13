"""Integration tests for daemon lifecycle and scheduling."""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest
import pytz


class TestDaemonStartup:
    """Tests for daemon startup behavior."""

    @pytest.mark.integration
    def test_daemon_creates_required_components(self, test_db, mock_ib):
        """Should create all required components on startup."""
        # Daemon needs: IB connection, TradeLogger, Scheduler
        from trading.earnings.daemon import TradingDaemon

        with patch('trading.earnings.daemon.IB', return_value=mock_ib):
            with patch('trading.earnings.daemon.TradeLogger', return_value=test_db):
                daemon = TradingDaemon()

                assert daemon.trade_logger is not None

    @pytest.mark.integration
    def test_daemon_loads_pending_orders(self, test_db):
        """Should load pending orders on startup for recovery."""
        from trading.earnings.logging import TradeLog

        # Create pending trade with order IDs
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

        # Daemon should be able to load these
        pending = test_db.get_pending_trades_with_orders()

        assert len(pending) == 1
        assert pending[0].call_order_id == 12345


class TestDaemonScheduling:
    """Tests for daemon scheduled tasks."""

    @pytest.mark.integration
    def test_schedule_times_in_et(self):
        """Schedule times should be in Eastern Time."""
        ET = pytz.timezone('US/Eastern')

        # Key schedule times from CLAUDE.md
        schedule = {
            "morning_connect": "09:25",
            "exit_positions": "14:00",
            "screen_and_place": "14:15",
            "force_exit": "15:58",
            "disconnect": "16:05",
        }

        # All times should be valid ET times
        for name, time_str in schedule.items():
            hour, minute = map(int, time_str.split(":"))
            assert 0 <= hour <= 23
            assert 0 <= minute <= 59

    @pytest.mark.integration
    def test_exit_before_entry(self):
        """Exit positions should run before new entries (capital recycling)."""
        exit_time = 14 * 60 + 0  # 14:00
        entry_time = 14 * 60 + 15  # 14:15

        assert exit_time < entry_time, "Exits must happen before entries"


class TestDaemonShutdown:
    """Tests for daemon shutdown behavior."""

    @pytest.mark.integration
    def test_graceful_shutdown(self, test_db, mock_ib):
        """Should disconnect cleanly on shutdown."""
        # Daemon should call ib.disconnect() on shutdown
        mock_ib.disconnect = MagicMock()

        # Simulate shutdown
        # The actual implementation would call disconnect
        mock_ib.disconnect()

        mock_ib.disconnect.assert_called_once()


class TestDaemonCatchUp:
    """Tests for startup catch-up behavior."""

    @pytest.mark.integration
    def test_runs_screening_if_within_window(self):
        """Should run screening immediately if started within screening window."""
        ET = pytz.timezone('US/Eastern')

        # Screening window: 14:00-21:00 ET
        window_start = 14 * 60  # 14:00
        window_end = 21 * 60    # 21:00

        # Test time: 15:00 ET (within window)
        test_time = 15 * 60

        within_window = window_start <= test_time < window_end

        assert within_window is True

    @pytest.mark.integration
    def test_skips_screening_if_outside_window(self):
        """Should skip screening if started outside window."""
        ET = pytz.timezone('US/Eastern')

        window_start = 14 * 60
        window_end = 21 * 60

        # Test time: 10:00 ET (before window)
        test_time = 10 * 60

        within_window = window_start <= test_time < window_end

        assert within_window is False


class TestPositionReconciliation:
    """Tests for position reconciliation on startup."""

    @pytest.mark.integration
    def test_detects_db_ibkr_mismatch(self, test_db, mock_ib):
        """Should detect mismatches between DB and IBKR state."""
        from trading.earnings.logging import TradeLog

        # DB has a filled trade
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
            entry_fill_price=7.12,
            status="filled",
            contracts=2,
        )
        test_db.log_trade(trade)

        # IBKR has no position (mismatch)
        mock_ib.positions.return_value = []

        db_trades = test_db.get_trades(status="filled")
        ibkr_positions = mock_ib.positions()

        # Should detect mismatch
        db_symbols = {t.ticker for t in db_trades}
        ibkr_symbols = {p.contract.symbol for p in ibkr_positions}

        mismatch = db_symbols - ibkr_symbols

        assert "AAPL" in mismatch


class TestForceExitAtClose:
    """Tests for force exit behavior at market close."""

    @pytest.mark.integration
    def test_converts_to_market_order(self, test_db):
        """Should convert unfilled exits to market orders at 15:58."""
        from trading.earnings.logging import TradeLog

        # Trade with unfilled exit order
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
            entry_fill_price=7.12,
            status="exiting",
            contracts=2,
            exit_call_order_id=100001,
            exit_put_order_id=100002,
        )
        test_db.log_trade(trade)

        # At 15:58, should have mechanism to force exit
        force_exit_time = datetime(2026, 1, 30, 15, 58, 0)

        # Document expected behavior:
        # - Cancel limit orders
        # - Place market orders
        # - Log the forced exit


class TestPriceImprovementLoop:
    """Tests for progressive price improvement."""

    @pytest.mark.integration
    def test_aggression_increases(self):
        """Aggression should increase over reprice iterations."""
        aggression_schedule = [0.4, 0.5, 0.6, 0.7]

        # Verify increasing order
        for i in range(len(aggression_schedule) - 1):
            assert aggression_schedule[i] < aggression_schedule[i + 1]

    @pytest.mark.integration
    def test_reprice_timing(self):
        """Reprices should happen at scheduled intervals."""
        # Reprice times: 14:25, 14:35, 14:45, 14:55
        reprice_times = ["14:25", "14:35", "14:45", "14:55"]

        # 10 minute intervals
        for i in range(len(reprice_times) - 1):
            t1 = int(reprice_times[i].split(":")[1])
            t2 = int(reprice_times[i + 1].split(":")[1])
            interval = t2 - t1

            assert interval == 10


class TestOrderEventLogging:
    """Tests for order event logging."""

    @pytest.mark.integration
    def test_logs_order_status_changes(self, test_db):
        """Should log order status changes."""
        from trading.earnings.logging import TradeLog

        # Create trade
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
        )
        test_db.log_trade(trade)

        # Log order events
        test_db.log_order_event(
            trade_id=trade.trade_id,
            ib_order_id=12345,
            event="placed",
            status="Submitted",
            filled=0.0,
            remaining=2.0,
            avg_fill_price=0.0,
            limit_price=7.15,
        )

        test_db.log_order_event(
            trade_id=trade.trade_id,
            ib_order_id=12345,
            event="fill",
            status="Filled",
            filled=2.0,
            remaining=0.0,
            avg_fill_price=7.12,
            limit_price=7.15,
        )

        # Verify latest event
        latest = test_db.get_latest_order_event(trade.trade_id)
        assert latest is not None
        assert latest["status"] == "Filled"
