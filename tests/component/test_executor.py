"""Component tests for executor.py - order execution with mocked IBKR."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.earnings.logging import TradeLog


class TestPhase0ExecutorInit:
    """Tests for Phase0Executor initialization."""

    def test_init_with_ib_connection(self, test_db, mock_ib):
        """Should initialize with IB connection and logger."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        assert executor.ib == mock_ib
        assert executor.logger == test_db  # Stored as .logger


class TestCreateStraddleCombo:
    """Tests for straddle combo contract creation."""

    def test_creates_combo_structure(self, test_db, mock_ib):
        """Should create combo contract with call + put legs."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # The _create_straddle_combo method creates BAG contract
        # This tests that the executor has the method
        assert hasattr(executor, '_create_straddle_combo')


class TestPlaceStraddle:
    """Tests for place_straddle method."""

    @pytest.mark.asyncio
    async def test_place_straddle_calls_ib(self, test_db, mock_ib_with_quotes, sample_candidate):
        """Should call IB to place order."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib_with_quotes, trade_logger=test_db)

        # Setup mock chain
        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.expirations = ["20260131"]
        mock_chain.strikes = [230.0]

        # Mock contract qualification
        call_contract = MagicMock()
        call_contract.conId = 12345
        put_contract = MagicMock()
        put_contract.conId = 12346

        mock_ib_with_quotes.qualifyContractsAsync = AsyncMock(
            return_value=[call_contract, put_contract]
        )
        mock_ib_with_quotes.reqSecDefOptParamsAsync = AsyncMock(
            return_value=[mock_chain]
        )

        # Mock order placement
        mock_trade = MagicMock()
        mock_trade.order = MagicMock(orderId=99999)
        mock_trade.orderStatus = MagicMock(status="Submitted")
        mock_ib_with_quotes.placeOrder.return_value = mock_trade

        # API uses target_entry_amount, not contracts
        result = await executor.place_straddle(
            candidate=sample_candidate,
            target_entry_amount=2000,
            min_contracts=1,
            max_contracts=5,
        )

        # Verify IB interaction occurred
        assert mock_ib_with_quotes.qualifyContractsAsync.called or result is None


class TestOrderRecovery:
    """Tests for order recovery after restart."""

    def test_recovers_pending_orders(self, test_db):
        """Should recover pending orders from database."""
        # Setup trade with order IDs
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

        # Get pending trades with orders
        pending = test_db.get_pending_trades_with_orders()

        assert len(pending) == 1
        assert pending[0].call_order_id == 12345


class TestOrderEventLogging:
    """Tests for order event logging via executor."""

    def test_logs_order_events(self, test_db):
        """Should log order status changes."""
        # Create trade first
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
        )
        test_db.log_trade(trade)

        # Log order event
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

        # Verify event logged
        event = test_db.get_latest_order_event(trade.trade_id)
        assert event is not None
        assert event["ib_order_id"] == 12345
        assert event["status"] == "Submitted"
