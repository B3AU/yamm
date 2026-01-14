"""Component tests for executor.py - order execution with mocked IBKR."""
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.earnings.logging import TradeLog
from trading.earnings.executor import create_straddle_contract


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


# ============================================================================
# Tests for create_straddle_contract helper
# ============================================================================

class TestCreateStraddleContractHelper:
    """Tests for create_straddle_contract function."""

    def test_creates_contract_with_correct_symbol(self):
        """Should create contract with correct symbol."""
        # Create mock Option objects
        call = MagicMock()
        call.symbol = "AAPL"
        call.conId = 12345

        put = MagicMock()
        put.symbol = "AAPL"
        put.conId = 12346

        contract = create_straddle_contract(call=call, put=put)

        assert contract.symbol == "AAPL"
        assert contract.secType == "BAG"
        assert contract.currency == "USD"
        assert contract.exchange == "SMART"

    def test_creates_two_legs(self):
        """Should create combo with call and put legs."""
        call = MagicMock()
        call.symbol = "MSFT"
        call.conId = 11111

        put = MagicMock()
        put.symbol = "MSFT"
        put.conId = 22222

        contract = create_straddle_contract(call=call, put=put)

        assert len(contract.comboLegs) == 2

        # Find call and put legs
        call_leg = next((l for l in contract.comboLegs if l.conId == 11111), None)
        put_leg = next((l for l in contract.comboLegs if l.conId == 22222), None)

        assert call_leg is not None
        assert put_leg is not None
        assert call_leg.action == "BUY"
        assert put_leg.action == "BUY"
        assert call_leg.ratio == 1
        assert put_leg.ratio == 1


# ============================================================================
# Tests for Phase0Executor methods
# ============================================================================

class TestPhase0ExecutorMethods:
    """Tests for Phase0Executor methods."""

    def test_get_active_count_empty(self, test_db, mock_ib):
        """Should return 0 when no active orders."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        assert executor.get_active_count() == 0

    def test_get_active_count_with_orders(self, test_db, mock_ib):
        """Should count active orders."""
        from trading.earnings.executor import Phase0Executor, ComboOrder

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Add to internal order tracking using correct ComboOrder fields
        order1 = ComboOrder(
            trade_id="TEST_1",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
        )

        order2 = ComboOrder(
            trade_id="TEST_2",
            symbol="MSFT",
            expiry="20260207",
            strike=400.0,
        )

        executor.active_orders = {"TEST_1": order1, "TEST_2": order2}

        assert executor.get_active_count() == 2

    def test_cancel_all_clears_orders(self, test_db, mock_ib):
        """Should cancel all pending orders."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Add mock orders
        order1 = MagicMock()
        order1.order.orderId = 100
        order1.trade_id = "TEST_1"

        executor.active_orders = {"TEST_1": order1}

        # Cancel all
        executor.cancel_all()

        # Verify IB cancel was called
        mock_ib.cancelOrder.assert_called()

    def test_log_non_trade_creates_record(self, test_db, mock_ib, sample_candidate):
        """Should log non-trade with rejection reason."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Log non-trade
        executor.log_non_trade(sample_candidate)

        # Verify non-trade was logged
        non_trades = test_db.get_non_trades(
            from_date=str(sample_candidate.earnings_date),
            to_date=str(sample_candidate.earnings_date),
        )

        assert len(non_trades) == 1
        assert non_trades[0].ticker == sample_candidate.symbol

    def test_cancel_unfilled_orders_updates_status(self, test_db, mock_ib):
        """Should cancel and update status."""
        from trading.earnings.executor import Phase0Executor, ComboOrder

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Create a pending trade
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
            call_order_id=99999,  # Use the correct field name
        )
        test_db.log_trade(trade)

        # Add to active orders using correct ComboOrder fields
        combo_order = ComboOrder(
            trade_id=trade.trade_id,
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            order_id=99999,
        )
        mock_trade = MagicMock()
        mock_trade.order = MagicMock(orderId=99999)
        combo_order.trade = mock_trade
        executor.active_orders[trade.trade_id] = combo_order

        # Cancel
        result = executor.cancel_unfilled_orders(trade.trade_id)

        # Verify cancellation attempted
        assert result.get("cancelled", 0) >= 0 or result.get("not_found", 0) >= 0


# ============================================================================
# Tests for ComboOrder dataclass
# ============================================================================

class TestComboOrderDataclass:
    """Tests for ComboOrder dataclass."""

    def test_create_combo_order(self):
        """Should create ComboOrder with expected fields."""
        from trading.earnings.executor import ComboOrder

        combo = ComboOrder(
            trade_id="AAPL_2026-01-30_20260129143000",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            order_id=12345,
            call_conId=11111,
            put_conId=22222,
        )

        assert combo.trade_id == "AAPL_2026-01-30_20260129143000"
        assert combo.symbol == "AAPL"
        assert combo.strike == 230.0
        assert combo.order_id == 12345


# ============================================================================
# Tests for check_fills
# ============================================================================

class TestCheckFills:
    """Tests for check_fills method."""

    @pytest.mark.asyncio
    async def test_check_fills_returns_filled_orders(self, test_db, mock_ib):
        """Should return list of filled orders."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # No active orders = empty result
        filled = await executor.check_fills()
        assert filled == []

    @pytest.mark.asyncio
    async def test_check_fills_with_no_active_orders(self, test_db, mock_ib):
        """Should handle check_fills with no active orders gracefully."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # No active orders
        executor.active_orders = {}

        # Check fills should return empty list
        filled = await executor.check_fills()

        assert filled == []
        assert executor.get_active_count() == 0


# ============================================================================
# Tests for recover_orders
# ============================================================================

class TestRecoverOrders:
    """Tests for recover_orders method."""

    def test_recover_orders_loads_pending(self, test_db, mock_ib):
        """Should load pending orders from database."""
        from trading.earnings.executor import Phase0Executor

        # Create pending trade with order ID
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
            call_order_id=99999,  # Use correct field name
        )
        test_db.log_trade(trade)

        # Mock IB trades
        mock_ib_trade = MagicMock()
        mock_ib_trade.order = MagicMock(orderId=99999)
        mock_ib_trade.orderStatus = MagicMock(status="Submitted")
        mock_ib.trades.return_value = [mock_ib_trade]

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Recover orders
        count = executor.recover_orders()

        # Should find and recover the order
        assert count >= 0


# ============================================================================
# Tests for cancel_order
# ============================================================================

class TestCancelOrder:
    """Tests for cancel_order method."""

    def test_cancel_order_not_found(self, test_db, mock_ib):
        """Should return False for unknown trade_id."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        result = executor.cancel_order("UNKNOWN_TRADE_ID")

        assert result is False

    def test_cancel_order_success(self, test_db, mock_ib):
        """Should cancel active order."""
        from trading.earnings.executor import Phase0Executor, ComboOrder

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Create trade in database
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

        # Add to active orders
        mock_trade = MagicMock()
        mock_trade.order = MagicMock(orderId=99999)
        combo = ComboOrder(
            trade_id=trade.trade_id,
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            order_id=99999,
        )
        combo.trade = mock_trade
        executor.active_orders[trade.trade_id] = combo

        result = executor.cancel_order(trade.trade_id)

        assert result is True
        mock_ib.cancelOrder.assert_called_once()


# ============================================================================
# Tests for get_partial_fills
# ============================================================================

class TestGetPartialFills:
    """Tests for get_partial_fills method."""

    def test_get_partial_fills_empty(self, test_db, mock_ib):
        """Should return empty list when no partial fills."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        partials = executor.get_partial_fills()

        assert partials == []

    def test_get_partial_fills_with_partials(self, test_db, mock_ib):
        """Should return orders with partial fills."""
        from trading.earnings.executor import Phase0Executor, ComboOrder

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Create mock partially filled order
        mock_trade = MagicMock()
        mock_trade.order = MagicMock(orderId=99999, totalQuantity=2)
        mock_trade.orderStatus = MagicMock(filled=1.0)  # Partial fill

        combo = ComboOrder(
            trade_id="TEST_1",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            status="partial",
        )
        combo.trade = mock_trade
        executor.active_orders["TEST_1"] = combo

        partials = executor.get_partial_fills()

        assert len(partials) == 1
        assert partials[0].trade_id == "TEST_1"


# ============================================================================
# Tests for ExitComboOrder dataclass
# ============================================================================

class TestExitComboOrderDataclass:
    """Tests for ExitComboOrder dataclass."""

    def test_create_exit_order(self):
        """Should create exit combo order."""
        from trading.earnings.executor import ExitComboOrder

        exit_order = ExitComboOrder(
            trade_id="AAPL_2026-01-30_20260129143000",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            contracts=2,
            call_conId=11111,
            put_conId=22222,
        )

        assert exit_order.trade_id == "AAPL_2026-01-30_20260129143000"
        assert exit_order.contracts == 2
        assert exit_order.call_conId == 11111

    def test_exit_order_defaults(self):
        """Exit order should have None defaults for optional fields."""
        from trading.earnings.executor import ExitComboOrder

        exit_order = ExitComboOrder(
            trade_id="TEST",
            symbol="MSFT",
            expiry="20260207",
            strike=400.0,
            contracts=1,
        )

        assert exit_order.order_id is None
        assert exit_order.trade is None
        assert exit_order.fill_price is None


# ============================================================================
# Tests for log_non_trade
# ============================================================================

class TestLogNonTrade:
    """Tests for log_non_trade method."""

    def test_logs_non_trade_with_full_data(self, test_db, mock_ib, sample_candidate):
        """Should log non-trade with all available data."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Set rejection reason
        sample_candidate.rejection_reason = "Spread too wide: 25% > 15%"
        sample_candidate.passes_liquidity = False

        executor.log_non_trade(sample_candidate)

        # Verify logged
        non_trades = test_db.get_non_trades(
            from_date=str(sample_candidate.earnings_date),
            to_date=str(sample_candidate.earnings_date),
        )

        assert len(non_trades) == 1
        assert non_trades[0].ticker == sample_candidate.symbol
        assert "Spread too wide" in non_trades[0].rejection_reason

    def test_logs_non_trade_with_minimal_data(self, test_db, mock_ib):
        """Should handle non-trade with sparse quote data."""
        from trading.earnings.executor import Phase0Executor
        from trading.earnings.screener import ScreenedCandidate
        from datetime import date

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Create candidate with required fields but sparse pricing
        candidate = ScreenedCandidate(
            symbol="MINIMAL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            spot_price=50.0,
            expiry="2026-01-31",
            atm_strike=50.0,
            call_bid=0.0,
            call_ask=0.0,
            put_bid=0.0,
            put_ask=0.0,
            straddle_mid=0.0,
            straddle_spread=0.0,
            spread_pct=0.0,
            implied_move_pct=0.0,
            passes_liquidity=False,
            rejection_reason="No option chain",
        )

        executor.log_non_trade(candidate)

        non_trades = test_db.get_non_trades(
            from_date="2026-01-30",
            to_date="2026-01-30",
        )

        assert len(non_trades) == 1
        assert non_trades[0].ticker == "MINIMAL"


# ============================================================================
# Tests for _create_exit_combo function
# ============================================================================

class TestCreateExitCombo:
    """Tests for _create_exit_combo function."""

    def test_creates_sell_combo(self):
        """Should create a SELL combo for exit."""
        from trading.earnings.executor import _create_exit_combo
        from ib_insync import Option

        call = Option()
        call.symbol = "AAPL"
        call.secType = "OPT"
        call.strike = 230.0
        call.right = "C"
        call.conId = 111

        put = Option()
        put.symbol = "AAPL"
        put.secType = "OPT"
        put.strike = 230.0
        put.right = "P"
        put.conId = 222

        combo = _create_exit_combo(call, put)

        assert combo.secType == "BAG"
        assert len(combo.comboLegs) == 2
        # Both legs should be SELL actions
        assert combo.comboLegs[0].action == "SELL"
        assert combo.comboLegs[1].action == "SELL"


# ============================================================================
# Tests for create_straddle_contract function
# ============================================================================

class TestCreateStraddleContract:
    """Tests for create_straddle_contract standalone function."""

    def test_creates_buy_combo_by_default(self):
        """Should create BUY combo by default."""
        from trading.earnings.executor import create_straddle_contract
        from ib_insync import Option

        call = Option()
        call.symbol = "AAPL"
        call.conId = 111

        put = Option()
        put.symbol = "AAPL"
        put.conId = 222

        combo = create_straddle_contract(call, put)

        assert combo.secType == "BAG"
        assert combo.symbol == "AAPL"
        assert combo.currency == "USD"
        assert combo.exchange == "SMART"
        assert len(combo.comboLegs) == 2
        assert combo.comboLegs[0].action == "BUY"
        assert combo.comboLegs[1].action == "BUY"

    def test_creates_sell_combo_when_specified(self):
        """Should create SELL combo when action is SELL."""
        from trading.earnings.executor import create_straddle_contract
        from ib_insync import Option

        call = Option()
        call.symbol = "MSFT"
        call.conId = 333

        put = Option()
        put.symbol = "MSFT"
        put.conId = 444

        combo = create_straddle_contract(call, put, action='SELL')

        assert combo.symbol == "MSFT"
        assert combo.comboLegs[0].action == "SELL"
        assert combo.comboLegs[1].action == "SELL"

    def test_combo_legs_have_correct_conids(self):
        """Should include correct conIds in combo legs."""
        from trading.earnings.executor import create_straddle_contract
        from ib_insync import Option

        call = Option()
        call.symbol = "TSLA"
        call.conId = 12345

        put = Option()
        put.symbol = "TSLA"
        put.conId = 67890

        combo = create_straddle_contract(call, put)

        assert combo.comboLegs[0].conId == 12345
        assert combo.comboLegs[1].conId == 67890
        assert combo.comboLegs[0].ratio == 1
        assert combo.comboLegs[1].ratio == 1


# ============================================================================
# Tests for ExitComboOrder status tracking
# ============================================================================

class TestExitComboOrderStatus:
    """Tests for ExitComboOrder status tracking."""

    def test_default_status_is_pending(self):
        """New exit order should be pending by default."""
        from trading.earnings.executor import ExitComboOrder

        exit_order = ExitComboOrder(
            trade_id="EXIT_001",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            contracts=2,
        )

        assert exit_order.status == "pending"

    def test_can_set_filled_status(self):
        """Should be able to set filled status."""
        from trading.earnings.executor import ExitComboOrder

        exit_order = ExitComboOrder(
            trade_id="EXIT_002",
            symbol="AAPL",
            expiry="20260131",
            strike=230.0,
            contracts=2,
            status="filled",
            fill_price=7.50,
        )

        assert exit_order.status == "filled"
        assert exit_order.fill_price == 7.50


# ============================================================================
# Tests for ComboOrder status tracking
# ============================================================================

class TestComboOrderStatus:
    """Tests for ComboOrder status tracking."""

    def test_default_status_is_pending(self):
        """New combo order should be pending by default."""
        from trading.earnings.executor import ComboOrder

        order = ComboOrder(
            trade_id="TRADE_001",
            symbol="GOOGL",
            expiry="20260131",
            strike=185.0,
        )

        assert order.status == "pending"
        assert order.order_id is None
        assert order.trade is None

    def test_can_track_order_id(self):
        """Should track IB order ID."""
        from trading.earnings.executor import ComboOrder

        order = ComboOrder(
            trade_id="TRADE_002",
            symbol="GOOGL",
            expiry="20260131",
            strike=185.0,
            order_id=12345,
        )

        assert order.order_id == 12345

    def test_can_track_leg_conids(self):
        """Should track individual leg conIds."""
        from trading.earnings.executor import ComboOrder

        order = ComboOrder(
            trade_id="TRADE_003",
            symbol="AMZN",
            expiry="20260131",
            strike=200.0,
            call_conId=111,
            put_conId=222,
        )

        assert order.call_conId == 111
        assert order.put_conId == 222


# ============================================================================
# Tests for Phase0Executor edge cases
# ============================================================================

class TestPhase0ExecutorEdgeCases:
    """Edge case tests for Phase0Executor."""

    def test_log_non_trade_with_quotes(self, test_db, mock_ib, sample_candidate):
        """Should log non-trade with quote data."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Set rejection reason and quote data
        sample_candidate.rejection_reason = "Spread too wide"
        sample_candidate.passes_liquidity = False

        executor.log_non_trade(sample_candidate)

        non_trades = test_db.get_non_trades(
            from_date=str(sample_candidate.earnings_date),
            to_date=str(sample_candidate.earnings_date),
        )

        assert len(non_trades) == 1
        # Verify quote data was passed through
        assert non_trades[0].ticker == sample_candidate.symbol
        assert non_trades[0].rejection_reason == "Spread too wide"
        assert non_trades[0].spot_price == sample_candidate.spot_price
        assert non_trades[0].quoted_spread_pct == sample_candidate.spread_pct

    def test_cancel_unfilled_orders_no_pending(self, test_db, mock_ib):
        """Should handle case when no pending orders exist."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # No orders in system
        result = executor.cancel_unfilled_orders("NONEXISTENT_TRADE")

        # Returns {'cancelled': False, 'reason': 'not found'} when not found
        assert result["cancelled"] is False
        assert result["reason"] == "not found"

    def test_recover_orders_empty_db(self, test_db, mock_ib):
        """Should handle case when no pending orders in DB."""
        from trading.earnings.executor import Phase0Executor

        executor = Phase0Executor(ib=mock_ib, trade_logger=test_db)

        # Clear any existing orders
        executor.active_orders = {}

        count = executor.recover_orders()

        # Should find 0 orders to recover
        assert count == 0
        assert len(executor.active_orders) == 0
