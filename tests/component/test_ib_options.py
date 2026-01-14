"""Component tests for ib_options.py - options client with mocked IB."""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from trading.earnings.ib_options import (
    OptionQuote,
    StraddleQuote,
    OptionOrder,
    IBOptionsClient,
)


# ============================================================================
# Tests for OptionQuote dataclass
# ============================================================================

class TestOptionQuote:
    """Tests for OptionQuote dataclass."""

    def test_create_call_quote(self):
        """Should create call option quote."""
        quote = OptionQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            bid=3.50,
            ask=3.70,
            mid=3.60,
            spread=0.20,
            spread_pct=5.56,
            volume=1000,
            open_interest=5000,
            iv=0.45,
            delta=0.52,
        )

        assert quote.symbol == "AAPL"
        assert quote.right == "C"
        assert quote.mid == 3.60
        assert quote.iv == 0.45

    def test_create_put_quote(self):
        """Should create put option quote."""
        quote = OptionQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="P",
            bid=3.40,
            ask=3.60,
            mid=3.50,
            spread=0.20,
            spread_pct=5.71,
        )

        assert quote.right == "P"
        assert quote.bid == 3.40

    def test_optional_greeks_default_none(self):
        """Greeks should default to None."""
        quote = OptionQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            bid=3.50,
            ask=3.70,
            mid=3.60,
            spread=0.20,
            spread_pct=5.56,
        )

        assert quote.iv is None
        assert quote.delta is None
        assert quote.gamma is None
        assert quote.theta is None
        assert quote.vega is None


# ============================================================================
# Tests for StraddleQuote dataclass
# ============================================================================

class TestStraddleQuote:
    """Tests for StraddleQuote dataclass."""

    def test_create_straddle_quote(self):
        """Should create straddle quote from call + put."""
        call = OptionQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            bid=3.50,
            ask=3.70,
            mid=3.60,
            spread=0.20,
            spread_pct=5.56,
        )

        put = OptionQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="P",
            bid=3.40,
            ask=3.60,
            mid=3.50,
            spread=0.20,
            spread_pct=5.71,
        )

        straddle = StraddleQuote(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            call=call,
            put=put,
            total_bid=6.90,
            total_ask=7.30,
            total_mid=7.10,
            total_spread=0.40,
            total_spread_pct=5.63,
            implied_move=0.031,
            spot_price=229.50,
        )

        assert straddle.call.right == "C"
        assert straddle.put.right == "P"
        assert straddle.total_mid == 7.10
        assert straddle.implied_move == pytest.approx(0.031)

    def test_implied_move_calculation_concept(self):
        """Verify implied move is straddle_mid / spot."""
        spot = 230.0
        straddle_mid = 7.10
        expected_implied_move = straddle_mid / spot  # 3.09%

        assert expected_implied_move == pytest.approx(0.0309, rel=0.01)


# ============================================================================
# Tests for OptionOrder dataclass
# ============================================================================

class TestOptionOrder:
    """Tests for OptionOrder dataclass."""

    def test_create_buy_order(self):
        """Should create buy order."""
        order = OptionOrder(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            action="BUY",
            quantity=2,
            order_type="LMT",
            limit_price=3.65,
        )

        assert order.action == "BUY"
        assert order.quantity == 2
        assert order.order_type == "LMT"
        assert order.status == "pending"

    def test_create_sell_order(self):
        """Should create sell order."""
        order = OptionOrder(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="P",
            action="SELL",
            quantity=1,
            order_type="MKT",
        )

        assert order.action == "SELL"
        assert order.order_type == "MKT"
        assert order.limit_price is None

    def test_order_with_fill_details(self):
        """Should store fill details."""
        fill_time = datetime(2026, 1, 29, 14, 35, 0)

        order = OptionOrder(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            action="BUY",
            quantity=2,
            order_type="LMT",
            limit_price=3.65,
            order_id=12345,
            status="filled",
            fill_price=3.62,
            fill_time=fill_time,
        )

        assert order.order_id == 12345
        assert order.status == "filled"
        assert order.fill_price == 3.62

    def test_order_with_error(self):
        """Should store error details."""
        order = OptionOrder(
            symbol="AAPL",
            expiry=date(2026, 1, 31),
            strike=230.0,
            right="C",
            action="BUY",
            quantity=2,
            order_type="LMT",
            status="error",
            error="Insufficient buying power",
        )

        assert order.status == "error"
        assert "buying power" in order.error


# ============================================================================
# Tests for IBOptionsClient
# ============================================================================

class TestIBOptionsClient:
    """Tests for IBOptionsClient class."""

    def test_client_initialization(self):
        """Should initialize with connection params."""
        client = IBOptionsClient(
            host="192.168.1.100",
            port=4002,
            client_id=5,
        )

        assert client.host == "192.168.1.100"
        assert client.port == 4002
        assert client.client_id == 5
        assert client._connected is False

    def test_default_connection_params(self):
        """Should use default connection params."""
        client = IBOptionsClient()

        assert client.host == "127.0.0.1"
        assert client.port == 7497
        assert client.client_id == 1

    def test_connect_success(self):
        """Should connect successfully when IB is available."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.host = "127.0.0.1"
            client.port = 4002
            client.client_id = 1
            client._connected = False
            client.ib = MagicMock()

            result = client.connect()

            client.ib.connect.assert_called_once()
            assert client._connected is True
            assert result is True

    def test_connect_failure(self):
        """Should handle connection failure."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.host = "127.0.0.1"
            client.port = 4002
            client.client_id = 1
            client._connected = False
            client.ib = MagicMock()
            client.ib.connect.side_effect = Exception("Connection refused")

            result = client.connect()

            assert client._connected is False
            assert result is False

    def test_disconnect(self):
        """Should disconnect from IB."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client._connected = True
            client.ib = MagicMock()

            client.disconnect()

            client.ib.disconnect.assert_called_once()
            assert client._connected is False

    def test_disconnect_when_not_connected(self):
        """Should not call disconnect if not connected."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client._connected = False
            client.ib = MagicMock()

            client.disconnect()

            client.ib.disconnect.assert_not_called()

    def test_is_connected_both_true(self):
        """is_connected should be True when both flags are True."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client._connected = True
            client.ib = MagicMock()
            client.ib.isConnected.return_value = True

            assert client.is_connected is True

    def test_is_connected_flag_false(self):
        """is_connected should be False when _connected is False."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client._connected = False
            client.ib = MagicMock()
            client.ib.isConnected.return_value = True

            assert client.is_connected is False

    def test_is_connected_ib_false(self):
        """is_connected should be False when IB reports disconnected."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client._connected = True
            client.ib = MagicMock()
            client.ib.isConnected.return_value = False

            assert client.is_connected is False

    def test_get_stock_price_success(self):
        """Should return stock price when available."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # Mock the ticker
            mock_ticker = MagicMock()
            mock_ticker.marketPrice.return_value = 150.50
            client.ib.reqMktData.return_value = mock_ticker

            price = client.get_stock_price("AAPL")

            assert price == 150.50
            client.ib.qualifyContracts.assert_called_once()
            client.ib.cancelMktData.assert_called_once()

    def test_get_stock_price_no_data(self):
        """Should return None when price not available."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # Mock the ticker with no price
            mock_ticker = MagicMock()
            mock_ticker.marketPrice.return_value = -1
            client.ib.reqMktData.return_value = mock_ticker

            price = client.get_stock_price("AAPL")

            assert price is None

    def test_get_stock_price_zero(self):
        """Should return None when price is zero."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            mock_ticker = MagicMock()
            mock_ticker.marketPrice.return_value = 0
            client.ib.reqMktData.return_value = mock_ticker

            price = client.get_stock_price("AAPL")

            assert price is None

    def test_cancel_order_success(self):
        """Should cancel order and return True."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # Mock an open trade with matching order ID
            mock_trade = MagicMock()
            mock_trade.order.orderId = 12345
            client.ib.openTrades.return_value = [mock_trade]

            result = client.cancel_order(12345)

            client.ib.cancelOrder.assert_called_once_with(mock_trade.order)
            assert result is True

    def test_cancel_order_not_found(self):
        """Should return False when order not in open trades."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # No matching order
            client.ib.openTrades.return_value = []

            result = client.cancel_order(99999)

            assert result is False
            client.ib.cancelOrder.assert_not_called()

    def test_cancel_all_orders(self):
        """Should call IB to cancel all orders."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            client.cancel_all_orders()

            client.ib.reqGlobalCancel.assert_called_once()

    def test_get_open_orders(self):
        """Should return list of open trades."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            mock_trades = [MagicMock(), MagicMock()]
            client.ib.openTrades.return_value = mock_trades

            result = client.get_open_orders()

            assert result == mock_trades
            client.ib.openTrades.assert_called_once()

    def test_get_positions_empty(self):
        """Should return empty dict when no positions."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()
            client.ib.positions.return_value = []

            result = client.get_positions()

            assert result == {}

    def test_get_positions_with_data(self):
        """Should return option positions as dict keyed by composite key."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # Create mock option position (only OPT secType included)
            pos1 = MagicMock()
            pos1.contract.secType = "OPT"
            pos1.contract.symbol = "AAPL"
            pos1.contract.lastTradeDateOrContractMonth = "20260131"
            pos1.contract.strike = 230.0
            pos1.contract.right = "C"
            pos1.position = 2
            pos1.avgCost = 3.50

            # Stock position should be excluded
            pos2 = MagicMock()
            pos2.contract.secType = "STK"
            pos2.contract.symbol = "MSFT"

            client.ib.positions.return_value = [pos1, pos2]

            result = client.get_positions()

            # Only the option position should be included
            expected_key = "AAPL_20260131_230.0_C"
            assert expected_key in result
            assert result[expected_key]["symbol"] == "AAPL"
            assert result[expected_key]["quantity"] == 2
            assert len(result) == 1  # Only 1 OPT position

    def test_get_account_summary(self):
        """Should return account summary dict with USD values only."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            # Mock account values - uses accountValues() not accountSummary
            val1 = MagicMock()
            val1.tag = "NetLiquidation"
            val1.currency = "USD"
            val1.value = "100000"

            val2 = MagicMock()
            val2.tag = "BuyingPower"
            val2.currency = "USD"
            val2.value = "50000"

            # Non-USD value should be excluded
            val3 = MagicMock()
            val3.tag = "NetLiquidation"
            val3.currency = "EUR"
            val3.value = "85000"

            # Non-included tag should be excluded
            val4 = MagicMock()
            val4.tag = "SomeOtherTag"
            val4.currency = "USD"
            val4.value = "999"

            client.ib.accountValues.return_value = [val1, val2, val3, val4]

            result = client.get_account_summary()

            assert result["NetLiquidation"] == 100000.0  # Converted to float
            assert result["BuyingPower"] == 50000.0
            assert len(result) == 2  # Only USD and included tags

    def test_sleep(self):
        """Should call IB sleep."""
        with patch.object(IBOptionsClient, '__init__', lambda x: None):
            client = IBOptionsClient()
            client.ib = MagicMock()

            client.sleep(2.5)

            client.ib.sleep.assert_called_once_with(2.5)
