"""Unit tests for screener.py utility functions."""
import math

import pytest

from trading.earnings.screener import is_valid_price


class TestIsValidPrice:
    """Tests for is_valid_price validation function."""

    def test_valid_positive_price(self):
        """Should return True for valid positive prices."""
        assert is_valid_price(100.0) is True
        assert is_valid_price(0.01) is True
        assert is_valid_price(1000000.0) is True
        assert is_valid_price(1) is True  # int

    def test_none_returns_false(self):
        """Should return False for None."""
        assert is_valid_price(None) is False

    def test_zero_returns_false(self):
        """Should return False for zero."""
        assert is_valid_price(0) is False
        assert is_valid_price(0.0) is False

    def test_negative_returns_false(self):
        """Should return False for negative prices."""
        assert is_valid_price(-1.0) is False
        assert is_valid_price(-100) is False
        assert is_valid_price(-0.01) is False

    def test_nan_returns_false(self):
        """Should return False for NaN values."""
        assert is_valid_price(float('nan')) is False
        assert is_valid_price(math.nan) is False

    def test_inf_returns_true(self):
        """Infinity is technically positive but unusual - test current behavior."""
        # Current implementation returns True for inf
        # This documents the behavior - may want to change in future
        assert is_valid_price(float('inf')) is True

    def test_string_number_converts(self):
        """String numbers should be converted and validated."""
        assert is_valid_price("100.0") is True
        assert is_valid_price("0.01") is True
        assert is_valid_price("-5.0") is False
        assert is_valid_price("0") is False

    def test_invalid_string_returns_false(self):
        """Non-numeric strings should return False."""
        assert is_valid_price("abc") is False
        assert is_valid_price("") is False
        assert is_valid_price("N/A") is False

    def test_list_returns_false(self):
        """Non-numeric types like lists should return False."""
        assert is_valid_price([100.0]) is False
        assert is_valid_price([]) is False

    def test_dict_returns_false(self):
        """Dict types should return False."""
        assert is_valid_price({"price": 100.0}) is False


class TestSpreadCalculation:
    """Tests for spread percentage calculations (inline logic)."""

    def test_spread_pct_calculation(self):
        """Verify spread percentage formula."""
        call_bid, call_ask = 3.50, 3.70
        put_bid, put_ask = 3.40, 3.60

        call_mid = (call_bid + call_ask) / 2
        put_mid = (put_bid + put_ask) / 2
        straddle_mid = call_mid + put_mid
        straddle_spread = (call_ask - call_bid) + (put_ask - put_bid)
        spread_pct = (straddle_spread / straddle_mid * 100) if straddle_mid > 0 else 100

        # Expected: spread = 0.20 + 0.20 = 0.40
        # Mid = 3.60 + 3.50 = 7.10
        # Spread % = 0.40 / 7.10 * 100 = 5.63%
        assert spread_pct == pytest.approx(5.63, rel=0.01)

    def test_wide_spread_percentage(self):
        """Wide spreads should have higher percentage."""
        call_bid, call_ask = 2.00, 3.00  # $1.00 spread
        put_bid, put_ask = 1.80, 2.80   # $1.00 spread

        call_mid = (call_bid + call_ask) / 2  # 2.50
        put_mid = (put_bid + put_ask) / 2    # 2.30
        straddle_mid = call_mid + put_mid    # 4.80
        straddle_spread = (call_ask - call_bid) + (put_ask - put_bid)  # 2.00
        spread_pct = (straddle_spread / straddle_mid * 100)

        # 2.00 / 4.80 * 100 = 41.67%
        assert spread_pct == pytest.approx(41.67, rel=0.01)

    def test_zero_mid_edge_case(self):
        """Zero mid should default to 100% spread (worst case)."""
        straddle_mid = 0.0
        straddle_spread = 0.10
        spread_pct = (straddle_spread / straddle_mid * 100) if straddle_mid > 0 else 100

        assert spread_pct == 100.0


class TestImpliedMoveCalculation:
    """Tests for implied move percentage calculations."""

    def test_implied_move_calculation(self):
        """Verify implied move formula."""
        straddle_mid = 7.10
        spot_price = 229.50

        implied_move_pct = (straddle_mid / spot_price * 100) if spot_price > 0 else 0

        # 7.10 / 229.50 * 100 = 3.09%
        assert implied_move_pct == pytest.approx(3.09, rel=0.01)

    def test_high_iv_large_implied_move(self):
        """High IV stocks should have larger implied moves."""
        straddle_mid = 25.0
        spot_price = 100.0

        implied_move_pct = (straddle_mid / spot_price * 100)

        # 25 / 100 * 100 = 25% implied move
        assert implied_move_pct == 25.0

    def test_zero_spot_price_edge_case(self):
        """Zero spot price should default to 0% implied move."""
        straddle_mid = 7.10
        spot_price = 0.0

        implied_move_pct = (straddle_mid / spot_price * 100) if spot_price > 0 else 0

        assert implied_move_pct == 0.0
