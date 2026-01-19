"""Unit tests for screener.py utility functions."""
import math
from datetime import date

import pytest

from trading.earnings.screener import is_valid_price


class TestIsValidPrice:
    """Tests for is_valid_price validation function."""

    def test_valid_positive_price(self):
        """Should return True for valid positive prices within max_value."""
        assert is_valid_price(100.0) is True
        assert is_valid_price(0.01) is True
        assert is_valid_price(99999.0) is True  # Just under default max_value
        assert is_valid_price(1) is True  # int

    def test_exceeds_max_value_returns_false(self):
        """Should return False for values exceeding max_value (data errors)."""
        # Default max_value is 100000
        assert is_valid_price(100000.0) is False  # At max is excluded
        assert is_valid_price(1000000.0) is False  # Way over max

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

    def test_inf_returns_false(self):
        """Infinity should return False (treated as invalid data)."""
        assert is_valid_price(float('inf')) is False
        assert is_valid_price(float('-inf')) is False

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


class TestExpirationSelection:
    """Tests for expiration selection logic based on earnings timing.
    
    The key rule:
    - AMC: expiry must be STRICTLY AFTER earnings_date (to capture next-day gap)
    - BMO/unknown: expiry can be ON OR AFTER earnings_date (gap is same day)
    
    This is critical for Friday AMC where the gap is Monday but options expire Friday.
    """

    def _select_expiry(self, expiries: list[str], earnings_date: date, timing: str) -> str | None:
        """Helper to replicate the expiration selection logic from screener.py."""
        from datetime import datetime
        
        target_expiry = None
        for exp in sorted(expiries):
            exp_date = datetime.strptime(exp, '%Y%m%d').date()
            if timing == 'AMC':
                if exp_date > earnings_date:
                    target_expiry = exp
                    break
            else:  # BMO or unknown - same day expiry is fine
                if exp_date >= earnings_date:
                    target_expiry = exp
                    break
        return target_expiry

    def test_friday_amc_selects_next_week_expiry(self):
        """Friday AMC should select next week's expiry to capture Monday gap.
        
        Scenario: Friday Jan 17 AMC earnings
        - Available expiries: Jan 17 (Friday), Jan 24 (next Friday)
        - Should select Jan 24 because options expire Friday EOD, before Monday gap
        """
        expiries = ['20260117', '20260124', '20260131']  # Fridays
        earnings_date = date(2026, 1, 17)  # Friday
        timing = 'AMC'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected == '20260124'  # Next Friday, NOT same-day

    def test_monday_amc_selects_next_day_or_later(self):
        """Monday AMC should select Tuesday or later expiry.
        
        Scenario: Monday Jan 20 AMC earnings
        - Available expiries: Jan 20 (Monday - unusual), Jan 24 (Friday)
        - Should select Jan 24 because gap is Tuesday morning
        """
        expiries = ['20260120', '20260124']
        earnings_date = date(2026, 1, 20)  # Monday
        timing = 'AMC'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected == '20260124'  # Friday, NOT Monday

    def test_tuesday_bmo_selects_same_day(self):
        """Tuesday BMO should select Tuesday expiry (same day is fine).
        
        Scenario: Tuesday Jan 21 BMO earnings
        - Available expiries: Jan 21 (Tuesday - unusual), Jan 24 (Friday)
        - Should select Jan 21 because gap happens at Tuesday open
        """
        expiries = ['20260121', '20260124']
        earnings_date = date(2026, 1, 21)  # Tuesday
        timing = 'BMO'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected == '20260121'  # Same day is fine for BMO

    def test_unknown_timing_treated_as_bmo(self):
        """Unknown timing should select same-day expiry (like BMO).
        
        Scenario: Friday Jan 17 unknown timing
        - Available expiries: Jan 17, Jan 24
        - Should select Jan 17 (conservative: assume BMO)
        """
        expiries = ['20260117', '20260124']
        earnings_date = date(2026, 1, 17)
        timing = 'unknown'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected == '20260117'  # Same day OK for unknown (treat as BMO)

    def test_amc_with_only_same_day_expiry_returns_none(self):
        """AMC with only same-day expiry available should find nothing.
        
        This would be rejected by the screener as 'No expiry after earnings'.
        """
        expiries = ['20260117']  # Only Friday available
        earnings_date = date(2026, 1, 17)  # Friday AMC
        timing = 'AMC'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected is None  # No valid expiry for AMC

    def test_bmo_friday_selects_same_friday(self):
        """Friday BMO should select same Friday expiry.
        
        Scenario: Friday Jan 17 BMO earnings
        - Gap happens Friday morning
        - Friday expiry is fine (exit by 14:00, expiry at 16:00)
        """
        expiries = ['20260117', '20260124']
        earnings_date = date(2026, 1, 17)
        timing = 'BMO'
        
        selected = self._select_expiry(expiries, earnings_date, timing)
        
        assert selected == '20260117'  # Same Friday OK for BMO
