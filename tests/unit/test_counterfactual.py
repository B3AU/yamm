"""Unit tests for counterfactual.py - pure functions for counterfactual P&L calculations."""
from datetime import date, timedelta

import pytest

from trading.earnings.counterfactual import (
    _find_closest_price,
    calculate_counterfactual_pnl,
)


class TestFindClosestPrice:
    """Tests for _find_closest_price helper function."""

    def test_exact_date_match(self):
        """Should return price when exact date exists."""
        prices = {
            "2026-01-28": 100.0,
            "2026-01-29": 105.0,
            "2026-01-30": 110.0,
        }
        target = date(2026, 1, 29)

        result = _find_closest_price(prices, target, direction="before")

        assert result is not None
        assert result[0] == 105.0
        assert result[1] == "2026-01-29"

    def test_find_before_with_gap(self):
        """Should find closest date before target when exact missing."""
        prices = {
            "2026-01-27": 100.0,  # Friday
            # 2026-01-28 is Saturday (no data)
            # 2026-01-29 is Sunday (no data)
            "2026-01-30": 110.0,  # Monday
        }
        # Target is weekend - should find Friday
        target = date(2026, 1, 28)

        result = _find_closest_price(prices, target, direction="before")

        assert result is not None
        assert result[0] == 100.0
        assert result[1] == "2026-01-27"

    def test_find_after_with_gap(self):
        """Should find closest date after target when exact missing."""
        prices = {
            "2026-01-27": 100.0,
            "2026-01-30": 110.0,
        }
        # Target is weekend - should find Monday when looking after
        target = date(2026, 1, 28)

        result = _find_closest_price(prices, target, direction="after")

        assert result is not None
        assert result[0] == 110.0
        assert result[1] == "2026-01-30"

    def test_no_price_found_returns_none(self):
        """Should return None when no price found within max_days."""
        prices = {
            "2026-01-15": 100.0,  # Too far away
        }
        target = date(2026, 1, 30)

        result = _find_closest_price(prices, target, direction="before", max_days=5)

        assert result is None

    def test_empty_prices_dict(self):
        """Should return None for empty prices dict."""
        prices = {}
        target = date(2026, 1, 30)

        result = _find_closest_price(prices, target, direction="before")

        assert result is None


class TestCalculateCounterfactualPnl:
    """Tests for calculate_counterfactual_pnl function."""

    def test_profitable_trade_move_exceeds_implied(self):
        """P&L should be positive when realized move exceeds implied move."""
        result = calculate_counterfactual_pnl(
            realized_move=0.08,    # 8% move
            implied_move=0.05,    # 5% implied (straddle priced for 5%)
            quoted_bid=4.50,
            quoted_ask=5.50,
            spot_price=100.0,
        )

        # Intrinsic value = 8% * $100 = $8
        # Mid price = $5
        # P&L at mid = $8 - $5 = $3 (profitable)
        assert result["profitable"] is True
        assert result["pnl_at_mid"] == pytest.approx(3.0)
        assert result["intrinsic_value"] == pytest.approx(8.0)

    def test_losing_trade_move_below_implied(self):
        """P&L should be negative when realized move is less than implied."""
        result = calculate_counterfactual_pnl(
            realized_move=0.03,    # 3% move
            implied_move=0.05,    # 5% implied
            quoted_bid=4.50,
            quoted_ask=5.50,
            spot_price=100.0,
        )

        # Intrinsic value = 3% * $100 = $3
        # Mid price = $5
        # P&L at mid = $3 - $5 = -$2 (loss)
        assert result["pnl_at_mid"] == pytest.approx(-2.0)
        assert result["intrinsic_value"] == pytest.approx(3.0)

    def test_zero_move_max_loss(self):
        """Zero move should result in loss equal to premium paid."""
        result = calculate_counterfactual_pnl(
            realized_move=0.0,
            implied_move=0.05,
            quoted_bid=4.50,
            quoted_ask=5.50,
            spot_price=100.0,
        )

        # Intrinsic value = 0
        # Mid price = $5
        # P&L at mid = 0 - $5 = -$5
        assert result["pnl_at_mid"] == pytest.approx(-5.0)
        assert result["intrinsic_value"] == 0.0
        assert result["profitable"] is False

    def test_spread_impact_on_pnl(self):
        """Wider spreads should result in worse P&L with spread."""
        # Same move, different spreads
        tight_spread = calculate_counterfactual_pnl(
            realized_move=0.06,
            implied_move=0.05,
            quoted_bid=4.90,
            quoted_ask=5.10,
            spot_price=100.0,
        )

        wide_spread = calculate_counterfactual_pnl(
            realized_move=0.06,
            implied_move=0.05,
            quoted_bid=4.00,
            quoted_ask=6.00,
            spot_price=100.0,
        )

        # pnl_with_spread should be worse for wide spread
        assert tight_spread["pnl_with_spread"] > wide_spread["pnl_with_spread"]

    def test_spread_adjusted_pnl_lower_than_mid(self):
        """P&L with spread should always be lower than P&L at mid."""
        result = calculate_counterfactual_pnl(
            realized_move=0.06,
            implied_move=0.05,
            quoted_bid=4.50,
            quoted_ask=5.50,
            spot_price=100.0,
        )

        assert result["pnl_with_spread"] < result["pnl_at_mid"]

    def test_large_move_very_profitable(self):
        """Large moves should result in large profits."""
        result = calculate_counterfactual_pnl(
            realized_move=0.15,    # 15% move (earnings surprise)
            implied_move=0.05,    # 5% implied
            quoted_bid=4.50,
            quoted_ask=5.50,
            spot_price=100.0,
        )

        # Intrinsic = 15% * $100 = $15
        # Mid = $5
        # P&L = $15 - $5 = $10
        assert result["pnl_at_mid"] == pytest.approx(10.0)
        assert result["profitable"] is True


# ============================================================================
# Tests Using Real Price Data (prices.pqt)
# ============================================================================

class TestWithRealPriceData:
    """Tests using production prices.pqt file."""

    def test_prices_parquet_structure(self, prices_parquet):
        """Verify prices parquet has expected structure."""
        import pandas as pd

        # Should have date index
        assert prices_parquet.index.name == 'date' or 'date' in prices_parquet.columns

        # Should have typical OHLCV columns
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        actual_cols = set(c.lower() for c in prices_parquet.columns)
        assert expected_cols.issubset(actual_cols), f"Missing columns: {expected_cols - actual_cols}"

    def test_prices_parquet_has_data(self, prices_parquet):
        """Verify prices parquet has meaningful data."""
        assert len(prices_parquet) > 0

        # Should have multiple symbols (multi-index or symbol column)
        if 'symbol' in prices_parquet.columns:
            unique_symbols = prices_parquet['symbol'].nunique()
            assert unique_symbols > 100, f"Expected many symbols, got {unique_symbols}"

    def test_find_closest_price_with_real_data(self, prices_parquet):
        """Test _find_closest_price with real historical prices."""
        import pandas as pd

        # Get AAPL prices if available
        if 'symbol' in prices_parquet.columns:
            aapl_prices = prices_parquet[prices_parquet['symbol'] == 'AAPL']
        else:
            # Try multi-index
            try:
                aapl_prices = prices_parquet.xs('AAPL', level='symbol')
            except (KeyError, TypeError):
                pytest.skip("Cannot extract AAPL prices from parquet")

        if len(aapl_prices) == 0:
            pytest.skip("No AAPL data in prices parquet")

        # Build price dict like the function expects
        if 'date' in aapl_prices.columns:
            price_dict = {
                str(row['date']): row['close']
                for _, row in aapl_prices.iterrows()
            }
        else:
            # Date is index
            price_dict = {
                str(idx): row['close']
                for idx, row in aapl_prices.iterrows()
            }

        # Pick a date that should exist
        dates = list(price_dict.keys())
        if dates:
            test_date = date.fromisoformat(dates[len(dates) // 2])
            result = _find_closest_price(price_dict, test_date, direction="before")
            assert result is not None
            assert result[0] > 0  # Price should be positive


class TestHistoricalEarningsData:
    """Tests using historical_earnings_moves.parquet."""

    def test_historical_earnings_structure(self, historical_earnings_parquet):
        """Verify historical earnings parquet has expected columns."""
        expected_cols = {'symbol', 'earnings_date'}
        actual_cols = set(historical_earnings_parquet.columns)
        assert expected_cols.issubset(actual_cols), f"Missing columns: {expected_cols - actual_cols}"

    def test_historical_earnings_has_moves(self, historical_earnings_parquet):
        """Should have realized move data."""
        # Should have move columns
        move_cols = [c for c in historical_earnings_parquet.columns if 'move' in c.lower()]
        assert len(move_cols) > 0, "Expected move columns in historical earnings"

    def test_historical_earnings_timing(self, historical_earnings_parquet):
        """Should have BMO/AMC timing data."""
        if 'timing' in historical_earnings_parquet.columns:
            timings = historical_earnings_parquet['timing'].dropna().unique()
            # Should have BMO and/or AMC
            assert len(timings) > 0
            valid_timings = {'BMO', 'AMC', 'unknown'}
            for t in timings:
                assert t in valid_timings or t is None
