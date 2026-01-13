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
