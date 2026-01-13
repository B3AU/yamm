"""Unit tests for dashboard.py formatting functions."""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
import pytz

from trading.earnings.dashboard import (
    format_currency,
    format_pct,
    get_market_status,
    make_sparkline,
    format_time_until,
    format_time_since,
)


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_positive_amount(self):
        """Should format positive amounts with $ prefix."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(100.0) == "$100.00"
        assert format_currency(0.01) == "$0.01"

    def test_negative_amount(self):
        """Should format negative amounts with -$ prefix."""
        assert format_currency(-1234.56) == "-$1,234.56"
        assert format_currency(-100.0) == "-$100.00"
        assert format_currency(-0.01) == "-$0.01"

    def test_zero(self):
        """Should format zero correctly."""
        assert format_currency(0) == "$0.00"
        assert format_currency(0.0) == "$0.00"

    def test_none_returns_na(self):
        """Should return N/A for None."""
        assert format_currency(None) == "N/A"

    def test_large_amounts(self):
        """Should handle large amounts with commas."""
        assert format_currency(1000000.0) == "$1,000,000.00"
        assert format_currency(1234567.89) == "$1,234,567.89"

    def test_small_amounts(self):
        """Should show cents for small amounts."""
        assert format_currency(0.99) == "$0.99"
        assert format_currency(0.005) == "$0.01"  # rounds up


class TestFormatPct:
    """Tests for format_pct function."""

    def test_positive_percentage(self):
        """Should format as percentage with one decimal."""
        assert format_pct(0.1234) == "12.3%"
        assert format_pct(0.05) == "5.0%"
        assert format_pct(1.0) == "100.0%"

    def test_negative_percentage(self):
        """Should handle negative percentages."""
        assert format_pct(-0.05) == "-5.0%"
        assert format_pct(-0.1234) == "-12.3%"

    def test_zero(self):
        """Should format zero correctly."""
        assert format_pct(0) == "0.0%"
        assert format_pct(0.0) == "0.0%"

    def test_none_returns_na(self):
        """Should return N/A for None."""
        assert format_pct(None) == "N/A"

    def test_small_percentage(self):
        """Should handle small percentages."""
        assert format_pct(0.001) == "0.1%"
        assert format_pct(0.0001) == "0.0%"


class TestGetMarketStatus:
    """Tests for get_market_status function."""

    ET = pytz.timezone('US/Eastern')

    def test_market_open(self):
        """Should return OPEN during trading hours (9:30-16:00 ET)."""
        # Wednesday at 11:00 AM ET
        mock_time = datetime(2026, 1, 28, 11, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            status, color = get_market_status()

            assert status == "OPEN"
            assert '\033[92m' in color  # Green

    def test_premarket(self):
        """Should return PRE-MARKET before 9:30 ET."""
        # Wednesday at 7:00 AM ET
        mock_time = datetime(2026, 1, 28, 7, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            status, color = get_market_status()

            assert status == "PRE-MARKET"
            assert '\033[93m' in color  # Yellow

    def test_after_hours(self):
        """Should return AFTER-HOURS between 16:00-20:00 ET."""
        # Wednesday at 5:00 PM ET
        mock_time = datetime(2026, 1, 28, 17, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            status, color = get_market_status()

            assert status == "AFTER-HOURS"
            assert '\033[93m' in color  # Yellow

    def test_closed_overnight(self):
        """Should return CLOSED after 20:00 ET."""
        # Wednesday at 10:00 PM ET
        mock_time = datetime(2026, 1, 28, 22, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            status, color = get_market_status()

            assert status == "CLOSED"
            assert '\033[90m' in color  # Gray

    def test_closed_weekend(self):
        """Should return CLOSED on weekends."""
        # Saturday at noon ET
        mock_time = datetime(2026, 1, 31, 12, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            status, color = get_market_status()

            assert "CLOSED" in status
            assert "Weekend" in status


class TestMakeSparkline:
    """Tests for make_sparkline function."""

    def test_increasing_values(self):
        """Should show increasing pattern."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = make_sparkline(values)

        # First char should be lower, last should be highest
        blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        assert result[0] in blocks[:3]  # Low blocks
        assert result[-1] in blocks[-3:]  # High blocks

    def test_decreasing_values(self):
        """Should show decreasing pattern."""
        values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = make_sparkline(values)

        blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        assert result[0] in blocks[-3:]  # High blocks first
        assert result[-1] in blocks[:3]  # Low blocks last

    def test_constant_values(self):
        """Should show flat line for constant values."""
        values = [5, 5, 5, 5, 5]
        result = make_sparkline(values)

        # All chars should be the same (mid-level block)
        assert len(set(result)) == 1

    def test_empty_list(self):
        """Should return empty string for empty list."""
        result = make_sparkline([])
        assert result == ""

    def test_single_value(self):
        """Should handle single value."""
        result = make_sparkline([5])
        assert len(result) == 1

    def test_width_limiting(self):
        """Should limit output to specified width."""
        values = list(range(100))
        result = make_sparkline(values, width=20)

        assert len(result) == 20

    def test_custom_width(self):
        """Should respect custom width parameter."""
        values = [1, 2, 3, 4, 5]
        result = make_sparkline(values, width=10)

        # Values shorter than width, should not exceed value count
        assert len(result) <= max(len(values), 10)


class TestFormatTimeUntil:
    """Tests for format_time_until function."""

    def test_hours_and_minutes(self):
        """Should format hours and minutes."""
        now = datetime(2026, 1, 28, 10, 0, 0)
        target = datetime(2026, 1, 28, 14, 30, 0)

        result = format_time_until(target, now)

        assert "4h" in result
        assert "30m" in result

    def test_minutes_only(self):
        """Should format minutes when less than hour."""
        now = datetime(2026, 1, 28, 10, 0, 0)
        target = datetime(2026, 1, 28, 10, 45, 0)

        result = format_time_until(target, now)

        assert "45m" in result
        assert "h" not in result

    def test_days(self):
        """Should show days for long periods."""
        now = datetime(2026, 1, 28, 10, 0, 0)
        target = datetime(2026, 1, 31, 10, 0, 0)

        result = format_time_until(target, now)

        assert "3d" in result

    def test_passed(self):
        """Should return 'passed' for past time."""
        now = datetime(2026, 1, 28, 10, 0, 0)
        target = datetime(2026, 1, 28, 8, 0, 0)

        result = format_time_until(target, now)

        assert result == "passed"


class TestFormatTimeSince:
    """Tests for format_time_since function."""

    def test_hours_ago(self):
        """Should format hours ago."""
        now = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)

        result = format_time_since(past, now)

        assert "4h" in result
        assert "ago" in result

    def test_minutes_ago(self):
        """Should format minutes when less than hour."""
        now = datetime(2026, 1, 28, 10, 30, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 15, 0, tzinfo=timezone.utc)

        result = format_time_since(past, now)

        assert "15m" in result
        assert "ago" in result

    def test_days_ago(self):
        """Should show days for long periods."""
        now = datetime(2026, 1, 31, 10, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)

        result = format_time_since(past, now)

        assert "3d" in result

    def test_string_datetime_input(self):
        """Should handle ISO string input."""
        now = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)

        result = format_time_since("2026-01-28T10:00:00", now)

        assert "ago" in result

    def test_future_time(self):
        """Should return 'future' for future time."""
        now = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)

        result = format_time_since(past, now)

        assert result == "future"
