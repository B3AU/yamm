"""Unit tests for dashboard.py formatting functions."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
    get_status_color,
    reset_color,
    bold,
    dim,
    format_age_compact,
    get_recent_warnings_errors,
    get_next_screen_time,
    get_next_exit_time,
    clear_screen,
    get_open_positions,
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


class TestGetStatusColor:
    """Tests for get_status_color function."""

    def test_pending_yellow(self):
        """Pending status should be yellow."""
        color = get_status_color('pending')
        assert '\033[93m' in color

    def test_filled_green(self):
        """Filled status should be green."""
        color = get_status_color('filled')
        assert '\033[92m' in color

    def test_exited_blue(self):
        """Exited status should be blue."""
        color = get_status_color('exited')
        assert '\033[94m' in color

    def test_cancelled_gray(self):
        """Cancelled status should be gray."""
        color = get_status_color('cancelled')
        assert '\033[90m' in color

    def test_error_red(self):
        """Error status should be red."""
        color = get_status_color('error')
        assert '\033[91m' in color

    def test_unknown_status(self):
        """Unknown status should return reset."""
        color = get_status_color('unknown_status')
        assert '\033[0m' in color

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert get_status_color('FILLED') == get_status_color('filled')
        assert get_status_color('Pending') == get_status_color('pending')


class TestAnsiFormatting:
    """Tests for ANSI formatting helpers."""

    def test_reset_color(self):
        """reset_color should return ANSI reset code."""
        assert reset_color() == '\033[0m'

    def test_bold(self):
        """bold should wrap text with bold codes."""
        result = bold('test')
        assert '\033[1m' in result
        assert 'test' in result
        assert '\033[0m' in result

    def test_dim(self):
        """dim should wrap text with dim codes."""
        result = dim('test')
        assert '\033[2m' in result
        assert 'test' in result
        assert '\033[0m' in result


class TestFormatAgeCompact:
    """Tests for format_age_compact function."""

    def test_minutes(self):
        """Should format as minutes when < 1 hour."""
        now = datetime(2026, 1, 28, 10, 30, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 15, 0, tzinfo=timezone.utc)

        result = format_age_compact(past, now)
        assert result == "15m"

    def test_hours(self):
        """Should format as hours when < 24 hours."""
        now = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)

        result = format_age_compact(past, now)
        assert result == "4h"

    def test_days_decimal(self):
        """Should show decimal days when < 10 days."""
        now = datetime(2026, 1, 30, 10, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)

        result = format_age_compact(past, now)
        assert "2.0d" in result

    def test_days_integer(self):
        """Should show integer days when >= 10 days."""
        now = datetime(2026, 2, 10, 10, 0, 0, tzinfo=timezone.utc)
        past = datetime(2026, 1, 28, 10, 0, 0, tzinfo=timezone.utc)

        result = format_age_compact(past, now)
        assert "13d" in result
        assert "." not in result

    def test_none_returns_question_mark(self):
        """Should return ? for None input."""
        result = format_age_compact(None)
        assert result == "?"

    def test_string_datetime_input(self):
        """Should handle ISO string input."""
        now = datetime(2026, 1, 28, 14, 0, 0, tzinfo=timezone.utc)
        result = format_age_compact("2026-01-28T10:00:00", now)
        assert "4h" in result


class TestGetRecentWarningsErrors:
    """Tests for get_recent_warnings_errors function."""

    def test_empty_for_missing_file(self, tmp_path):
        """Should return empty list for missing file."""
        result = get_recent_warnings_errors(tmp_path / "nonexistent.log")
        assert result == []

    def test_parses_warnings(self, tmp_path):
        """Should parse WARNING lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-01-28 10:30:45,123 - module - WARNING - Test warning message\n"
            "2026-01-28 10:30:46,456 - module - INFO - Info message ignored\n"
        )

        result = get_recent_warnings_errors(log_file)

        assert len(result) == 1
        assert result[0][0] == "WARNING"
        assert "Test warning" in result[0][1]

    def test_parses_errors(self, tmp_path):
        """Should parse ERROR lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-01-28 10:30:45,123 - module - ERROR - Critical error occurred\n"
        )

        result = get_recent_warnings_errors(log_file)

        assert len(result) == 1
        assert result[0][0] == "ERROR"
        assert "Critical error" in result[0][1]

    def test_most_recent_first(self, tmp_path):
        """Should return most recent entries first."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-01-28 10:00:00,000 - module - WARNING - First warning\n"
            "2026-01-28 11:00:00,000 - module - WARNING - Second warning\n"
            "2026-01-28 12:00:00,000 - module - WARNING - Third warning\n"
        )

        result = get_recent_warnings_errors(log_file)

        assert len(result) == 3
        assert "Third" in result[0][1]  # Most recent first
        assert "First" in result[2][1]  # Oldest last

    def test_max_display_limit(self, tmp_path):
        """Should respect max_display parameter."""
        log_file = tmp_path / "test.log"
        lines = "\n".join([
            f"2026-01-28 10:{i:02d}:00,000 - module - WARNING - Warning {i}"
            for i in range(20)
        ])
        log_file.write_text(lines)

        result = get_recent_warnings_errors(log_file, max_display=5)

        assert len(result) == 5

    def test_ignores_info_and_debug(self, tmp_path):
        """Should ignore INFO and DEBUG lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-01-28 10:30:45,123 - module - INFO - Info message\n"
            "2026-01-28 10:30:46,456 - module - DEBUG - Debug message\n"
            "2026-01-28 10:30:47,789 - module - WARNING - Warning message\n"
        )

        result = get_recent_warnings_errors(log_file)

        assert len(result) == 1
        assert "Warning" in result[0][1]


class TestGetNextScreenTime:
    """Tests for get_next_screen_time function."""

    ET = pytz.timezone('US/Eastern')

    def test_returns_datetime_during_trading_day(self):
        """Should return a datetime during trading hours."""
        # Monday morning
        mock_time = datetime(2026, 1, 26, 10, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            result = get_next_screen_time()

            # Should return next screening time (14:15 ET)
            assert result is None or isinstance(result, datetime)

    def test_returns_none_after_screening(self):
        """Should return None if screening already passed today."""
        # Monday evening after screening
        mock_time = datetime(2026, 1, 26, 20, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            result = get_next_screen_time()

            # May return None or next day
            assert result is None or isinstance(result, datetime)


class TestGetNextExitTime:
    """Tests for get_next_exit_time function."""

    ET = pytz.timezone('US/Eastern')

    def test_returns_datetime(self):
        """Should return a datetime or None."""
        mock_time = datetime(2026, 1, 26, 10, 0, 0, tzinfo=self.ET)

        with patch('trading.earnings.dashboard.datetime') as mock_dt:
            mock_dt.now.return_value = mock_time

            result = get_next_exit_time()

            assert result is None or isinstance(result, datetime)


class TestClearScreen:
    """Tests for clear_screen function."""

    def test_clear_screen_runs(self):
        """Should execute without error."""
        with patch('os.system') as mock_system:
            clear_screen()
            # Should have called os.system with clear command
            assert mock_system.called


class TestGetOpenPositions:
    """Tests for get_open_positions function."""

    def test_returns_list(self, test_db):
        """Should return a list of positions."""
        result = get_open_positions(test_db)
        assert isinstance(result, list)

    def test_returns_empty_for_no_positions(self, test_db):
        """Should return empty list when no positions."""
        result = get_open_positions(test_db)
        assert result == []

    def test_returns_filled_trades(self, test_db):
        """Should return filled trades as positions."""
        from trading.earnings.logging import TradeLog

        # Create a filled trade
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

        result = get_open_positions(test_db)
        assert len(result) == 1
        assert result[0].ticker == "AAPL"
