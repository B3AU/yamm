"""Component tests for daemon.py - daemon functions and configuration."""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
import pytz


# ============================================================================
# Tests for today_et() function
# ============================================================================

class TestTodayET:
    """Tests for today_et helper function."""

    def test_returns_date_in_et(self):
        """Should return today's date in Eastern Time."""
        from trading.earnings.daemon import today_et

        result = today_et()

        assert isinstance(result, date)

    def test_matches_et_timezone(self):
        """Should match Eastern Time date."""
        from trading.earnings.daemon import today_et, ET

        # Get expected date in ET
        expected = datetime.now(ET).date()
        result = today_et()

        assert result == expected


# ============================================================================
# Tests for CONFIG dictionary
# ============================================================================

class TestConfig:
    """Tests for daemon configuration."""

    def test_config_has_required_keys(self):
        """CONFIG should have all required keys."""
        from trading.earnings.daemon import CONFIG

        required_keys = [
            'ib_host', 'ib_port', 'ib_client_id',
            'spread_threshold', 'max_daily_trades',
            'target_entry_amount', 'min_contracts', 'max_contracts',
            'db_path', 'log_path', 'dry_run',
        ]

        for key in required_keys:
            assert key in CONFIG, f"Missing required config key: {key}"

    def test_config_spread_threshold_is_float(self):
        """spread_threshold should be a float."""
        from trading.earnings.daemon import CONFIG

        assert isinstance(CONFIG['spread_threshold'], float)
        assert CONFIG['spread_threshold'] >= 0

    def test_config_edge_threshold_exists(self):
        """edge_threshold should exist."""
        from trading.earnings.daemon import CONFIG

        assert 'edge_threshold' in CONFIG
        assert isinstance(CONFIG['edge_threshold'], float)

    def test_config_max_daily_trades_is_int(self):
        """max_daily_trades should be an integer."""
        from trading.earnings.daemon import CONFIG

        assert isinstance(CONFIG['max_daily_trades'], int)
        assert CONFIG['max_daily_trades'] > 0

    def test_config_has_ib_connection_params(self):
        """Should have IB connection parameters."""
        from trading.earnings.daemon import CONFIG

        assert 'ib_host' in CONFIG
        assert 'ib_port' in CONFIG
        assert 'ib_client_id' in CONFIG

        assert isinstance(CONFIG['ib_port'], int)
        assert isinstance(CONFIG['ib_client_id'], int)

    def test_config_position_sizing_params(self):
        """Should have position sizing parameters."""
        from trading.earnings.daemon import CONFIG

        assert CONFIG['target_entry_amount'] > 0
        assert CONFIG['min_contracts'] >= 1
        assert CONFIG['max_contracts'] >= CONFIG['min_contracts']

    def test_config_llm_sanity_threshold(self):
        """Should have LLM sanity threshold."""
        from trading.earnings.daemon import CONFIG

        assert 'llm_sanity_threshold' in CONFIG
        valid_thresholds = {'PASS', 'WARN', 'NO_TRADE', 'DISABLED', 'LOG_ONLY'}
        assert CONFIG['llm_sanity_threshold'] in valid_thresholds


# ============================================================================
# Tests for PAPER_MODE flag
# ============================================================================

class TestPaperMode:
    """Tests for paper trading mode flag."""

    def test_paper_mode_is_bool(self):
        """PAPER_MODE should be a boolean."""
        from trading.earnings.daemon import PAPER_MODE

        assert isinstance(PAPER_MODE, bool)


# ============================================================================
# Tests for TradingDaemon class
# ============================================================================

class TestTradingDaemonInit:
    """Tests for TradingDaemon initialization."""

    def test_daemon_initialization(self):
        """Should initialize daemon with default state."""
        from trading.earnings.daemon import TradingDaemon

        with patch('trading.earnings.daemon.TradeLogger'):
            with patch('trading.earnings.daemon.AsyncIOScheduler'):
                daemon = TradingDaemon()

                # IB is initialized but not connected
                assert daemon.ib is not None
                assert daemon.connected is False

    def test_daemon_has_trade_logger(self):
        """Daemon should have trade logger."""
        from trading.earnings.daemon import TradingDaemon

        with patch('trading.earnings.daemon.TradeLogger') as MockLogger:
            with patch('trading.earnings.daemon.AsyncIOScheduler'):
                daemon = TradingDaemon()

                assert daemon.trade_logger is not None
                MockLogger.assert_called_once()

    def test_daemon_has_scheduler(self):
        """Daemon should have APScheduler."""
        from trading.earnings.daemon import TradingDaemon

        with patch('trading.earnings.daemon.TradeLogger'):
            with patch('trading.earnings.daemon.AsyncIOScheduler') as MockScheduler:
                daemon = TradingDaemon()

                assert daemon.scheduler is not None
                MockScheduler.assert_called_once()


# ============================================================================
# Tests for ET timezone
# ============================================================================

class TestETTimezone:
    """Tests for Eastern Time timezone constant."""

    def test_et_timezone_exists(self):
        """ET timezone constant should exist."""
        from trading.earnings.daemon import ET

        assert ET is not None
        assert hasattr(ET, 'zone')

    def test_et_is_eastern(self):
        """ET should be Eastern timezone."""
        from trading.earnings.daemon import ET

        assert 'Eastern' in str(ET) or 'America/New_York' in str(ET)
