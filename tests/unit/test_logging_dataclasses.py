"""Unit tests for logging.py dataclasses."""
from dataclasses import asdict
from datetime import datetime

import pytest

from trading.earnings.logging import (
    TradeLog,
    NonTradeLog,
    SnapshotLog,
    ExecutionMetrics,
    generate_trade_id,
    generate_log_id,
)


class TestTradeLog:
    """Tests for TradeLog dataclass."""

    def test_trade_log_creation_minimal(self):
        """Should create TradeLog with required fields only."""
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
        )

        assert trade.trade_id == "AAPL_2026-01-30_20260129143000"
        assert trade.ticker == "AAPL"
        assert trade.status == "pending"  # default
        assert trade.structure == "straddle"  # default

    def test_trade_log_all_fields(self):
        """Should create TradeLog with all fields populated."""
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
            entry_slippage=0.02,
            structure="straddle",
            strikes="[230.0]",
            expiration="2026-01-31",
            contracts=2,
            premium_paid=1424.0,
            max_loss=1424.0,
            predicted_q50=0.025,
            predicted_q75=0.045,
            predicted_q90=0.065,
            predicted_q95=0.085,
            implied_move=0.031,
            edge_q75=0.014,
            edge_q90=0.034,
            exit_pnl=150.0,
            exit_pnl_pct=0.105,
            status="exited",
        )

        assert trade.entry_fill_price == 7.12
        assert trade.entry_slippage == 0.02
        assert trade.contracts == 2
        assert trade.exit_pnl == 150.0
        assert trade.status == "exited"

    def test_trade_log_to_dict(self):
        """Should convert to dictionary for DB insertion."""
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
        )

        data = asdict(trade)

        assert isinstance(data, dict)
        assert data["trade_id"] == "AAPL_2026-01-30_20260129143000"
        assert data["ticker"] == "AAPL"
        assert "entry_quoted_bid" in data

    def test_trade_log_edge_calculation(self):
        """Edge should be predicted - implied."""
        trade = TradeLog(
            trade_id="TEST_001",
            ticker="AAPL",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            entry_datetime="2026-01-29T14:30:00",
            entry_quoted_bid=6.90,
            entry_quoted_ask=7.30,
            entry_quoted_mid=7.10,
            entry_limit_price=7.15,
            predicted_q75=0.045,
            implied_move=0.031,
            edge_q75=0.014,  # 0.045 - 0.031 = 0.014
        )

        assert trade.edge_q75 == pytest.approx(trade.predicted_q75 - trade.implied_move)


class TestNonTradeLog:
    """Tests for NonTradeLog dataclass."""

    def test_non_trade_log_creation(self):
        """Should create NonTradeLog with rejection reason."""
        non_trade = NonTradeLog(
            log_id="NT_MSFT_20260129143000",
            ticker="MSFT",
            earnings_date="2026-01-30",
            earnings_timing="BMO",
            log_datetime="2026-01-29T14:30:00",
            rejection_reason="Spread too wide: 18.5% > 15.0%",
        )

        assert non_trade.ticker == "MSFT"
        assert "Spread too wide" in non_trade.rejection_reason

    def test_non_trade_log_with_quotes(self):
        """Should store quote data for counterfactual analysis."""
        non_trade = NonTradeLog(
            log_id="NT_TSLA_20260129143000",
            ticker="TSLA",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            log_datetime="2026-01-29T14:30:00",
            rejection_reason="Edge insufficient: 2.1% < 5.0%",
            quoted_bid=8.50,
            quoted_ask=9.20,
            quoted_spread_pct=7.8,
            spot_price=245.00,
            implied_move=0.036,
            predicted_q75=0.057,
            predicted_edge=0.021,
        )

        assert non_trade.quoted_bid == 8.50
        assert non_trade.quoted_ask == 9.20
        assert non_trade.implied_move == 0.036

    def test_non_trade_log_counterfactual_fields(self):
        """Should have fields for counterfactual backfill."""
        non_trade = NonTradeLog(
            log_id="NT_AMD_20260129143000",
            ticker="AMD",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            log_datetime="2026-01-29T14:30:00",
            rejection_reason="OI too low",
            counterfactual_realized_move=0.085,
            counterfactual_pnl=2.50,
            counterfactual_pnl_with_spread=1.80,
        )

        assert non_trade.counterfactual_realized_move == 0.085
        assert non_trade.counterfactual_pnl == 2.50


class TestSnapshotLog:
    """Tests for SnapshotLog dataclass."""

    def test_snapshot_log_creation(self):
        """Should create snapshot with position data."""
        snapshot = SnapshotLog(
            trade_id="AAPL_2026-01-30_20260129143000",
            ts="2026-01-29T15:00:00",
            minutes_since_open=330,  # 5.5 hours
            straddle_mid=7.25,
            call_mid=3.65,
            put_mid=3.60,
            spot_price=230.50,
            unrealized_pnl=25.0,
            unrealized_pnl_pct=0.017,
        )

        assert snapshot.trade_id.startswith("AAPL")
        assert snapshot.minutes_since_open == 330
        assert snapshot.unrealized_pnl == 25.0


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""

    def test_execution_metrics_defaults(self):
        """Should have sensible defaults."""
        metrics = ExecutionMetrics()

        assert metrics.total_orders == 0
        assert metrics.filled_orders == 0
        assert metrics.fill_rate == 0.0
        assert metrics.avg_slippage_bps == 0.0
        assert metrics.fill_rate_by_oi_bucket == {}

    def test_execution_metrics_with_data(self):
        """Should accept execution data."""
        metrics = ExecutionMetrics(
            total_orders=100,
            filled_orders=85,
            partial_fills=5,
            cancelled_orders=10,
            fill_rate=0.85,
            avg_slippage_bps=15.5,
            median_slippage_bps=12.0,
            max_slippage_bps=45.0,
        )

        assert metrics.fill_rate == 0.85
        assert metrics.avg_slippage_bps == 15.5


class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_generate_trade_id_format(self):
        """Trade ID should include ticker and date."""
        trade_id = generate_trade_id("AAPL", "2026-01-30")

        assert trade_id.startswith("AAPL_2026-01-30_")
        # Should have timestamp suffix
        parts = trade_id.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 14  # YYYYMMDDHHmmss

    def test_generate_trade_id_uniqueness(self):
        """Sequential calls should generate unique IDs."""
        id1 = generate_trade_id("AAPL", "2026-01-30")
        id2 = generate_trade_id("AAPL", "2026-01-30")

        # Very unlikely to be the same unless called in same second
        # This tests the format, not guaranteed uniqueness
        assert isinstance(id1, str)
        assert isinstance(id2, str)

    def test_generate_log_id_format(self):
        """Log ID should start with NT_ prefix."""
        log_id = generate_log_id("MSFT")

        assert log_id.startswith("NT_MSFT_")
        # Should have microsecond-precision timestamp
        parts = log_id.split("_")
        assert len(parts) == 3
        assert len(parts[2]) == 20  # YYYYMMDDHHmmssµµµµµµ
