"""Shared fixtures for earnings trading tests."""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import responses

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def project_root() -> Path:
    """Path to project root."""
    return PROJECT_ROOT


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def test_db(tmp_path) -> Generator:
    """Create isolated test database with TradeLogger.

    Uses tmp_path for automatic cleanup.
    """
    from trading.earnings.logging import TradeLogger

    db_path = tmp_path / "test_trades.db"
    logger = TradeLogger(str(db_path))
    yield logger


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_earnings_event():
    """Sample EarningsEvent for testing."""
    from trading.earnings.screener import EarningsEvent

    return EarningsEvent(
        symbol="AAPL",
        earnings_date=date(2026, 1, 30),
        timing="AMC",
        eps_estimate=6.05,
        revenue_estimate=123_400_000_000,
    )


@pytest.fixture
def sample_candidate():
    """Sample ScreenedCandidate for testing."""
    from trading.earnings.screener import ScreenedCandidate

    return ScreenedCandidate(
        symbol="AAPL",
        earnings_date=date(2026, 1, 30),
        timing="AMC",
        expiry="20260131",
        atm_strike=230.0,
        spot_price=229.50,
        call_bid=3.50,
        call_ask=3.70,
        put_bid=3.40,
        put_ask=3.60,
        straddle_mid=7.10,
        straddle_spread=0.40,
        spread_pct=5.6,
        implied_move_pct=3.1,
        call_iv=0.45,
        put_iv=0.44,
        passes_liquidity=True,
        passes_edge=True,
    )


@pytest.fixture
def sample_trade_log():
    """Sample TradeLog for testing."""
    from trading.earnings.logging import TradeLog

    return TradeLog(
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


@pytest.fixture
def sample_non_trade_log():
    """Sample NonTradeLog for testing."""
    from trading.earnings.logging import NonTradeLog

    return NonTradeLog(
        log_id="NT_MSFT_20260129143000000000",
        ticker="MSFT",
        earnings_date="2026-01-30",
        earnings_timing="BMO",
        log_datetime="2026-01-29T14:30:00",
        rejection_reason="Spread too wide: 18.5% > 15.0%",
        quoted_bid=4.20,
        quoted_ask=5.10,
        quoted_spread_pct=18.5,
        spot_price=420.00,
        implied_move=0.022,
        straddle_premium=4.65,
        predicted_q75=0.035,
        predicted_edge=0.013,
    )


# ============================================================================
# Mock IBKR Fixtures
# ============================================================================

@pytest.fixture
def mock_ib():
    """Mock ib_insync.IB connection."""
    with patch('ib_insync.IB') as MockIB:
        ib = MockIB.return_value
        ib.isConnected.return_value = True
        ib.managedAccounts.return_value = ['DU123456']
        ib.positions.return_value = []
        ib.openTrades.return_value = []
        ib.fills.return_value = []
        ib.accountValues.return_value = []

        # Async methods
        ib.connectAsync = AsyncMock()
        ib.qualifyContractsAsync = AsyncMock(return_value=[])
        ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[])

        yield ib


@pytest.fixture
def mock_ib_with_quotes(mock_ib):
    """Mock IB with market data quotes."""
    # Create mock ticker with bid/ask/last
    mock_ticker = MagicMock()
    mock_ticker.bid = 3.50
    mock_ticker.ask = 3.70
    mock_ticker.last = 3.60
    mock_ticker.close = 3.55
    mock_ticker.marketPrice.return_value = 3.60
    mock_ticker.modelGreeks = MagicMock(impliedVol=0.45)

    mock_ib.reqMktData.return_value = mock_ticker
    return mock_ib


# ============================================================================
# Mock API Fixtures
# ============================================================================

@pytest.fixture
def mock_fmp_responses(fixtures_dir):
    """Mock FMP API responses using responses library."""
    with responses.RequestsMock() as rsps:
        base_url = "https://financialmodelingprep.com"

        # Load fixture data
        earnings_file = fixtures_dir / "fmp_earnings.json"
        prices_file = fixtures_dir / "fmp_prices.json"

        if earnings_file.exists():
            with open(earnings_file) as f:
                earnings_data = json.load(f)
        else:
            earnings_data = []

        if prices_file.exists():
            with open(prices_file) as f:
                prices_data = json.load(f)
        else:
            prices_data = []

        # Mock earnings endpoint
        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/earnings.*"),
            json=earnings_data,
            status=200,
        )

        # Mock historical prices endpoint
        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/historical-price-eod.*"),
            json=prices_data,
            status=200,
        )

        # Mock fundamentals endpoints
        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/key-metrics.*"),
            json=[],
            status=200,
        )

        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/ratios.*"),
            json=[],
            status=200,
        )

        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/financial-growth.*"),
            json=[],
            status=200,
        )

        # Mock news endpoint
        rsps.add(
            responses.GET,
            re.compile(rf"{base_url}/stable/news/stock.*"),
            json=[],
            status=200,
        )

        yield rsps


@pytest.fixture
def mock_nasdaq_responses(fixtures_dir):
    """Mock Nasdaq API responses."""
    with responses.RequestsMock() as rsps:
        calendar_file = fixtures_dir / "nasdaq_calendar.json"

        if calendar_file.exists():
            with open(calendar_file) as f:
                calendar_data = json.load(f)
        else:
            calendar_data = {"data": {"rows": []}}

        rsps.add(
            responses.GET,
            re.compile(r"https://api\.nasdaq\.com/api/calendar/earnings.*"),
            json=calendar_data,
            status=200,
        )

        yield rsps


@pytest.fixture
def mock_tavily_responses():
    """Mock Tavily web search API."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={
                "results": [
                    {
                        "title": "Apple Q1 Earnings Preview",
                        "url": "https://example.com/apple-earnings",
                        "content": "Apple is expected to report strong Q1 earnings...",
                    }
                ],
                "query": "Apple AAPL earnings",
            },
            status=200,
        )
        yield rsps


@pytest.fixture
def mock_openrouter_responses():
    """Mock OpenRouter LLM API."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "decision": "PASS",
                                "risk_flags": [],
                                "reasons": ["No significant risks identified"],
                            })
                        }
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 100},
            },
            status=200,
        )
        yield rsps


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing."""
    monkeypatch.setenv("FMP_API_KEY", "test_fmp_key")
    monkeypatch.setenv("IB_HOST", "127.0.0.1")
    monkeypatch.setenv("IB_PORT", "4002")
    monkeypatch.setenv("IB_CLIENT_ID", "99")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.setenv("DRY_RUN", "true")


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
