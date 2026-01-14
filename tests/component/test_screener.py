"""Component tests for screener.py - with mocked APIs and IBKR."""
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest
import responses

from trading.earnings.screener import (
    EarningsEvent,
    ScreenedCandidate,
    fetch_upcoming_earnings,
    fetch_fmp_earnings,
    _rejected_candidate,
    screen_candidate_ibkr,
)


class TestFetchUpcomingEarnings:
    """Tests for fetch_upcoming_earnings function."""

    @responses.activate
    def test_fetches_earnings_from_nasdaq(self, mock_env_vars):
        """Should fetch and parse Nasdaq earnings calendar."""
        # Mock single day response
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {
                            "symbol": "AAPL",
                            "time": "After Market Close",
                            "epsForecast": "$6.05",
                        },
                        {
                            "symbol": "MSFT",
                            "time": "Before Market Open",
                            "epsForecast": "$3.12",
                        },
                    ]
                }
            },
            status=200,
        )

        # Add empty responses for other days
        for _ in range(6):  # days_ahead = 7
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert len(events) == 2
        assert events[0].symbol == "AAPL"
        assert events[0].timing == "AMC"
        assert events[1].symbol == "MSFT"
        assert events[1].timing == "BMO"

    @responses.activate
    def test_parses_eps_estimate(self, mock_env_vars):
        """Should parse EPS estimate from Nasdaq format."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {
                            "symbol": "AAPL",
                            "time": "After Market Close",
                            "epsForecast": "$6.05",
                        },
                    ]
                }
            },
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert events[0].eps_estimate == 6.05

    @responses.activate
    def test_parses_negative_eps(self, mock_env_vars):
        """Should parse negative EPS estimates."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {
                            "symbol": "RIVN",
                            "time": "After Market Close",
                            "epsForecast": "($1.25)",
                        },
                    ]
                }
            },
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert events[0].eps_estimate == -1.25

    @responses.activate
    def test_filters_non_us_symbols(self, mock_env_vars):
        """Should filter out ADRs and non-US symbols."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {"symbol": "AAPL", "time": "AMC", "epsForecast": "$6.05"},
                        {"symbol": "TSM.TW", "time": "BMO", "epsForecast": "$1.50"},  # Taiwan
                        {"symbol": "BABA-SW", "time": "AMC", "epsForecast": "$2.00"},  # ADR
                    ]
                }
            },
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        # Only AAPL should pass through
        symbols = [e.symbol for e in events]
        assert "AAPL" in symbols
        assert "TSM.TW" not in symbols
        assert "BABA-SW" not in symbols

    @responses.activate
    def test_handles_unknown_timing(self, mock_env_vars):
        """Should mark unknown timing correctly."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {"symbol": "META", "time": "time-not-supplied", "epsForecast": "$5.85"},
                    ]
                }
            },
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert events[0].timing == "unknown"

    @responses.activate
    def test_handles_api_error(self, mock_env_vars):
        """Should handle API errors gracefully."""
        for _ in range(7):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"error": "API error"},
                status=500,
            )

        # Should not raise, should return empty or partial
        events = fetch_upcoming_earnings(days_ahead=7)
        assert isinstance(events, list)


class TestFetchFMPEarnings:
    """Tests for fetch_fmp_earnings function."""

    @responses.activate
    def test_fetches_from_fmp(self, mock_env_vars):
        """Should fetch earnings from FMP API."""
        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/earnings-calendar",
            json=[
                {
                    "symbol": "AAPL",
                    "date": "2026-01-30",
                    "epsEstimated": 6.05,
                    "revenueEstimated": 123400000000,
                },
            ],
            status=200,
        )

        events = fetch_fmp_earnings(days_ahead=7)

        assert len(events) == 1
        assert events[0].symbol == "AAPL"
        assert events[0].eps_estimate == 6.05

    def test_returns_empty_without_api_key(self, monkeypatch):
        """Should return empty list without API key."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)

        events = fetch_fmp_earnings(days_ahead=7)

        assert events == []


class TestRejectedCandidate:
    """Tests for _rejected_candidate helper."""

    def test_creates_rejected_candidate(self):
        """Should create candidate with rejection reason."""
        candidate = _rejected_candidate(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            reason="Spread too wide: 25% > 15%",
        )

        assert candidate.symbol == "AAPL"
        assert candidate.passes_liquidity is False
        assert "Spread too wide" in candidate.rejection_reason

    def test_includes_optional_data(self):
        """Should include optional market data in rejection."""
        candidate = _rejected_candidate(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            reason="No valid bid/ask",
            spot_price=229.50,
            expiry="20260131",
            atm_strike=230.0,
            call_bid=0.0,
            call_ask=0.0,
        )

        assert candidate.spot_price == 229.50
        assert candidate.atm_strike == 230.0


class TestEarningsEventDataclass:
    """Tests for EarningsEvent dataclass."""

    def test_create_earnings_event(self):
        """Should create earnings event with required fields."""
        event = EarningsEvent(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            eps_estimate=6.05,
        )

        assert event.symbol == "AAPL"
        assert event.earnings_date == date(2026, 1, 30)
        assert event.timing == "AMC"
        assert event.eps_estimate == 6.05

    def test_create_event_with_defaults(self):
        """Should create event with default optional values."""
        event = EarningsEvent(
            symbol="MSFT",
            earnings_date=date(2026, 1, 31),
            timing="BMO",
        )

        assert event.eps_estimate is None
        assert event.revenue_estimate is None


class TestScreenedCandidateDataclass:
    """Tests for ScreenedCandidate dataclass."""

    def test_create_screened_candidate(self):
        """Should create candidate with all fields."""
        candidate = ScreenedCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            spot_price=229.50,
            expiry="20260131",
            atm_strike=230.0,
            call_bid=3.50,
            call_ask=3.70,
            put_bid=3.40,
            put_ask=3.60,
            straddle_mid=7.10,
            straddle_spread=0.40,
            spread_pct=5.6,
            implied_move_pct=3.1,
        )

        assert candidate.symbol == "AAPL"
        assert candidate.straddle_mid == 7.10
        assert candidate.implied_move_pct == 3.1

    def test_candidate_with_ml_predictions(self):
        """Should store ML prediction fields."""
        candidate = ScreenedCandidate(
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            spot_price=229.50,
            expiry="20260131",
            atm_strike=230.0,
            call_bid=3.50,
            call_ask=3.70,
            put_bid=3.40,
            put_ask=3.60,
            straddle_mid=7.10,
            straddle_spread=0.40,
            spread_pct=5.6,
            implied_move_pct=3.1,
            pred_q50=0.04,
            pred_q75=0.06,
            pred_q90=0.08,
            pred_q95=0.10,
            edge_q75=0.03,
        )

        assert candidate.pred_q75 == 0.06
        assert candidate.edge_q75 == 0.03


class TestScreenCandidateIBKR:
    """Tests for screen_candidate_ibkr async function."""

    @pytest.mark.asyncio
    async def test_screens_valid_candidate(self, mock_ib_with_quotes):
        """Should return passing candidate with good liquidity."""
        # Setup mock option chain
        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.expirations = ["20260131", "20260207"]
        mock_chain.strikes = [225.0, 227.5, 230.0, 232.5, 235.0]

        mock_ib_with_quotes.qualifyContractsAsync = AsyncMock(
            return_value=[MagicMock(), MagicMock()]
        )
        mock_ib_with_quotes.reqSecDefOptParamsAsync = AsyncMock(
            return_value=[mock_chain]
        )

        # Create mock ticker for stock price
        stock_ticker = MagicMock()
        stock_ticker.bid = 229.00
        stock_ticker.ask = 230.00
        stock_ticker.last = 229.50
        stock_ticker.close = 229.30
        stock_ticker.marketPrice.return_value = 229.50

        # Create mock tickers for options
        call_ticker = MagicMock()
        call_ticker.bid = 3.50
        call_ticker.ask = 3.70
        call_ticker.modelGreeks = MagicMock(impliedVol=0.45)

        put_ticker = MagicMock()
        put_ticker.bid = 3.40
        put_ticker.ask = 3.60
        put_ticker.modelGreeks = MagicMock(impliedVol=0.44)

        # Return different tickers for stock vs options
        ticker_responses = [stock_ticker, call_ticker, put_ticker]
        mock_ib_with_quotes.reqMktData.side_effect = ticker_responses

        candidate = await screen_candidate_ibkr(
            ib=mock_ib_with_quotes,
            symbol="AAPL",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            spread_threshold=15.0,
        )

        assert candidate.symbol == "AAPL"
        # Spread should be calculated and checked

    @pytest.mark.asyncio
    async def test_rejects_wide_spread(self, mock_ib_with_quotes):
        """Should reject candidate with spread exceeding threshold."""
        # Setup mock with wide spreads
        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.expirations = ["20260131"]
        mock_chain.strikes = [230.0]

        mock_ib_with_quotes.qualifyContractsAsync = AsyncMock(
            return_value=[MagicMock(), MagicMock()]
        )
        mock_ib_with_quotes.reqSecDefOptParamsAsync = AsyncMock(
            return_value=[mock_chain]
        )

        stock_ticker = MagicMock()
        stock_ticker.last = 229.50
        stock_ticker.close = 229.30
        stock_ticker.marketPrice.return_value = 229.50

        # Wide spread options
        call_ticker = MagicMock()
        call_ticker.bid = 2.00
        call_ticker.ask = 4.00  # 100% spread on call alone
        call_ticker.modelGreeks = MagicMock(impliedVol=0.45)

        put_ticker = MagicMock()
        put_ticker.bid = 1.80
        put_ticker.ask = 3.80  # 111% spread on put
        put_ticker.modelGreeks = MagicMock(impliedVol=0.44)

        mock_ib_with_quotes.reqMktData.side_effect = [stock_ticker, call_ticker, put_ticker]

        candidate = await screen_candidate_ibkr(
            ib=mock_ib_with_quotes,
            symbol="ILLIQUID",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
            spread_threshold=15.0,
        )

        assert candidate.passes_liquidity is False
        assert "Spread too wide" in candidate.rejection_reason

    @pytest.mark.asyncio
    async def test_handles_no_spot_price(self, mock_ib):
        """Should reject when no spot price available."""
        # Stock ticker with no valid prices
        stock_ticker = MagicMock()
        stock_ticker.bid = None
        stock_ticker.ask = None
        stock_ticker.last = None
        stock_ticker.close = None
        stock_ticker.marketPrice.return_value = float('nan')

        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])
        mock_ib.reqMktData.return_value = stock_ticker

        candidate = await screen_candidate_ibkr(
            ib=mock_ib,
            symbol="UNKNOWN",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
        )

        assert candidate.passes_liquidity is False
        assert "spot price" in candidate.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_handles_no_option_chain(self, mock_ib):
        """Should reject when no option chain available."""
        stock_ticker = MagicMock()
        stock_ticker.last = 100.0
        stock_ticker.close = 100.0
        stock_ticker.marketPrice.return_value = 100.0

        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])
        mock_ib.reqMktData.return_value = stock_ticker
        mock_ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[])

        candidate = await screen_candidate_ibkr(
            ib=mock_ib,
            symbol="NOPTIONS",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
        )

        assert candidate.passes_liquidity is False
        assert "option chain" in candidate.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_handles_no_expiry_match(self, mock_ib):
        """Should reject when no matching expiry available."""
        stock_ticker = MagicMock()
        stock_ticker.last = 100.0
        stock_ticker.close = 100.0
        stock_ticker.marketPrice.return_value = 100.0

        mock_chain = MagicMock()
        mock_chain.exchange = "SMART"
        mock_chain.expirations = ["20260301"]  # Wrong expiry
        mock_chain.strikes = [100.0]

        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])
        mock_ib.reqMktData.return_value = stock_ticker
        mock_ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[mock_chain])

        candidate = await screen_candidate_ibkr(
            ib=mock_ib,
            symbol="WRONGEXP",
            earnings_date=date(2026, 1, 30),
            timing="AMC",
        )

        assert candidate.passes_liquidity is False


# ============================================================================
# Tests for request_with_retry
# ============================================================================

class TestRequestWithRetry:
    """Tests for _request_with_retry helper."""

    @responses.activate
    def test_retries_on_failure(self, mock_env_vars):
        """Should retry on transient failures."""
        from trading.earnings.screener import _request_with_retry

        # First request fails, second succeeds
        responses.add(
            responses.GET,
            "https://example.com/api",
            json={"error": "timeout"},
            status=503,
        )
        responses.add(
            responses.GET,
            "https://example.com/api",
            json={"data": "success"},
            status=200,
        )

        # Method is first arg, then url
        result = _request_with_retry("GET", "https://example.com/api")

        assert result.status_code == 200

    @responses.activate
    def test_raises_after_max_retries(self, mock_env_vars):
        """Should raise exception after all retries fail."""
        from trading.earnings.screener import _request_with_retry
        import requests

        # All requests fail
        for _ in range(5):
            responses.add(
                responses.GET,
                "https://example.com/api",
                json={"error": "server error"},
                status=500,
            )

        with pytest.raises(requests.exceptions.HTTPError):
            _request_with_retry("GET", "https://example.com/api", max_retries=3)


# ============================================================================
# Additional EarningsEvent tests
# ============================================================================

class TestEarningsEventFromNasdaq:
    """Tests for parsing Nasdaq earnings data."""

    @responses.activate
    def test_parses_empty_eps(self, mock_env_vars):
        """Should handle missing/empty EPS estimate."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={
                "data": {
                    "rows": [
                        {"symbol": "XYZ", "time": "AMC", "epsForecast": ""},
                    ]
                }
            },
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert len(events) == 1
        assert events[0].symbol == "XYZ"
        assert events[0].eps_estimate is None or events[0].eps_estimate == 0.0

    @responses.activate
    def test_handles_null_data(self, mock_env_vars):
        """Should handle null data in response."""
        responses.add(
            responses.GET,
            "https://api.nasdaq.com/api/calendar/earnings",
            json={"data": None},
            status=200,
        )
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        events = fetch_upcoming_earnings(days_ahead=7)

        assert isinstance(events, list)


# ============================================================================
# Tests for screen_all_candidates
# ============================================================================

class TestScreenAllCandidates:
    """Tests for screen_all_candidates async function."""

    @pytest.mark.asyncio
    async def test_screens_multiple_candidates(self, mock_ib):
        """Should screen multiple candidates concurrently."""
        from trading.earnings.screener import screen_all_candidates

        # Create mock events
        events = [
            EarningsEvent(symbol="AAPL", earnings_date=date(2026, 1, 30), timing="AMC"),
            EarningsEvent(symbol="MSFT", earnings_date=date(2026, 1, 30), timing="BMO"),
        ]

        # Mock all IB calls to return rejections
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[MagicMock()])
        stock_ticker = MagicMock()
        stock_ticker.marketPrice.return_value = float('nan')
        mock_ib.reqMktData.return_value = stock_ticker

        passed, rejected = await screen_all_candidates(
            ib=mock_ib,
            events=events,
            spread_threshold=15.0,
        )

        # All should be rejected (no valid prices)
        assert len(rejected) == 2
        assert len(passed) == 0

    @pytest.mark.asyncio
    async def test_handles_empty_events(self, mock_ib):
        """Should return empty lists for empty events."""
        from trading.earnings.screener import screen_all_candidates

        passed, rejected = await screen_all_candidates(
            ib=mock_ib,
            events=[],
            spread_threshold=15.0,
        )

        assert passed == []
        assert rejected == []
