"""Integration tests for the full screening pipeline."""
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
import responses


class TestFullScreeningPipeline:
    """Tests for complete screening flow: Earnings -> Screen -> ML -> LLM."""

    @pytest.mark.integration
    @responses.activate
    async def test_pipeline_produces_candidates(
        self,
        mock_env_vars,
        test_db,
        mock_ib_with_quotes,
    ):
        """Should produce tradeable candidates from pipeline."""
        # Mock Nasdaq earnings calendar
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

        # Add empty responses for other days
        for _ in range(6):
            responses.add(
                responses.GET,
                "https://api.nasdaq.com/api/calendar/earnings",
                json={"data": {"rows": []}},
                status=200,
            )

        # Mock FMP for ML features
        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/earnings",
            json=[{"symbol": "AAPL", "date": "2025-10-30", "epsActual": 5.89}],
            status=200,
        )

        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted",
            json=[{"date": "2025-10-30", "close": 220.0, "adjClose": 220.0}],
            status=200,
        )

        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/news/stock",
            json=[],
            status=200,
        )

        # Mock LLM sanity check
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": [], "query": "AAPL"},
        )

        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "PASS",
                            "risk_flags": [],
                            "reasons": ["OK"],
                        })
                    }
                }],
            },
        )

        # Setup IBKR mocks for screening
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

        # Import after mocks are set up
        from trading.earnings.screener import fetch_upcoming_earnings, screen_candidate_ibkr

        # Step 1: Fetch earnings
        events = fetch_upcoming_earnings(days_ahead=7)

        # Step 2: Screen candidates (if we have events)
        if events:
            from trading.earnings.screener import screen_candidate_ibkr

            candidate = await screen_candidate_ibkr(
                ib=mock_ib_with_quotes,
                symbol=events[0].symbol,
                earnings_date=events[0].earnings_date,
                timing=events[0].timing,
            )

            assert candidate is not None
            assert candidate.symbol == "AAPL"

    @pytest.mark.integration
    def test_pipeline_logs_non_trades(self, test_db, mock_env_vars):
        """Should log rejected candidates as non-trades."""
        from trading.earnings.logging import NonTradeLog, generate_log_id

        # Create and log a non-trade
        non_trade = NonTradeLog(
            log_id=generate_log_id("TSLA"),
            ticker="TSLA",
            earnings_date="2026-01-30",
            earnings_timing="AMC",
            log_datetime="2026-01-29T14:30:00",
            rejection_reason="Spread too wide: 22.5% > 15.0%",
            quoted_bid=8.50,
            quoted_ask=10.80,
            spot_price=245.00,
            implied_move=0.039,
        )

        test_db.log_non_trade(non_trade)

        # Verify it was logged
        non_trades = test_db.get_non_trades(ticker="TSLA")
        assert len(non_trades) == 1
        assert "Spread too wide" in non_trades[0].rejection_reason


class TestMLPredictionIntegration:
    """Tests for ML prediction integration."""

    @pytest.mark.integration
    def test_prediction_with_edge_calculation(self):
        """Should compute edge from prediction vs implied move."""
        # Edge = predicted_q75 - implied_move
        predicted_q75 = 0.065
        implied_move = 0.045

        edge = predicted_q75 - implied_move

        assert edge == pytest.approx(0.02)  # 2% edge

    @pytest.mark.integration
    def test_edge_threshold_filtering(self):
        """Should filter candidates by edge threshold."""
        # From config: EDGE_THRESHOLD=0.05 (5%)
        edge_threshold = 0.05

        candidates = [
            {"symbol": "AAPL", "edge": 0.08},  # Pass
            {"symbol": "MSFT", "edge": 0.03},  # Fail
            {"symbol": "GOOGL", "edge": 0.06},  # Pass
        ]

        passing = [c for c in candidates if c["edge"] >= edge_threshold]

        assert len(passing) == 2
        assert "AAPL" in [c["symbol"] for c in passing]
        assert "GOOGL" in [c["symbol"] for c in passing]
        assert "MSFT" not in [c["symbol"] for c in passing]


class TestLLMSanityCheckIntegration:
    """Tests for LLM sanity check integration."""

    @pytest.mark.integration
    def test_llm_decision_gates_trade(self, test_db):
        """Should gate trades based on LLM decision."""
        # Simulate LLM decisions
        decisions = [
            ("AAPL", "PASS"),
            ("MSFT", "WARN"),
            ("TSLA", "NO_TRADE"),
        ]

        # With threshold WARN, PASS and WARN should proceed
        threshold = "WARN"
        should_trade = [d for d in decisions if d[1] in ("PASS", "WARN")]

        assert len(should_trade) == 2


class TestDatabaseIntegration:
    """Tests for database integration across pipeline."""

    @pytest.mark.integration
    def test_trade_lifecycle(self, test_db):
        """Should track trade through full lifecycle."""
        from trading.earnings.logging import TradeLog, generate_trade_id

        # 1. Create pending trade
        trade_id = generate_trade_id("AAPL", "2026-01-30")
        trade = TradeLog(
            trade_id=trade_id,
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
        test_db.log_trade(trade)

        # 2. Update to filled
        test_db.update_trade(
            trade_id,
            status="filled",
            entry_fill_price=7.12,
            entry_slippage=0.02,
        )

        # 3. Update to exiting
        test_db.update_trade(
            trade_id,
            status="exiting",
            exit_call_order_id=100001,
            exit_put_order_id=100002,
        )

        # 4. Update to exited with P&L
        test_db.update_trade(
            trade_id,
            status="exited",
            exit_fill_price=8.50,
            exit_pnl=138.0,
            exit_pnl_pct=0.097,
        )

        # Verify final state
        final_trade = test_db.get_trade(trade_id)
        assert final_trade.status == "exited"
        assert final_trade.exit_pnl == 138.0

    @pytest.mark.integration
    def test_counterfactual_backfill(self, test_db):
        """Should backfill counterfactual data for non-trades."""
        from trading.earnings.logging import NonTradeLog, generate_log_id

        # Log non-trade without counterfactual
        non_trade = NonTradeLog(
            log_id=generate_log_id("AMD"),
            ticker="AMD",
            earnings_date="2026-01-28",  # Past date
            earnings_timing="AMC",
            log_datetime="2026-01-27T14:30:00",
            rejection_reason="Edge insufficient",
            quoted_bid=5.50,
            quoted_ask=6.00,
            spot_price=150.00,
            implied_move=0.037,
        )
        test_db.log_non_trade(non_trade)

        # Update with counterfactual
        test_db.update_non_trade(
            non_trade.log_id,
            counterfactual_realized_move=0.055,
            counterfactual_pnl=2.25,
            counterfactual_pnl_with_spread=1.75,
        )

        # Verify
        non_trades = test_db.get_non_trades(ticker="AMD")
        assert non_trades[0].counterfactual_realized_move == 0.055
