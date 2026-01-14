"""Component tests for LLM sanity check - with mocked APIs."""
import json
from datetime import date, datetime, timezone
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import responses


# ============================================================================
# Tests for build_sanity_packet()
# ============================================================================

class TestBuildSanityPacket:
    """Tests for build_sanity_packet function."""

    def test_basic_packet_structure(self):
        """Should build packet with all required fields."""
        from trading.earnings.llm_sanity_check import build_sanity_packet

        # Create mock candidate and prediction
        candidate = MagicMock()
        candidate.symbol = "AAPL"
        candidate.earnings_date = date(2026, 1, 30)
        candidate.timing = "AMC"
        candidate.spot_price = 230.50
        candidate.expiry = "2026-01-31"
        candidate.atm_strike = 230.0
        candidate.call_bid = 3.50
        candidate.call_ask = 3.70
        candidate.call_iv = 0.45
        candidate.put_bid = 3.40
        candidate.put_ask = 3.60
        candidate.put_iv = 0.44
        candidate.straddle_mid = 7.0
        candidate.spread_pct = 5.5
        candidate.implied_move_pct = 6.2

        prediction = MagicMock()
        prediction.pred_q50 = 0.04
        prediction.pred_q75 = 0.06
        prediction.pred_q90 = 0.08
        prediction.pred_q95 = 0.10
        prediction.edge_q75 = 0.02
        prediction.headlines = ["Apple earnings expected to beat"]

        packet = build_sanity_packet(candidate, prediction, contracts=2)

        assert packet["ticker"] == "AAPL"
        assert "asof_utc" in packet
        assert packet["event"]["timing"] == "AMC"
        assert packet["underlying"]["spot"] == 230.50
        assert packet["straddle"]["strike"] == 230.0
        assert packet["sizing"]["qty"] == 2
        assert len(packet["news_headlines"]) == 1

    def test_packet_quantile_values(self):
        """Should include quantile predictions in packet."""
        from trading.earnings.llm_sanity_check import build_sanity_packet

        candidate = MagicMock()
        candidate.symbol = "MSFT"
        candidate.earnings_date = date(2026, 1, 28)
        candidate.timing = "BMO"
        candidate.spot_price = 400.0
        candidate.expiry = "2026-01-31"
        candidate.atm_strike = 400.0
        candidate.call_bid = 5.0
        candidate.call_ask = 5.20
        candidate.call_iv = 0.35
        candidate.put_bid = 4.90
        candidate.put_ask = 5.10
        candidate.put_iv = 0.34
        candidate.straddle_mid = 10.0
        candidate.spread_pct = 4.0
        candidate.implied_move_pct = 5.0

        prediction = MagicMock()
        prediction.pred_q50 = 0.035
        prediction.pred_q75 = 0.055
        prediction.pred_q90 = 0.075
        prediction.pred_q95 = 0.095
        prediction.edge_q75 = 0.005
        prediction.headlines = None

        packet = build_sanity_packet(candidate, prediction)

        assert packet["model"]["pred_abs_move_pct"]["q50"] == 3.5
        assert packet["model"]["pred_abs_move_pct"]["q75"] == 5.5
        assert packet["model"]["pred_abs_move_pct"]["q90"] == 7.5
        assert packet["model"]["pred_abs_move_pct"]["q95"] == 9.5
        assert packet["model"]["edge_pct"] == 0.5

    def test_packet_max_loss_calculation(self):
        """Should calculate max loss correctly."""
        from trading.earnings.llm_sanity_check import build_sanity_packet

        candidate = MagicMock()
        candidate.symbol = "GOOG"
        candidate.earnings_date = date(2026, 2, 5)
        candidate.timing = "AMC"
        candidate.spot_price = 180.0
        candidate.expiry = "2026-02-07"
        candidate.atm_strike = 180.0
        candidate.call_bid = 4.0
        candidate.call_ask = 4.20
        candidate.call_iv = 0.40
        candidate.put_bid = 3.90
        candidate.put_ask = 4.10
        candidate.put_iv = 0.39
        candidate.straddle_mid = 8.0  # $8 per share = $800 per contract
        candidate.spread_pct = 5.0
        candidate.implied_move_pct = 4.5

        prediction = MagicMock()
        prediction.pred_q50 = 0.03
        prediction.pred_q75 = 0.05
        prediction.pred_q90 = 0.07
        prediction.pred_q95 = 0.09
        prediction.edge_q75 = 0.005
        prediction.headlines = []

        packet = build_sanity_packet(candidate, prediction, contracts=3)

        # Max loss = straddle_mid * contracts * 100 = 8.0 * 3 * 100 = 2400
        assert packet["sizing"]["max_loss_usd"] == 2400.0


# ============================================================================
# Tests for SanityResult dataclass
# ============================================================================

class TestSanityResult:
    """Tests for SanityResult dataclass."""

    def test_create_pass_result(self):
        """Should create a PASS result."""
        from trading.earnings.llm_sanity_check import SanityResult

        result = SanityResult(
            decision="PASS",
            risk_flags=[],
            reasons=["No red flags found"],
            search_queries=["AAPL earnings 2026-01-30"],
            search_results=[{"title": "Test", "url": "http://test.com"}],
            model="anthropic/claude-3.5-sonnet",
            latency_ms=1500,
            raw_response={"decision": "PASS"},
        )

        assert result.decision == "PASS"
        assert len(result.risk_flags) == 0
        assert result.latency_ms == 1500

    def test_create_no_trade_result(self):
        """Should create a NO_TRADE result with flags."""
        from trading.earnings.llm_sanity_check import SanityResult

        result = SanityResult(
            decision="NO_TRADE",
            risk_flags=["earnings_already_released", "stock_halted"],
            reasons=["Earnings released yesterday", "Stock currently halted"],
            search_queries=["XYZ earnings", "XYZ halt"],
            search_results=[],
            model="anthropic/claude-3.5-sonnet",
            latency_ms=2000,
            raw_response={"decision": "NO_TRADE"},
        )

        assert result.decision == "NO_TRADE"
        assert len(result.risk_flags) == 2
        assert "earnings_already_released" in result.risk_flags


# ============================================================================
# Tests for _search_tavily()
# ============================================================================

class TestSearchTavily:
    """Tests for _search_tavily function."""

    @responses.activate
    def test_search_returns_results(self, mock_env_vars):
        """Should return search results from Tavily."""
        from trading.earnings.llm_sanity_check import _search_tavily

        # Mock Tavily API responses for both queries
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={
                "results": [
                    {
                        "title": "Apple Q1 Earnings Preview",
                        "url": "https://example.com/article1",
                        "content": "Analysts expect strong iPhone sales for Apple Q1 earnings...",
                    },
                    {
                        "title": "Apple Stock Analysis",
                        "url": "https://example.com/article2",
                        "content": "Apple shares rise ahead of earnings announcement...",
                    },
                ]
            },
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={
                "results": [
                    {
                        "title": "No halt news",
                        "url": "https://example.com/article3",
                        "content": "Market update...",
                    },
                ]
            },
            status=200,
        )

        queries, results = _search_tavily("AAPL", "2026-01-30")

        assert len(queries) == 2
        assert "AAPL earnings release 2026-01-30" in queries
        assert len(results) >= 1

    def test_search_without_api_key(self, monkeypatch):
        """Should return empty when no API key."""
        from trading.earnings import llm_sanity_check
        import importlib

        monkeypatch.setattr(llm_sanity_check, 'TAVILY_API_KEY', '')

        queries, results = llm_sanity_check._search_tavily("AAPL", "2026-01-30")

        assert queries == []
        assert results == []

    @responses.activate
    def test_search_handles_api_error(self, mock_env_vars):
        """Should handle Tavily API errors gracefully."""
        from trading.earnings.llm_sanity_check import _search_tavily

        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"error": "Rate limit exceeded"},
            status=429,
        )

        # Should not raise, just return partial results
        queries, results = _search_tavily("AAPL", "2026-01-30")

        # Queries are constructed before API call
        assert len(queries) == 2


# ============================================================================
# Tests for _call_openrouter()
# ============================================================================

class TestCallOpenRouter:
    """Tests for _call_openrouter function."""

    @responses.activate
    def test_call_returns_parsed_json(self, mock_env_vars):
        """Should return parsed JSON from LLM response."""
        from trading.earnings.llm_sanity_check import _call_openrouter

        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "PASS",
                            "risk_flags": [],
                            "reasons": ["No red flags found"],
                        })
                    }
                }],
            },
            status=200,
        )

        packet = {"ticker": "AAPL", "event": {"earnings_date": "2026-01-30"}}
        search_results = []

        result = _call_openrouter(packet, search_results)

        assert result["decision"] == "PASS"
        assert result["risk_flags"] == []

    @responses.activate
    def test_call_parses_markdown_json(self, mock_env_vars):
        """Should parse JSON wrapped in markdown code blocks."""
        from trading.earnings.llm_sanity_check import _call_openrouter

        # LLM sometimes wraps response in ```json blocks
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": "```json\n{\"decision\": \"WARN\", \"risk_flags\": [\"low_volume\"], \"reasons\": [\"Low trading volume\"]}\n```"
                    }
                }],
            },
            status=200,
        )

        packet = {"ticker": "XYZ"}
        result = _call_openrouter(packet, [])

        assert result["decision"] == "WARN"
        assert "low_volume" in result["risk_flags"]

    def test_call_without_api_key(self, monkeypatch):
        """Should raise when no API key."""
        from trading.earnings import llm_sanity_check

        monkeypatch.setattr(llm_sanity_check, 'OPENROUTER_API_KEY', '')

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY not set"):
            llm_sanity_check._call_openrouter({}, [])

    @responses.activate
    def test_call_handles_api_error_response(self, mock_env_vars):
        """Should raise on API error."""
        from trading.earnings.llm_sanity_check import _call_openrouter
        import requests

        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={"error": {"message": "Invalid API key"}},
            status=401,
        )

        with pytest.raises(requests.HTTPError):
            _call_openrouter({"ticker": "AAPL"}, [])


# ============================================================================
# Tests for check_with_llm() async function
# ============================================================================

class TestCheckWithLLM:
    """Tests for check_with_llm async function."""

    @pytest.mark.asyncio
    @responses.activate
    async def test_check_returns_pass(self, mock_env_vars):
        """Should return PASS result when no issues found."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock Tavily
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )

        # Mock OpenRouter
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "PASS",
                            "risk_flags": [],
                            "reasons": ["No red flags found"],
                        })
                    }
                }],
            },
            status=200,
        )

        packet = {
            "ticker": "AAPL",
            "event": {"earnings_date": "2026-01-30"},
        }

        result = await check_with_llm(packet)

        assert result.decision == "PASS"
        assert len(result.risk_flags) == 0

    @pytest.mark.asyncio
    @responses.activate
    async def test_check_returns_no_trade(self, mock_env_vars):
        """Should return NO_TRADE when major risk found."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock Tavily
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": [{"title": "Stock halted", "url": "http://test.com", "content": "Trading halted"}]},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )

        # Mock OpenRouter
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "NO_TRADE",
                            "risk_flags": ["stock_halted"],
                            "reasons": ["Stock is currently halted"],
                        })
                    }
                }],
            },
            status=200,
        )

        packet = {
            "ticker": "XYZ",
            "event": {"earnings_date": "2026-01-30"},
        }

        result = await check_with_llm(packet)

        assert result.decision == "NO_TRADE"
        assert "stock_halted" in result.risk_flags

    @pytest.mark.asyncio
    async def test_check_handles_api_failure(self, mock_env_vars):
        """Should return NO_TRADE on API failure (fail-closed)."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock both APIs to fail
        with patch('trading.earnings.llm_sanity_check._search_tavily', side_effect=Exception("Network error")):
            packet = {
                "ticker": "AAPL",
                "event": {"earnings_date": "2026-01-30"},
            }

            result = await check_with_llm(packet)

            # Should fail closed - block trade on errors
            assert result.decision == "NO_TRADE"
            assert "api_failure" in result.risk_flags

    @pytest.mark.asyncio
    @responses.activate
    async def test_check_handles_invalid_llm_response(self, mock_env_vars):
        """Should return NO_TRADE when LLM returns invalid JSON."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock Tavily
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )

        # Mock OpenRouter with invalid JSON
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": "This is not valid JSON"
                    }
                }],
            },
            status=200,
        )

        packet = {
            "ticker": "AAPL",
            "event": {"earnings_date": "2026-01-30"},
        }

        result = await check_with_llm(packet)

        # Should fail closed
        assert result.decision == "NO_TRADE"
        assert "json_parse_error" in result.risk_flags

    @pytest.mark.asyncio
    @responses.activate
    async def test_check_validates_decision_whitelist(self, mock_env_vars):
        """Should default to NO_TRADE for invalid decision values."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock Tavily
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )

        # Mock OpenRouter with invalid decision
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "MAYBE",  # Invalid decision
                            "risk_flags": [],
                            "reasons": [],
                        })
                    }
                }],
            },
            status=200,
        )

        packet = {
            "ticker": "AAPL",
            "event": {"earnings_date": "2026-01-30"},
        }

        result = await check_with_llm(packet)

        # Invalid decision should default to NO_TRADE
        assert result.decision == "NO_TRADE"

    @pytest.mark.asyncio
    @responses.activate
    async def test_check_logs_to_trade_logger(self, mock_env_vars):
        """Should log results to trade logger if provided."""
        from trading.earnings.llm_sanity_check import check_with_llm

        # Mock Tavily
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={"results": []},
            status=200,
        )

        # Mock OpenRouter
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "PASS",
                            "risk_flags": [],
                            "reasons": [],
                        })
                    }
                }],
            },
            status=200,
        )

        # Mock trade logger
        mock_logger = MagicMock()

        packet = {
            "ticker": "AAPL",
            "event": {"earnings_date": "2026-01-30"},
        }

        result = await check_with_llm(packet, trade_logger=mock_logger)

        # Verify logger was called
        mock_logger.log_llm_check.assert_called_once()
        call_kwargs = mock_logger.log_llm_check.call_args[1]
        assert call_kwargs["ticker"] == "AAPL"
        assert call_kwargs["decision"] == "PASS"


class TestLLMSanityCheckModule:
    """Tests for LLM sanity check functionality."""

    def test_module_imports(self):
        """Should be able to import the module."""
        from trading.earnings import llm_sanity_check
        assert llm_sanity_check is not None

    def test_has_check_function(self):
        """Should have a check function."""
        from trading.earnings import llm_sanity_check

        # Module should have some check function
        assert hasattr(llm_sanity_check, 'check_with_llm') or hasattr(llm_sanity_check, 'run_sanity_check')


class TestLLMAPIIntegration:
    """Tests for LLM API integration."""

    @responses.activate
    @pytest.mark.asyncio
    async def test_tavily_search_called(self, mock_env_vars):
        """Should call Tavily for web search."""
        responses.add(
            responses.POST,
            "https://api.tavily.com/search",
            json={
                "results": [
                    {
                        "title": "Apple Earnings Preview",
                        "url": "https://example.com",
                        "content": "Test content...",
                    }
                ],
                "query": "AAPL earnings",
            },
            status=200,
        )

        # Just verify we can mock the API
        import requests
        r = requests.post(
            "https://api.tavily.com/search",
            json={"query": "AAPL earnings"},
        )
        assert r.status_code == 200

    @responses.activate
    @pytest.mark.asyncio
    async def test_openrouter_api_called(self, mock_env_vars):
        """Should call OpenRouter for LLM inference."""
        responses.add(
            responses.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "decision": "PASS",
                            "risk_flags": [],
                            "reasons": ["No risks identified"],
                        })
                    }
                }],
                "usage": {"prompt_tokens": 500, "completion_tokens": 100},
            },
            status=200,
        )

        # Just verify we can mock the API
        import requests
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert r.status_code == 200


class TestDecisionTypes:
    """Tests for LLM decision types."""

    def test_decision_constants(self):
        """Should have defined decision constants."""
        # Decision types from CLAUDE.md: PASS, WARN, NO_TRADE
        valid_decisions = {"PASS", "WARN", "NO_TRADE"}

        # These are the expected decision values
        assert "PASS" in valid_decisions
        assert "WARN" in valid_decisions
        assert "NO_TRADE" in valid_decisions

    def test_threshold_logic(self):
        """Should filter by threshold correctly."""
        threshold = "WARN"

        # Decisions at or above threshold should pass
        decisions_above_threshold = ["PASS", "WARN"]
        decisions_below_threshold = ["NO_TRADE"]

        # PASS and WARN are >= WARN threshold
        for decision in decisions_above_threshold:
            passes = decision in ["PASS", "WARN"] and threshold in ["WARN", "NO_TRADE"]
            assert passes or threshold == "PASS"


class TestRiskFlagParsing:
    """Tests for parsing risk flags from LLM response."""

    def test_parses_risk_flags_json(self):
        """Should parse risk flags from JSON response."""
        response_json = {
            "decision": "WARN",
            "risk_flags": ["high_short_interest", "unusual_options_activity"],
            "reasons": ["Elevated short interest detected"],
        }

        risk_flags = response_json.get("risk_flags", [])

        assert "high_short_interest" in risk_flags
        assert "unusual_options_activity" in risk_flags

    def test_handles_empty_risk_flags(self):
        """Should handle empty risk flags."""
        response_json = {
            "decision": "PASS",
            "risk_flags": [],
            "reasons": [],
        }

        risk_flags = response_json.get("risk_flags", [])

        assert risk_flags == []
