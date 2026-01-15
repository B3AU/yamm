"""LLM sanity check for earnings straddle trades.

Uses Tavily for web search and OpenRouter for LLM reasoning.
Validates that there are no red flags before placing a trade.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# API keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')

# Default model (can be overridden via env)
DEFAULT_MODEL = os.getenv('LLM_SANITY_MODEL', 'anthropic/claude-3.5-sonnet')

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.0  # 1s, 2s, 4s exponential backoff

# Timeouts
OPENROUTER_TIMEOUT = 90  # seconds per LLM request
OVERALL_LLM_TIMEOUT = 300  # seconds total for check_with_llm()

# App-level error codes that should be retried (OpenRouter returns 200 with error in body)
RETRYABLE_ERROR_CODES = {429, 500, 502, 503, 504, 529}  # 429=rate limit, 529=overloaded


def _get_retry_session() -> requests.Session:
    """Create a requests session with retry logic and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET", "POST"],
        raise_on_status=False,  # Don't raise, let us handle it
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class SanityResult:
    """Result from LLM sanity check."""
    decision: str  # PASS, WARN, NO_TRADE
    risk_flags: list[str]
    reasons: list[str]
    search_queries: list[str]
    search_results: list[dict]
    model: str
    latency_ms: int
    raw_response: dict


def build_sanity_packet(
    candidate,  # ScreenedCandidate
    prediction,  # EdgePrediction
    contracts: int = 1,
) -> dict:
    """Build the JSON packet for LLM sanity check."""
    return {
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": candidate.symbol,
        "event": {
            "earnings_date": candidate.earnings_date.isoformat(),
            "timing": candidate.timing,
        },
        "underlying": {
            "spot": candidate.spot_price,
        },
        "model": {
            "pred_abs_move_pct": {
                "q50": round(prediction.pred_q50 * 100, 1),
                "q75": round(prediction.pred_q75 * 100, 1),
                "q90": round(prediction.pred_q90 * 100, 1),
                "q95": round(prediction.pred_q95 * 100, 1),
            },
            "edge_pct": round(prediction.edge_q75 * 100, 1),
        },
        "straddle": {
            "expiry": candidate.expiry,
            "strike": candidate.atm_strike,
            "call": {
                "bid": candidate.call_bid,
                "ask": candidate.call_ask,
                "iv": candidate.call_iv,
            },
            "put": {
                "bid": candidate.put_bid,
                "ask": candidate.put_ask,
                "iv": candidate.put_iv,
            },
            "mid": candidate.straddle_mid,
            "spread_pct": round(candidate.spread_pct, 1),
            "implied_move_pct": round(candidate.implied_move_pct, 1),
        },
        "sizing": {
            "qty": contracts,
            "max_loss_usd": round(candidate.straddle_mid * contracts * 100, 0),
        },
        "news_headlines": prediction.headlines or [],
    }


def _search_tavily(ticker: str, earnings_date: str) -> tuple[list[str], list[dict]]:
    """Perform web search using Tavily API.

    Returns (queries_used, search_results).
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set, skipping web search")
        return [], []

    queries = [
        f"{ticker} earnings release {earnings_date}",
        f"{ticker} stock halt offering acquisition",
    ]

    all_results = []

    session = _get_retry_session()
    for query in queries:
        try:
            response = session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            for result in data.get("results", []):
                all_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:500],  # Truncate
                    "query": query,
                })

        except Exception as e:
            logger.error(f"Tavily search error for '{query}': {e}")

    return queries, all_results


def _call_openrouter(packet: dict, search_results: list[dict]) -> dict:
    """Call OpenRouter LLM with the sanity check prompt.

    Returns parsed JSON response or raises exception.
    Retries on transient app-level errors (500, 502, 503, 504, 429, 529).
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Build prompt
    system_prompt = """You are a trading risk analyst. Your job is to review proposed earnings straddle trades and identify any red flags that would make the trade risky.

You will receive:
1. Trade details (ticker, earnings date, pricing)
2. Recent news headlines from financial sources
3. Web search results

Your task is to check for:
- Has earnings already been released? (check dates carefully)
- Is there a stock halt, trading suspension, or delisting risk?
- Is there a recent or pending offering, secondary, or dilution?
- Is there an acquisition, merger, or buyout pending?
- Any other material event that makes an earnings straddle inappropriate?

Respond with JSON only:
{
  "decision": "PASS|WARN|NO_TRADE",
  "risk_flags": ["FLAG1", "FLAG2"],
  "reasons": ["Explanation of each flag"]
}

Decision guidelines:
- PASS: No red flags found, trade looks safe
- WARN: Minor concerns but trade can proceed (e.g., low news coverage, unconfirmed timing)
- NO_TRADE: Major red flag found (e.g., earnings already released, stock halted, pending acquisition)

Be concise. Only flag real issues, not hypothetical ones."""

    user_message = f"""Review this proposed earnings straddle trade:

TRADE DETAILS:
{json.dumps(packet, indent=2)}

WEB SEARCH RESULTS:
{json.dumps(search_results, indent=2) if search_results else "No web search results available."}

Analyze and respond with JSON only."""

    session = _get_retry_session()

    for attempt in range(MAX_RETRIES + 1):
        attempt_label = f" (attempt {attempt + 1}/{MAX_RETRIES + 1})" if attempt > 0 else ""
        logger.info(f"Calling OpenRouter with model: {DEFAULT_MODEL}{attempt_label}")

        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.1,  # Low temperature for consistent output
                "max_tokens": 5000,  # Enough for reasoning models + response
            },
            timeout=OPENROUTER_TIMEOUT,
        )

        # Log response status for debugging
        if not response.ok:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text[:500]}")
        response.raise_for_status()

        data = response.json()

        # Check for app-level errors (OpenRouter sometimes returns 200 with error in body)
        if "error" in data:
            error_info = data["error"]
            error_code = error_info.get("code", 0) if isinstance(error_info, dict) else 0
            error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)

            # Check if this error is retryable
            if error_code in RETRYABLE_ERROR_CODES and attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"OpenRouter app-level error{attempt_label}, "
                    f"retrying in {delay:.1f}s: [{error_code}] {error_msg}"
                )
                time.sleep(delay)
                continue

            # Non-retryable or exhausted retries
            raise ValueError(f"OpenRouter error: {error_info}")

        # Extract content
        if "choices" not in data or not data["choices"]:
            logger.error(f"Unexpected OpenRouter response: {data}")
            raise ValueError("No choices in OpenRouter response")

        content = data["choices"][0]["message"]["content"]

        if not content:
            logger.error(f"Empty content in OpenRouter response: {data}")
            raise ValueError("Empty content from LLM")

        logger.debug(f"Raw LLM response: {content[:200]}...")

        # Parse JSON from response (handle markdown code blocks safely)
        json_content = content
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                inner_parts = parts[1].split("```")
                if len(inner_parts) > 0:
                    json_content = inner_parts[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) > 2:
                json_content = parts[1]

        try:
            return json.loads(json_content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed. Raw content: {content[:500]}")
            raise  # Re-raise to be caught by outer handler with json_parse_error flag

    # Should never reach here, but just in case
    raise ValueError("OpenRouter failed after all retries")


async def check_with_llm(
    packet: dict,
    trade_logger=None,  # TradeLogger for logging results
    ticker: Optional[str] = None,
) -> SanityResult:
    """Run LLM sanity check on a trade packet.

    Args:
        packet: Trade details from build_sanity_packet()
        trade_logger: Optional TradeLogger to log results
        ticker: Ticker symbol (extracted from packet if not provided)

    Returns:
        SanityResult with decision and details
    """
    ticker = ticker or packet.get("ticker", "UNKNOWN")
    start_time = time.time()

    # Default result for errors - fail-closed for safety
    default_result = SanityResult(
        decision="NO_TRADE",  # Fail closed - block trades on LLM errors to prevent trading halted stocks etc.
        risk_flags=["api_failure"],
        reasons=["LLM check failed, blocking trade for safety"],
        search_queries=[],
        search_results=[],
        model=DEFAULT_MODEL,
        latency_ms=0,
        raw_response={},
    )

    try:
        # Wrap in overall timeout to prevent indefinite hangs
        return await asyncio.wait_for(
            _check_with_llm_inner(packet, trade_logger, ticker, start_time),
            timeout=OVERALL_LLM_TIMEOUT,
        )

    except asyncio.TimeoutError:
        elapsed = int(time.time() - start_time)
        logger.error(f"{ticker}: LLM check timed out after {elapsed}s (limit: {OVERALL_LLM_TIMEOUT}s)")
        default_result.reasons = [f"Overall timeout after {elapsed}s", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "overall_timeout"]
        default_result.latency_ms = elapsed * 1000
        return default_result

    except Exception as e:
        # Catch-all for any unexpected errors not handled by inner function
        logger.exception(f"{ticker}: Unexpected error in LLM sanity check: {e}")
        default_result.reasons = [f"Unexpected error: {e}", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "unexpected_error"]
        default_result.latency_ms = int((time.time() - start_time) * 1000)
        return default_result


async def _check_with_llm_inner(
    packet: dict,
    trade_logger,
    ticker: str,
    start_time: float,
) -> SanityResult:
    """Inner implementation of check_with_llm (extracted for timeout wrapper).

    This function contains the actual LLM check logic. It's separated to allow
    the outer function to wrap it with asyncio.wait_for() for overall timeout.
    """
    # Default result for errors - fail-closed for safety
    default_result = SanityResult(
        decision="NO_TRADE",
        risk_flags=["api_failure"],
        reasons=["LLM check failed, blocking trade for safety"],
        search_queries=[],
        search_results=[],
        model=DEFAULT_MODEL,
        latency_ms=0,
        raw_response={},
    )

    try:
        # Step 1: Web search (blocking I/O - run in thread)
        earnings_date = packet.get("event", {}).get("earnings_date", "")
        queries, search_results = await asyncio.to_thread(_search_tavily, ticker, earnings_date)

        # Step 2: LLM analysis (blocking I/O - run in thread)
        llm_response = await asyncio.to_thread(_call_openrouter, packet, search_results)

        latency_ms = int((time.time() - start_time) * 1000)

        # Validate decision against whitelist (fail-closed on invalid values)
        VALID_DECISIONS = {"PASS", "WARN", "NO_TRADE"}
        raw_decision = llm_response.get("decision", "")
        if raw_decision not in VALID_DECISIONS:
            logger.warning(f"{ticker}: Invalid LLM decision '{raw_decision}', defaulting to NO_TRADE")
            raw_decision = "NO_TRADE"

        # Ensure risk_flags and reasons are lists
        risk_flags = llm_response.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = []
        reasons = llm_response.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []

        result = SanityResult(
            decision=raw_decision,
            risk_flags=risk_flags,
            reasons=reasons,
            search_queries=queries,
            search_results=search_results,
            model=DEFAULT_MODEL,
            latency_ms=latency_ms,
            raw_response=llm_response,
        )

        logger.info(
            f"{ticker}: LLM sanity check -> {result.decision} "
            f"(flags={result.risk_flags}, latency={latency_ms}ms)"
        )

        # Log to database if logger provided
        if trade_logger:
            try:
                trade_logger.log_llm_check(
                    ticker=ticker,
                    decision=result.decision,
                    risk_flags=result.risk_flags,
                    reasons=result.reasons,
                    search_queries=queries,
                    search_results=search_results,
                    packet=packet,
                    response=llm_response,
                    latency_ms=latency_ms,
                    model=DEFAULT_MODEL,
                )
            except Exception as e:
                logger.error(f"Failed to log LLM check: {e}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"{ticker}: LLM response not valid JSON: {e}")
        default_result.reasons = [f"LLM response parse error: {e}", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "json_parse_error"]
        return default_result

    except requests.Timeout as e:
        logger.error(f"{ticker}: LLM API timeout after {OPENROUTER_TIMEOUT}s: {e}")
        default_result.reasons = [f"LLM API timeout after {OPENROUTER_TIMEOUT}s: {e}", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "timeout"]
        return default_result

    except requests.RequestException as e:
        logger.error(f"{ticker}: LLM API error: {e}")
        default_result.reasons = [f"LLM API error: {e}", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "request_error"]
        return default_result

    except Exception as e:
        logger.exception(f"{ticker}: Unexpected error in LLM sanity check: {e}")
        default_result.reasons = [f"Unexpected error: {e}", "Trade blocked for safety"]
        default_result.risk_flags = ["api_failure", "unexpected_error"]
        return default_result
