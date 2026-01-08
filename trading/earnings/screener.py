"""Earnings screener - finds tradeable candidates for Phase 0.

Combines:
- Nasdaq earnings calendar (upcoming earnings with BMO/AMC timing)
- IBKR option quotes (real-time bid/ask, spreads, OI)
- Liquidity gates from V1 plan
"""
from __future__ import annotations

import os
import requests
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Headers to mimic browser for Nasdaq API
NASDAQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nasdaq.com',
    'Referer': 'https://www.nasdaq.com/',
}


@dataclass
class EarningsEvent:
    """Upcoming earnings announcement."""
    symbol: str
    earnings_date: date
    timing: str  # 'BMO', 'AMC', or 'unknown'
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None


@dataclass
class ScreenedCandidate:
    """A candidate that passed initial screening."""
    symbol: str
    earnings_date: date
    timing: str
    spot_price: float

    # Option details
    expiry: str
    atm_strike: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float

    # Computed metrics
    straddle_mid: float
    straddle_spread: float
    spread_pct: float
    implied_move_pct: float

    # IV if available
    call_iv: Optional[float] = None
    put_iv: Optional[float] = None

    # ML predictions
    pred_q75: Optional[float] = None
    hist_move_mean: Optional[float] = None
    edge_q75: Optional[float] = None

    # Why it passed/failed
    passes_liquidity: bool = False
    passes_edge: bool = False
    rejection_reason: Optional[str] = None


def fetch_upcoming_earnings(days_ahead: int = 7) -> list[EarningsEvent]:
    """Fetch upcoming earnings from Nasdaq API (has BMO/AMC timing)."""
    from_date = date.today()
    to_date = from_date + timedelta(days=days_ahead)

    events = []

    # Nasdaq API requires fetching one day at a time
    current_date = from_date
    while current_date <= to_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://api.nasdaq.com/api/calendar/earnings?date={date_str}"

        try:
            r = requests.get(url, headers=NASDAQ_HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()

            rows = data.get('data', {}).get('rows', [])
            if not rows:
                current_date += timedelta(days=1)
                continue

            for row in rows:
                symbol = row.get('symbol', '')

                # Filter to US stocks (no dots/dashes in symbol)
                if '.' in symbol or '-' in symbol or not symbol:
                    continue

                # Parse timing from Nasdaq format
                time_str = row.get('time', '').lower()
                if 'pre-market' in time_str or 'before' in time_str:
                    timing = 'BMO'
                elif 'after-hours' in time_str or 'after' in time_str:
                    timing = 'AMC'
                else:
                    timing = 'unknown'

                # Parse EPS estimate
                eps_estimate = None
                eps_str = row.get('epsForecast', '')
                if eps_str:
                    try:
                        # Remove $ and parentheses for negative
                        eps_clean = eps_str.replace('$', '').replace(',', '')
                        if '(' in eps_clean:
                            eps_clean = '-' + eps_clean.replace('(', '').replace(')', '')
                        eps_estimate = float(eps_clean)
                    except (ValueError, TypeError):
                        pass

                events.append(EarningsEvent(
                    symbol=symbol,
                    earnings_date=current_date,
                    timing=timing,
                    eps_estimate=eps_estimate,
                    revenue_estimate=None,  # Nasdaq doesn't provide this
                ))

        except Exception as e:
            logger.warning(f"Failed to fetch earnings for {date_str}: {e}")

        current_date += timedelta(days=1)

    logger.info(f"Fetched {len(events)} earnings events from Nasdaq API")
    return events


async def screen_candidate_ibkr(
    ib,  # IB connection
    symbol: str,
    earnings_date: date,
    timing: str,
    spread_threshold: float = 15.0,  # max spread %
    min_oi: int = 50,  # minimum open interest (can't check via mkt data, skip for now)
) -> ScreenedCandidate:
    """
    Screen a single candidate using IBKR for option quotes (Async).

    Returns ScreenedCandidate with passes_liquidity set appropriately.
    """
    from ib_insync import Stock, Option

    # Get stock price
    stock = Stock(symbol, 'SMART', 'USD')
    try:
        await ib.qualifyContractsAsync(stock)
    except Exception as e:
        return _rejected_candidate(symbol, earnings_date, timing, f"Could not qualify stock: {e}")

    ticker = ib.reqMktData(stock, '', False, False)

    # Wait for price (max 2s)
    for _ in range(20):
        if ticker.last == ticker.last and ticker.last > 0:
            break
        if ticker.close == ticker.close and ticker.close > 0:
            break
        await asyncio.sleep(0.1)

    spot = ticker.marketPrice()
    if math.isnan(spot) or spot <= 0:  # nan check
        spot = ticker.last
    if math.isnan(spot) or spot <= 0:
        spot = ticker.close

    ib.cancelMktData(stock)

    if not spot or spot <= 0 or math.isnan(spot):
        return _rejected_candidate(symbol, earnings_date, timing, "No spot price")

    # Get option chain
    try:
        # Check if async version exists, otherwise use reliable synchronous wrapper via to_thread?
        # ib_insync usually provides Async methods.
        chains = await ib.reqSecDefOptParamsAsync(stock.symbol, '', stock.secType, stock.conId)
    except Exception as e:
        return _rejected_candidate(symbol, earnings_date, timing, f"No option chain: {e}")

    if not chains:
        return _rejected_candidate(symbol, earnings_date, timing, "No option chain")

    chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])

    # Find expiry after earnings
    expiries = sorted(chain.expirations)
    target_expiry = None
    for exp in expiries:
        exp_date = datetime.strptime(exp, '%Y%m%d').date()
        if exp_date > earnings_date:
            target_expiry = exp
            break

    if not target_expiry:
        return _rejected_candidate(symbol, earnings_date, timing, "No expiry after earnings")

    # Find ATM strike
    strikes = sorted(chain.strikes)
    atm_strike = min(strikes, key=lambda s: abs(s - spot))

    # Get option quotes
    call = Option(symbol, target_expiry, atm_strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, target_expiry, atm_strike, 'P', 'SMART', tradingClass=symbol)

    try:
        qualified = await ib.qualifyContractsAsync(call, put)
        if len(qualified) < 2:
            return _rejected_candidate(symbol, earnings_date, timing, "Could not qualify options")
    except Exception as e:
        return _rejected_candidate(symbol, earnings_date, timing, f"Option qualification error: {e}")

    call_ticker = ib.reqMktData(call, '', False, False)
    put_ticker = ib.reqMktData(put, '', False, False)

    # Wait for quotes (max 2s)
    for _ in range(20):
        if (call_ticker.bid > 0 and call_ticker.ask > 0 and
            put_ticker.bid > 0 and put_ticker.ask > 0):
            break
        await asyncio.sleep(0.1)

    # Extract bid/ask
    def _valid(p) -> float:
        return 0.0 if (p is None or math.isnan(p)) else float(p)

    call_bid = _valid(call_ticker.bid)
    call_ask = _valid(call_ticker.ask)
    put_bid = _valid(put_ticker.bid)
    put_ask = _valid(put_ticker.ask)

    # Get IV if available
    call_iv = None
    put_iv = None
    if call_ticker.modelGreeks:
        call_iv = call_ticker.modelGreeks.impliedVol
    if put_ticker.modelGreeks:
        put_iv = put_ticker.modelGreeks.impliedVol

    ib.cancelMktData(call)
    ib.cancelMktData(put)

    # Check if we have valid quotes
    if call_bid <= 0 or call_ask <= 0 or put_bid <= 0 or put_ask <= 0:
        return _rejected_candidate(
            symbol, earnings_date, timing, "No valid bid/ask",
            spot_price=spot, expiry=target_expiry, atm_strike=atm_strike,
            call_bid=call_bid, call_ask=call_ask, put_bid=put_bid, put_ask=put_ask
        )

    # Calculate metrics
    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2
    straddle_mid = call_mid + put_mid
    straddle_spread = (call_ask - call_bid) + (put_ask - put_bid)
    spread_pct = (straddle_spread / straddle_mid * 100) if straddle_mid > 0 else 100
    implied_move_pct = (straddle_mid / spot * 100) if spot > 0 else 0

    # Check liquidity gate
    passes = spread_pct <= spread_threshold
    rejection = None if passes else f"Spread too wide: {spread_pct:.1f}% > {spread_threshold}%"

    return ScreenedCandidate(
        symbol=symbol,
        earnings_date=earnings_date,
        timing=timing,
        spot_price=spot,
        expiry=target_expiry,
        atm_strike=atm_strike,
        call_bid=call_bid,
        call_ask=call_ask,
        put_bid=put_bid,
        put_ask=put_ask,
        straddle_mid=straddle_mid,
        straddle_spread=straddle_spread,
        spread_pct=spread_pct,
        implied_move_pct=implied_move_pct,
        call_iv=call_iv,
        put_iv=put_iv,
        passes_liquidity=passes,
        rejection_reason=rejection,
    )


def _rejected_candidate(
    symbol: str,
    earnings_date: date,
    timing: str,
    reason: str,
    **kwargs
) -> ScreenedCandidate:
    """Helper to create a rejected candidate."""
    return ScreenedCandidate(
        symbol=symbol,
        earnings_date=earnings_date,
        timing=timing,
        spot_price=kwargs.get('spot_price', 0),
        expiry=kwargs.get('expiry', ''),
        atm_strike=kwargs.get('atm_strike', 0),
        call_bid=kwargs.get('call_bid', 0),
        call_ask=kwargs.get('call_ask', 0),
        put_bid=kwargs.get('put_bid', 0),
        put_ask=kwargs.get('put_ask', 0),
        straddle_mid=0,
        straddle_spread=0,
        spread_pct=100,
        implied_move_pct=0,
        passes_liquidity=False,
        rejection_reason=reason,
    )


async def screen_all_candidates(
    ib,
    events: list[EarningsEvent],
    spread_threshold: float = 15.0,
    max_candidates: int = 50,
    skip_symbols: set[str] = None,
) -> tuple[list[ScreenedCandidate], list[ScreenedCandidate]]:
    """
    Screen all earnings events (Async).
    Uses semaphores to parallelize IBKR requests without overloading.

    Returns:
        (passed, rejected) - lists of candidates
    """
    skip_symbols = skip_symbols or set()

    passed = []
    rejected = []

    # If max_candidates is 0 or None, screen all
    events_to_screen = events if (max_candidates is None or max_candidates <= 0) else events[:max_candidates]
    total_to_screen = len(events_to_screen)

    # Use semaphore to limit concurrent IB requests (IBKR Gateway can handle ~50 concurrent)
    # Be conservative with 10 to avoid pacing violations
    sem = asyncio.Semaphore(10)

    async def _screen_safe(event):
        if event.symbol in skip_symbols:
            return None

        async with sem:
            logger.info(f"Screening {event.symbol}...")
            return await screen_candidate_ibkr(
                ib,
                event.symbol,
                event.earnings_date,
                event.timing,
                spread_threshold=spread_threshold,
            )

    # Launch all tasks
    tasks = [_screen_safe(event) for event in events_to_screen]
    results = await asyncio.gather(*tasks)

    for candidate in results:
        if not candidate:
            continue

        if candidate.passes_liquidity:
            passed.append(candidate)
        else:
            rejected.append(candidate)

    return passed, rejected
