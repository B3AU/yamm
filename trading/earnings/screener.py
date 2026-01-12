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
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional
import logging

from dotenv import load_dotenv
import pytz

load_dotenv()
logger = logging.getLogger(__name__)


def _request_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    **kwargs
) -> requests.Response:
    """Make HTTP request with exponential backoff retry on failure.

    Args:
        method: HTTP method ('GET', 'POST', etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        backoff_factor: Base delay multiplier for exponential backoff
        **kwargs: Additional arguments passed to requests.request()

    Returns:
        Response object

    Raises:
        requests.RequestException: If all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = backoff_factor * (2 ** attempt)  # 1s, 2s, 4s...
                logger.warning(f"Request to {url} failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Request to {url} failed after {max_retries} attempts: {e}")

    raise last_exception

# Timezone for earnings timing
ET = pytz.timezone('US/Eastern')

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

    # ML predictions (all quantiles for calibration tracking)
    pred_q50: Optional[float] = None
    pred_q75: Optional[float] = None
    pred_q90: Optional[float] = None
    pred_q95: Optional[float] = None
    hist_move_mean: Optional[float] = None
    edge_q75: Optional[float] = None
    edge_q90: Optional[float] = None
    news_count: Optional[int] = None  # number of FMP news articles found

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
            r = _request_with_retry('GET', url, headers=NASDAQ_HEADERS, timeout=10)
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


def fetch_fmp_earnings(from_date: date = None, to_date: date = None, days_ahead: int = 7) -> list[EarningsEvent]:
    """Fetch upcoming earnings from FMP API."""
    FMP_API_KEY = os.getenv('FMP_API_KEY', '')
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not set, skipping FMP earnings fetch")
        return []

    if from_date is None:
        from_date = date.today()
    if to_date is None:
        to_date = from_date + timedelta(days=days_ahead)

    url = "https://financialmodelingprep.com/stable/earnings-calendar"
    params = {
        'from': from_date.isoformat(),
        'to': to_date.isoformat(),
        'apikey': FMP_API_KEY
    }

    events = []
    try:
        r = _request_with_retry('GET', url, params=params, timeout=15)
        data = r.json()

        for row in data:
            symbol = row.get('symbol', '')

            # Filter to US stocks
            if '.' in symbol or '-' in symbol or not symbol:
                continue

            # Parse date
            date_str = row.get('date', '')
            if not date_str:
                continue
            try:
                earnings_dt = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                continue

            # FMP doesn't provide timing (BMO/AMC) consistently
            timing = 'unknown'

            # Parse estimates
            eps_estimate = row.get('epsEstimated')
            revenue_estimate = row.get('revenueEstimated')

            events.append(EarningsEvent(
                symbol=symbol,
                earnings_date=earnings_dt,
                timing=timing,
                eps_estimate=eps_estimate,
                revenue_estimate=revenue_estimate,
            ))

    except Exception as e:
        logger.warning(f"Failed to fetch FMP earnings: {e}")

    logger.info(f"Fetched {len(events)} earnings events from FMP API")
    return events


def fetch_timing_from_yfinance(symbol: str) -> Optional[str]:
    """Get BMO/AMC timing from yfinance earningsTimestamp.

    Returns 'BMO' if earnings before 10 AM ET, 'AMC' if after 4 PM ET,
    'unknown' for during market hours, or None if no data.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        ts = ticker.info.get('earningsTimestamp')
        if not ts:
            return None

        dt = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(ET)
        if dt.hour < 10:
            return 'BMO'
        elif dt.hour >= 16:
            return 'AMC'
        return 'unknown'  # During market hours (rare)
    except Exception:
        return None


def fetch_yahoo_earnings(
    from_date: date = None,
    to_date: date = None,
    days_ahead: int = 7,
    symbols: list[str] = None,
) -> list[EarningsEvent]:
    """Fetch earnings from Yahoo Finance using earningsTimestamp.

    Uses info['earningsTimestamp'] instead of calendar for more accurate dates,
    and derives BMO/AMC timing from the hour.

    If symbols is provided, checks those specific symbols.
    Otherwise, uses symbols from Nasdaq calendar as a reference list.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, skipping Yahoo earnings fetch")
        return []

    if from_date is None:
        from_date = date.today()
    if to_date is None:
        to_date = from_date + timedelta(days=days_ahead)

    # If no symbols provided, get them from Nasdaq
    if symbols is None:
        nasdaq_events = fetch_upcoming_earnings(days_ahead)
        symbols = list(set(e.symbol for e in nasdaq_events))

    events = []
    checked = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            ts = info.get('earningsTimestamp')
            if not ts:
                checked += 1
                continue

            # Convert timestamp to datetime in ET
            dt = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(ET)
            earnings_dt = dt.date()

            # Check if within range
            if not (from_date <= earnings_dt <= to_date):
                checked += 1
                continue

            # Derive timing from hour
            if dt.hour < 10:
                timing = 'BMO'
            elif dt.hour >= 16:
                timing = 'AMC'
            else:
                timing = 'unknown'

            # Get estimates from calendar if available
            cal = ticker.calendar
            eps_est = cal.get('Earnings Average') if cal else None
            rev_est = cal.get('Revenue Average') if cal else None

            events.append(EarningsEvent(
                symbol=symbol,
                earnings_date=earnings_dt,
                timing=timing,
                eps_estimate=eps_est,
                revenue_estimate=rev_est,
            ))

            checked += 1
            if checked % 50 == 0:
                logger.debug(f"Yahoo: checked {checked}/{len(symbols)} symbols")

        except Exception as e:
            logger.debug(f"Yahoo fetch failed for {symbol}: {e}")
            checked += 1
            continue

    logger.info(f"Fetched {len(events)} earnings events from Yahoo Finance (earningsTimestamp, checked {checked} symbols)")
    return events


def fetch_all_earnings_sources(
    days_ahead: int = 7,
    trade_logger=None,
) -> dict[str, list[EarningsEvent]]:
    """
    Fetch earnings from all sources and optionally log to DB.

    Returns dict mapping source name to list of events.
    """
    from_date = date.today()
    to_date = from_date + timedelta(days=days_ahead)

    results = {}

    # Nasdaq (primary source)
    results['nasdaq'] = fetch_upcoming_earnings(days_ahead)

    # FMP
    results['fmp'] = fetch_fmp_earnings(from_date, to_date)

    # Yahoo
    results['yahoo'] = fetch_yahoo_earnings(from_date, to_date)

    # Log to database if logger provided
    if trade_logger:
        for source, events in results.items():
            for event in events:
                try:
                    trade_logger.log_earnings_calendar(
                        symbol=event.symbol,
                        earnings_date=event.earnings_date,
                        timing=event.timing,
                        source=source,
                        eps_estimate=event.eps_estimate,
                        revenue_estimate=event.revenue_estimate,
                    )
                except Exception as e:
                    logger.debug(f"Failed to log {event.symbol} from {source}: {e}")

    # Summary
    for source, events in results.items():
        logger.info(f"{source}: {len(events)} events")

    return results


def verify_earnings_date(symbol: str, nasdaq_date: date, trade_logger) -> tuple[bool, Optional[str]]:
    """Verify Nasdaq earnings date against FMP.

    Returns:
        (is_verified, rejection_reason)
        - (True, None) if dates agree or FMP has no data
        - (False, reason) if dates disagree
    """
    # Query FMP date from DB (already fetched by fetch_all_earnings_sources)
    fmp_events = trade_logger.get_earnings_calendar(
        from_date=nasdaq_date - timedelta(days=3),
        to_date=nasdaq_date + timedelta(days=3),
        source='fmp'
    )

    fmp_date = None
    for e in fmp_events:
        if e['symbol'] == symbol:
            fmp_date_str = e['earnings_date']
            # Parse date string if needed
            if isinstance(fmp_date_str, str):
                fmp_date = datetime.strptime(fmp_date_str, '%Y-%m-%d').date()
            else:
                fmp_date = fmp_date_str
            break

    if fmp_date is None:
        # FMP has no data for this symbol - accept Nasdaq (no verification available)
        return True, None

    if fmp_date == nasdaq_date:
        return True, None

    return False, f"Date mismatch: Nasdaq={nasdaq_date}, FMP={fmp_date}"


def get_tradeable_candidates(
    days_ahead: int = 3,
    trade_logger=None,
    fill_timing: bool = True,
    verify_dates: bool = True,
    screening_date: date = None,
) -> tuple[list[EarningsEvent], list[EarningsEvent]]:
    """Get tradeable earnings candidates with timing fill and date verification.

    Unified function used by both daemon and dashboard.

    Args:
        days_ahead: How many days ahead to fetch earnings
        trade_logger: TradeLogger instance (required for date verification)
        fill_timing: Whether to fill unknown timing from yfinance
        verify_dates: Whether to verify dates against FMP
        screening_date: Date to use as "today" for categorization. If None,
                       uses date.today(). Pass tomorrow's date after market
                       close to preview next screening session's candidates.

    Returns:
        (bmo_tomorrow, amc_today) - lists of verified candidates
    """
    today = screening_date or date.today()
    tomorrow = today + timedelta(days=1)

    # Fetch Nasdaq events
    events = fetch_upcoming_earnings(days_ahead=days_ahead)

    # Pre-fetch all sources for verification (if trade_logger provided)
    if verify_dates and trade_logger:
        fetch_all_earnings_sources(days_ahead=days_ahead, trade_logger=trade_logger)

    bmo_tomorrow = []
    amc_today = []

    for e in events:
        # Fill unknown timing from yfinance
        if fill_timing and e.timing == 'unknown':
            yf_timing = fetch_timing_from_yfinance(e.symbol)
            if yf_timing and yf_timing != 'unknown':
                e.timing = yf_timing

        # Verify date against FMP (if trade_logger provided)
        if verify_dates and trade_logger:
            is_verified, _ = verify_earnings_date(e.symbol, e.earnings_date, trade_logger)
            if not is_verified:
                continue

        # Categorize by timing
        if e.earnings_date == tomorrow and e.timing == 'BMO':
            bmo_tomorrow.append(e)
        elif e.earnings_date == today and e.timing == 'AMC':
            amc_today.append(e)

    return bmo_tomorrow, amc_today


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

    ticker = None
    try:
        ticker = ib.reqMktData(stock, '', False, False)

        # Wait for price (max 2s)
        for _ in range(20):
            if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
                break
            if ticker.close and not math.isnan(ticker.close) and ticker.close > 0:
                break
            await asyncio.sleep(0.1)

        spot = ticker.marketPrice()
        if math.isnan(spot) or spot <= 0:  # nan check
            spot = ticker.last
        if math.isnan(spot) or spot <= 0:
            spot = ticker.close
    finally:
        if ticker is not None:
            try:
                ib.cancelMktData(stock)
            except Exception:
                pass

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

    call_ticker = None
    put_ticker = None
    call_bid = call_ask = put_bid = put_ask = 0.0
    call_iv = put_iv = None
    try:
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
        if call_ticker.modelGreeks:
            call_iv = call_ticker.modelGreeks.impliedVol
        if put_ticker.modelGreeks:
            put_iv = put_ticker.modelGreeks.impliedVol
    finally:
        # Always cancel market data subscriptions to prevent leaks
        if call_ticker is not None:
            try:
                ib.cancelMktData(call)
            except Exception:
                pass
        if put_ticker is not None:
            try:
                ib.cancelMktData(put)
            except Exception:
                pass

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
