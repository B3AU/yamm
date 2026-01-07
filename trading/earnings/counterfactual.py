"""Counterfactual logging for non-traded candidates.

After earnings are announced, fetch the realized stock move and calculate
what the straddle P&L would have been if we had traded.

This prevents survivorship bias by showing whether our rejections were correct.
"""
from __future__ import annotations

import os
import logging
from datetime import date, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

from trading.earnings.logging import TradeLogger, NonTradeLog

# Load environment
load_dotenv()
FMP_API_KEY = os.getenv('FMP_API_KEY', '')

logger = logging.getLogger(__name__)


def fetch_realized_move(
    symbol: str,
    earnings_date: date,
    timing: str,
) -> Optional[dict]:
    """Fetch the realized stock move around earnings.

    Args:
        symbol: Stock ticker
        earnings_date: Date of earnings announcement
        timing: "BMO" (before market open) or "AMC" (after market close)

    Returns:
        dict with:
            - close_before: Close price day before earnings reaction
            - close_after: Close price day of earnings reaction
            - realized_move: Absolute move as decimal (e.g., 0.05 = 5%)
            - direction: "up" or "down"
        Or None if data unavailable.
    """
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not set - cannot fetch realized moves")
        return None

    try:
        # Determine the relevant dates based on timing
        # BMO: Earnings before market open, reaction happens at open same day
        #      Entry was T-1 close, exit is T close
        # AMC: Earnings after market close, reaction happens next day open
        #      Entry was T close, exit is T+1 close

        if timing == 'BMO':
            # Entry: close of earnings_date - 1
            # Exit: close of earnings_date
            entry_date = earnings_date - timedelta(days=1)
            exit_date = earnings_date
        else:  # AMC or unknown
            # Entry: close of earnings_date
            # Exit: close of earnings_date + 1
            entry_date = earnings_date
            exit_date = earnings_date + timedelta(days=1)

        # Fetch prices for the range (with buffer for weekends)
        start = entry_date - timedelta(days=5)
        end = exit_date + timedelta(days=5)

        url = (
            f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
            f"?symbol={symbol}&from={start}&to={end}&apikey={FMP_API_KEY}"
        )

        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            logger.debug(f"{symbol}: FMP API error {r.status_code}")
            return None

        data = r.json()
        if not data:
            logger.debug(f"{symbol}: No price data returned")
            return None

        # Build date -> close price map
        prices = {}
        for bar in data:
            bar_date = bar.get('date', '')[:10]  # YYYY-MM-DD
            close = bar.get('adjClose') or bar.get('close')
            if bar_date and close:
                prices[bar_date] = float(close)

        # Find the actual trading days
        entry_close = _find_closest_price(prices, entry_date, direction='before')
        exit_close = _find_closest_price(prices, exit_date, direction='after')

        if not entry_close or not exit_close:
            logger.debug(f"{symbol}: Could not find entry/exit prices")
            return None

        entry_price, entry_actual_date = entry_close
        exit_price, exit_actual_date = exit_close

        # Calculate move
        move = (exit_price - entry_price) / entry_price
        abs_move = abs(move)
        direction = 'up' if move > 0 else 'down'

        logger.debug(
            f"{symbol}: Entry {entry_actual_date}=${entry_price:.2f}, "
            f"Exit {exit_actual_date}=${exit_price:.2f}, Move={move*100:.1f}%"
        )

        return {
            'close_before': entry_price,
            'close_after': exit_price,
            'realized_move': abs_move,
            'direction': direction,
            'entry_date': entry_actual_date,
            'exit_date': exit_actual_date,
        }

    except Exception as e:
        logger.error(f"{symbol}: Error fetching realized move: {e}")
        return None


def _find_closest_price(
    prices: dict[str, float],
    target_date: date,
    direction: str = 'before',
    max_days: int = 5,
) -> Optional[tuple[float, str]]:
    """Find the closest price to target date.

    Args:
        prices: Dict of date string -> price
        target_date: Target date
        direction: 'before' to look backwards, 'after' to look forwards
        max_days: Maximum days to search

    Returns:
        Tuple of (price, date_string) or None
    """
    for i in range(max_days + 1):
        if direction == 'before':
            check_date = target_date - timedelta(days=i)
        else:
            check_date = target_date + timedelta(days=i)

        date_str = check_date.strftime('%Y-%m-%d')
        if date_str in prices:
            return prices[date_str], date_str

    return None


def calculate_counterfactual_pnl(
    realized_move: float,
    implied_move: float,
    quoted_bid: float,
    quoted_ask: float,
    spot_price: float,
) -> dict:
    """Calculate counterfactual P&L for a straddle.

    Args:
        realized_move: Absolute realized move as decimal
        implied_move: Implied move from straddle price as decimal
        quoted_bid: Straddle bid price
        quoted_ask: Straddle ask price
        spot_price: Spot price at entry

    Returns:
        dict with:
            - pnl_at_mid: P&L assuming mid fill
            - pnl_with_spread: P&L assuming realistic fills (buy at ask, sell at bid)
            - profitable: bool
    """
    # Straddle intrinsic value at expiration = |move| * spot
    # (simplified - assumes ATM straddle where one leg is ITM by the move amount)
    intrinsic_value = realized_move * spot_price

    # Mid price entry/exit
    mid = (quoted_bid + quoted_ask) / 2

    # Entry at mid, exit at intrinsic value (no spread on exit for simplicity)
    pnl_at_mid = intrinsic_value - mid

    # Entry at ask, exit at intrinsic value minus half the percentage spread
    # (conservative estimate of exit slippage)
    spread = quoted_ask - quoted_bid
    spread_pct = spread / mid if mid > 0 else 0

    # Entry cost = ask, exit value = intrinsic minus slippage
    entry_cost = quoted_ask
    exit_value = intrinsic_value * (1 - spread_pct * 0.5)
    pnl_with_spread = exit_value - entry_cost

    return {
        'pnl_at_mid': pnl_at_mid,
        'pnl_with_spread': pnl_with_spread,
        'profitable': pnl_with_spread > 0,
        'intrinsic_value': intrinsic_value,
    }


def backfill_counterfactuals(
    trade_logger: TradeLogger,
    earnings_date: date,
) -> dict:
    """Backfill counterfactual data for all non-trades on a given earnings date.

    Should be called after market close on the day after earnings.

    Returns:
        dict with counts of updated/skipped/failed records
    """
    earnings_date_str = str(earnings_date)
    pending = trade_logger.get_non_trades_pending_counterfactual(earnings_date_str)

    if not pending:
        logger.info(f"No non-trades pending counterfactual backfill for {earnings_date_str}")
        return {'updated': 0, 'skipped': 0, 'failed': 0}

    logger.info(f"Backfilling counterfactuals for {len(pending)} non-trades on {earnings_date_str}")

    updated = 0
    skipped = 0
    failed = 0

    for non_trade in pending:
        # Fetch realized move
        move_data = fetch_realized_move(
            non_trade.ticker,
            earnings_date,
            non_trade.earnings_timing,
        )

        if not move_data:
            failed += 1
            logger.warning(f"{non_trade.ticker}: Failed to fetch realized move")
            continue

        realized_move = move_data['realized_move']

        # Calculate counterfactual P&L if we have enough data
        if (non_trade.quoted_bid and non_trade.quoted_ask and
            non_trade.spot_price and non_trade.implied_move):

            pnl_data = calculate_counterfactual_pnl(
                realized_move=realized_move,
                implied_move=non_trade.implied_move,
                quoted_bid=non_trade.quoted_bid,
                quoted_ask=non_trade.quoted_ask,
                spot_price=non_trade.spot_price,
            )

            trade_logger.update_non_trade(
                non_trade.log_id,
                counterfactual_realized_move=realized_move,
                counterfactual_pnl=pnl_data['pnl_at_mid'],
                counterfactual_pnl_with_spread=pnl_data['pnl_with_spread'],
            )

            status = "WIN" if pnl_data['profitable'] else "LOSS"
            logger.info(
                f"{non_trade.ticker}: Realized {realized_move*100:.1f}% vs "
                f"implied {non_trade.implied_move*100:.1f}%, "
                f"counterfactual P&L: ${pnl_data['pnl_with_spread']:.2f} ({status})"
            )
        else:
            # Just log the realized move without P&L calculation
            trade_logger.update_non_trade(
                non_trade.log_id,
                counterfactual_realized_move=realized_move,
            )
            logger.info(
                f"{non_trade.ticker}: Realized {realized_move*100:.1f}% (incomplete quote data)"
            )

        updated += 1

    logger.info(f"Counterfactual backfill complete: {updated} updated, {skipped} skipped, {failed} failed")

    return {'updated': updated, 'skipped': skipped, 'failed': failed}


def get_recent_counterfactual_summary(trade_logger: TradeLogger, days: int = 7) -> dict:
    """Get summary of counterfactual outcomes for recent non-trades.

    Useful for analyzing whether rejection criteria are too tight/loose.
    """
    from datetime import datetime

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    non_trades = trade_logger.get_non_trades(
        from_date=str(start_date),
        to_date=str(end_date),
    )

    # Filter to those with counterfactual data
    with_cf = [nt for nt in non_trades if nt.counterfactual_realized_move is not None]

    if not with_cf:
        return {
            'total_non_trades': len(non_trades),
            'with_counterfactual': 0,
            'would_have_profited': 0,
            'would_have_lost': 0,
            'avg_missed_pnl': 0,
            'by_rejection_reason': {},
        }

    profitable = [nt for nt in with_cf if (nt.counterfactual_pnl_with_spread or 0) > 0]
    losing = [nt for nt in with_cf if (nt.counterfactual_pnl_with_spread or 0) <= 0]

    # Average missed P&L (what we would have made/lost)
    pnls = [nt.counterfactual_pnl_with_spread for nt in with_cf if nt.counterfactual_pnl_with_spread]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    # Group by rejection reason
    by_reason = {}
    for nt in with_cf:
        reason = nt.rejection_reason or 'unknown'
        if reason not in by_reason:
            by_reason[reason] = {'count': 0, 'profitable': 0, 'total_pnl': 0}
        by_reason[reason]['count'] += 1
        if (nt.counterfactual_pnl_with_spread or 0) > 0:
            by_reason[reason]['profitable'] += 1
        by_reason[reason]['total_pnl'] += nt.counterfactual_pnl_with_spread or 0

    return {
        'total_non_trades': len(non_trades),
        'with_counterfactual': len(with_cf),
        'would_have_profited': len(profitable),
        'would_have_lost': len(losing),
        'avg_missed_pnl': avg_pnl,
        'by_rejection_reason': by_reason,
    }
