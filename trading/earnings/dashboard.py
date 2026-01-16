#!/usr/bin/env python3
"""CLI Dashboard for earnings trading.

Shows open trades, pending orders, and P&L with live IBKR data.

Usage:
    python -m trading.earnings.dashboard
    python -m trading.earnings.dashboard --watch  # Auto-refresh every 5s
    python -m trading.earnings.dashboard --live   # Connect to IBKR for live prices

Interactive commands (in watch mode):
    c - Close a position
    r - Refresh now
    q - Quit
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys
import time
import asyncio
import json
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

from trading.earnings.logging import TradeLogger
from trading.earnings.screener import get_tradeable_candidates
from trading.earnings.utils import should_exit_today
from trading.earnings.config import (
    EXIT_TIME_ET, SCREEN_TIME_ET,
    get_exit_time, get_screen_time,
)

DB_PATH = PROJECT_ROOT / 'data' / 'earnings_trades.db'
LOG_PATH = PROJECT_ROOT / 'logs' / 'daemon.log'

# Global IB connection for live mode
_ib = None


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def format_currency(value: float) -> str:
    if value is None:
        return "N/A"
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def format_pct(value: float) -> str:
    if value is None:
        return "N/A"
    return f"{value*100:.1f}%"


def get_status_color(status: str) -> str:
    colors = {
        'pending': '\033[93m',      # Yellow
        'submitted': '\033[93m',    # Yellow
        'filled': '\033[92m',       # Green
        'partial': '\033[91m',      # Red - partial fills are warnings
        'exiting': '\033[96m',      # Cyan - exit orders placed
        'partial_exit': '\033[93m', # Yellow - one leg exited
        'overdue': '\033[91m',      # Red - past exit time, no exit order
        'exited': '\033[94m',       # Blue - completed
        'completed': '\033[94m',    # Blue
        'cancelled': '\033[90m',    # Gray
        'error': '\033[91m',        # Red
    }
    return colors.get(status.lower(), '\033[0m')


def reset_color() -> str:
    return '\033[0m'


def bold(text: str) -> str:
    return f'\033[1m{text}\033[0m'


def dim(text: str) -> str:
    return f'\033[2m{text}\033[0m'


# Timezone for market hours
ET = pytz.timezone('US/Eastern')
PT = pytz.timezone('Europe/Lisbon')  # Portugal time


def get_market_status() -> tuple[str, str]:
    """Get current market status and color.

    Returns (status_text, color_code)
    """
    now = datetime.now(ET)
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    time_val = hour * 60 + minute

    # Weekend
    if weekday >= 5:
        return "CLOSED (Weekend)", '\033[90m'

    # Pre-market: 4:00 - 9:30 ET
    if 240 <= time_val < 570:
        return "PRE-MARKET", '\033[93m'

    # Regular hours: 9:30 - 16:00 ET
    if 570 <= time_val < 960:
        return "OPEN", '\033[92m'

    # After hours: 16:00 - 20:00 ET
    if 960 <= time_val < 1200:
        return "AFTER-HOURS", '\033[93m'

    return "CLOSED", '\033[90m'


def format_time_until(target_dt: datetime, now: datetime = None) -> str:
    """Format time until a target datetime."""
    if now is None:
        now = datetime.now(timezone.utc)

    # Normalize both to UTC for comparison
    if target_dt.tzinfo is None:
        # Assume naive datetime is in local time, convert via UTC
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    else:
        target_dt = target_dt.astimezone(timezone.utc)

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    diff = target_dt - now
    total_seconds = diff.total_seconds()

    if total_seconds < 0:
        return "passed"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    if hours > 24:
        days = hours // 24
        return f"in {days}d"
    elif hours > 0:
        return f"in {hours}h {minutes}m"
    else:
        return f"in {minutes}m"


def format_time_since(past_dt: datetime, now: datetime = None) -> str:
    """Format time since a past datetime."""
    if now is None:
        now = datetime.now(timezone.utc)

    if isinstance(past_dt, str):
        past_dt = datetime.fromisoformat(past_dt)

    # Convert both to UTC for proper comparison
    if past_dt.tzinfo:
        past_dt = past_dt.astimezone(timezone.utc)
    else:
        past_dt = past_dt.replace(tzinfo=timezone.utc)

    if now.tzinfo:
        now = now.astimezone(timezone.utc)
    else:
        now = now.replace(tzinfo=timezone.utc)

    diff = now - past_dt
    total_seconds = diff.total_seconds()

    if total_seconds < 0:
        return "future"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    if hours > 24:
        days = hours // 24
        return f"{days}d ago"
    elif hours > 0:
        return f"{hours}h {minutes}m ago"
    else:
        return f"{minutes}m ago"


def format_age_compact(past_dt: datetime, now: datetime = None) -> str:
    """Format age as compact string (e.g., '12h', '1.5d')."""
    if past_dt is None:
        return "?"

    if now is None:
        now = datetime.now(timezone.utc)

    if isinstance(past_dt, str):
        past_dt = datetime.fromisoformat(past_dt)

    # Convert both to UTC for proper comparison
    if past_dt.tzinfo:
        past_dt = past_dt.astimezone(timezone.utc)
    else:
        past_dt = past_dt.replace(tzinfo=timezone.utc)

    if now.tzinfo:
        now = now.astimezone(timezone.utc)
    else:
        now = now.replace(tzinfo=timezone.utc)

    diff = now - past_dt
    total_hours = diff.total_seconds() / 3600

    if total_hours < 1:
        return f"{int(diff.total_seconds() // 60)}m"
    elif total_hours < 24:
        return f"{total_hours:.0f}h"
    else:
        days = total_hours / 24
        if days < 10:
            return f"{days:.1f}d"
        return f"{int(days)}d"


def get_next_screen_time() -> Optional[datetime]:
    """Get next screening time based on config."""
    screen_hour, screen_min = get_screen_time()
    now_et = datetime.now(ET)
    screen_time = now_et.replace(hour=screen_hour, minute=screen_min, second=0, microsecond=0)

    # If past screen time today, next screen is tomorrow
    if now_et.hour > screen_hour or (now_et.hour == screen_hour and now_et.minute >= screen_min):
        screen_time += timedelta(days=1)

    # Skip weekends
    while screen_time.weekday() >= 5:
        screen_time += timedelta(days=1)

    # Return as naive datetime in local time for comparison
    return screen_time.astimezone().replace(tzinfo=None)


def get_next_exit_time() -> Optional[datetime]:
    """Get next exit time based on config."""
    exit_hour, exit_min = get_exit_time()
    now_et = datetime.now(ET)
    exit_time = now_et.replace(hour=exit_hour, minute=exit_min, second=0, microsecond=0)

    # If past exit time today, next exit is tomorrow
    if now_et.hour > exit_hour or (now_et.hour == exit_hour and now_et.minute >= exit_min):
        exit_time += timedelta(days=1)

    # Skip weekends
    while exit_time.weekday() >= 5:
        exit_time += timedelta(days=1)

    return exit_time.astimezone().replace(tzinfo=None)


def make_sparkline(values: list[float], width: int = 20) -> str:
    """Create a text-based sparkline from values."""
    if not values:
        return ""

    blocks = " ▁▂▃▄▅▆▇█"

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        return blocks[4] * min(len(values), width)

    # Sample values if too many
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    result = ""
    for v in sampled:
        normalized = (v - min_val) / val_range
        idx = int(normalized * (len(blocks) - 1))
        result += blocks[idx]

    return result


def get_recent_warnings_errors(log_path: Path, max_lines: int = 1000, max_display: int = 8) -> list[tuple[str, str]]:
    """Read recent WARNING and ERROR lines from log file.

    Returns list of (level, message) tuples, most recent first.
    """
    if not log_path.exists():
        return []

    try:
        with open(log_path, 'r') as f:
            # Read last N lines efficiently
            lines = f.readlines()[-max_lines:]

        warnings_errors = []
        for line in reversed(lines):
            line = line.strip()
            if ' - WARNING - ' in line or ' - ERROR - ' in line:
                # Parse: "2025-01-07 10:30:45,123 - module - WARNING - message"
                parts = line.split(' - ', 3)
                if len(parts) >= 4:
                    level = parts[2]
                    message = parts[3][:60]  # Truncate long messages
                    timestamp = parts[0].split(',')[0][-8:]  # Just HH:MM:SS
                    warnings_errors.append((level, f"{timestamp} {message}"))
                    if len(warnings_errors) >= max_display:
                        break

        return warnings_errors
    except Exception:
        return []


def connect_ib(client_id: int = 20) -> Optional[object]:
    """Connect to IB Gateway (Synchronous)."""
    global _ib
    if _ib is not None and _ib.isConnected():
        return _ib

    try:
        from ib_insync import IB
        _ib = IB()
        # Note: If called within an async loop, this might be problematic if using run().
        # But we use simple connect() which blocks.
        _ib.connect('127.0.0.1', 4002, clientId=client_id, timeout=10)
        _ib.reqMarketDataType(1)  # Live data
        return _ib
    except Exception as e:
        print(f"  [Could not connect to IBKR: {e}]")
        return None


async def connect_ib_async(client_id: int = 20) -> Optional[object]:
    """Connect to IB Gateway for live prices (Async)."""
    global _ib
    if _ib is not None and _ib.isConnected():
        return _ib

    try:
        from ib_insync import IB
        _ib = IB()
        await _ib.connectAsync('127.0.0.1', 4002, clientId=client_id, timeout=10)
        _ib.reqMarketDataType(1)  # Live data
        return _ib
    except Exception as e:
        print(f"  [Could not connect to IBKR: {e}]")
        return None


def disconnect_ib():
    """Disconnect from IB Gateway."""
    global _ib
    if _ib is not None and _ib.isConnected():
        _ib.disconnect()
        _ib = None


async def get_live_option_price_async(ib, symbol: str, expiry: str, strike: float, right: str) -> Optional[dict]:
    """Get live bid/ask for an option (Async)."""
    if ib is None:
        return None

    try:
        from ib_insync import Option
        import asyncio

        opt = Option(symbol, expiry, strike, right, 'SMART', tradingClass=symbol)

        # Async qualification
        try:
            qualified = await ib.qualifyContractsAsync(opt)
            if not qualified:
                 # Try without tradingClass
                opt = Option(symbol, expiry, strike, right, 'SMART')
                qualified = await ib.qualifyContractsAsync(opt)
                if not qualified:
                    return {'error': 'contract not found'}
        except Exception:
             # Try without tradingClass
            opt = Option(symbol, expiry, strike, right, 'SMART')
            try:
                qualified = await ib.qualifyContractsAsync(opt)
                if not qualified:
                    return {'error': 'contract not found'}
            except Exception:
                 return {'error': 'contract qualification failed'}

        if not opt.conId:
            return {'error': 'contract not found'}

        ticker = ib.reqMktData(opt, '', False, False)
        try:
            # Wait longer for data, check incrementally
            for _ in range(20): # 2 seconds max
                if ticker.bid and ticker.bid > 0:
                    break
                await asyncio.sleep(0.1)

            # IBKR returns -1.0 when no quote is available (market closed, no liquidity)
            # Also check for NaN
            def valid_price(p):
                return p is not None and not math.isnan(p) and p > 0

            result = {
                'bid': ticker.bid if valid_price(ticker.bid) else None,
                'ask': ticker.ask if valid_price(ticker.ask) else None,
                'last': ticker.last if valid_price(ticker.last) else None,
            }
            return result
        finally:
            ib.cancelMktData(opt)
    except Exception as e:
        return {'error': str(e)}


async def get_position_live_value_async(ib, trade) -> Optional[dict]:
    """Get live value for a position (Async)."""
    if ib is None or not trade.strikes or not trade.expiration:
        return {'error': 'missing trade data'}

    try:
        import json
        try:
            strikes = json.loads(trade.strikes) if isinstance(trade.strikes, str) else trade.strikes
        except (json.JSONDecodeError, TypeError):
            strikes = []
        strike = strikes[0] if isinstance(strikes, list) and strikes else float(trade.strikes or 0)
        expiry = trade.expiration.replace('-', '')

        # Fetch in parallel
        import asyncio
        call_task = asyncio.create_task(get_live_option_price_async(ib, trade.ticker, expiry, strike, 'C'))
        put_task = asyncio.create_task(get_live_option_price_async(ib, trade.ticker, expiry, strike, 'P'))

        call_data, put_data = await asyncio.gather(call_task, put_task)

        # Check for errors from option price fetch
        if call_data and 'error' in call_data:
            return {'error': f"call: {call_data['error']}"}
        if put_data and 'error' in put_data:
            return {'error': f"put: {put_data['error']}"}

        if call_data and put_data:
            call_mid = None
            put_mid = None
            call_used_last = False
            put_used_last = False

            if call_data.get('bid') and call_data.get('ask'):
                call_mid = (call_data['bid'] + call_data['ask']) / 2
            elif call_data.get('last'):
                call_mid = call_data['last']
                call_used_last = True

            if put_data.get('bid') and put_data.get('ask'):
                put_mid = (put_data['bid'] + put_data['ask']) / 2
            elif put_data.get('last'):
                put_mid = put_data['last']
                put_used_last = True

            if call_mid and put_mid:
                current_value = (call_mid + put_mid) * trade.contracts * 100
                entry_value = trade.entry_quoted_mid * trade.contracts * 100 if trade.entry_quoted_mid else None

                return {
                    'call_mid': call_mid,
                    'put_mid': put_mid,
                    'straddle_mid': call_mid + put_mid,
                    'current_value': current_value,
                    'entry_value': entry_value,
                    'unrealized_pnl': current_value - entry_value if entry_value else None,
                    'pnl_pct': ((call_mid + put_mid) / trade.entry_quoted_mid - 1) * 100 if trade.entry_quoted_mid else None,
                    'used_last': call_used_last or put_used_last,
                }
            else:
                # No valid prices
                missing = []
                if not call_mid:
                    missing.append('call')
                if not put_mid:
                    missing.append('put')
                return {'error': f"no quote for {'+'.join(missing)}"}

        return {'error': 'no data returned'}
    except Exception as e:
        return {'error': str(e)}


def render_dashboard(
    logger: TradeLogger,
    show_all: bool = False,
    live: bool = False,
    compact: bool = False,
    live_data_map: Dict[str, Any] = None
):
    """Render the dashboard.

    Args:
        logger: TradeLogger instance
        show_all: If True, show all trades including completed
        live: If True, indicates live mode enabled (passed for info)
        compact: If True, show condensed 1-line per position format
        live_data_map: Optional dict mapping trade_id to live data dict
    """
    now = datetime.now()
    live_data_map = live_data_map or {}

    # Market status
    market_status, market_color = get_market_status()

    # Time to market open/close (PT time display)
    now_et = datetime.now(ET)
    now_pt = datetime.now(PT)

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    time_info = ""
    if market_status == "PRE-MARKET":
        time_info = f" | Opens {format_time_until(market_open, now_et)}"
    elif market_status == "OPEN":
        time_info = f" | Closes {format_time_until(market_close, now_et)}"
    elif market_status == "CLOSED":
        # Next open logic
        next_open = market_open
        if now_et >= market_close:
            next_open += timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        time_info = f" | Opens {format_time_until(next_open, now_et)}"

    # Get PT timezone abbreviation (WET/WEST)
    pt_tz_name = now_pt.strftime('%Z')

    print(bold("=" * 90))
    print(bold(f"  EARNINGS TRADING DASHBOARD - {now_pt.strftime('%Y-%m-%d %H:%M:%S')} ({pt_tz_name})"))
    print(f"  Market: {market_color}{market_status}{reset_color()}{time_info}")
    print(bold("=" * 90))
    print()

    # Get all trades
    all_trades = logger.get_trades()

    if not all_trades:
        print("  No trades recorded yet.")
        print()
        return

    # Build LLM check map for all trades
    llm_check_map = {}
    for trade in all_trades:
        if trade.entry_datetime:
            llm_check = logger.get_llm_check_for_trade(trade.ticker, trade.entry_datetime)
            if llm_check:
                llm_check_map[trade.trade_id] = llm_check

    # Categorize trades into 3 buckets:
    # 1. OPEN: Entry filled, not yet time to exit
    # 2. EXITING: Currently exiting (includes partial_exit, exiting, and stuck positions)
    # 3. COMPLETED: Fully closed

    def _is_exiting_trade(t):
        """Check if trade belongs in EXITING section."""
        if t.status in ('exiting', 'partial_exit'):
            return True
        # Only include 'filled' positions AFTER 14:00 ET on exit day
        if t.status == 'filled':
            now_et = datetime.now(ET)
            if now_et.hour < 14:
                return False  # Before exit time, stay in OPEN
            try:
                earnings_date = datetime.strptime(t.earnings_date, '%Y-%m-%d').date()
                return should_exit_today(earnings_date, t.earnings_timing)
            except (ValueError, TypeError):
                pass
        return False

    open_trades = [t for t in all_trades
                   if t.status in ('pending', 'submitted', 'filled', 'partial')
                   and not _is_exiting_trade(t)]
    exiting_trades = [t for t in all_trades if _is_exiting_trade(t)]
    completed_trades = [t for t in all_trades if t.status in ('completed', 'exited')]

    # === Open Positions ===
    print(bold("  OPEN POSITIONS"))

    total_unrealized_pnl = 0
    has_live_data = False

    if open_trades:
        if compact:
            # Compact format: 1 line per position
            print(f"  {'Sym':<5} {'Status':<7} {'Age':<5} {'Strike':<7} {'Exp':<5} {'Entry':<8} {'Curr':<8} {'P&L':<8} {'Edge':<5} {'Impl':<5} {'Sprd':<5} {'LLM':<4}")
            print("  " + "-" * 94)
        else:
            print(f"  {'Symbol':<8} {'Earnings':<12} {'Age':<6} {'Status':<10} {'Entry':<10} {'Current':<10} {'P&L':<10} {'LLM':<4}")
            print("  " + "-" * 88)

        for trade in open_trades:
            status_color = get_status_color(trade.status)

            # Parse entry info
            entry_price = "N/A"
            entry_price_short = "N/A"
            if trade.entry_fill_price:
                entry_price = format_currency(trade.entry_fill_price)
                entry_price_short = f"${trade.entry_fill_price:.2f}"
            elif trade.entry_quoted_mid:
                entry_price = f"${trade.entry_quoted_mid:.2f}"
                entry_price_short = f"${trade.entry_quoted_mid:.2f}"

            # Get live prices if available
            current_price = "N/A"
            current_price_short = "N/A"
            pnl_str = "N/A"
            pnl_color = reset_color()
            live_error = None

            if live_data_map and trade.trade_id in live_data_map:
                live_data = live_data_map[trade.trade_id]
                if live_data:
                    if 'error' in live_data:
                        live_error = live_data['error']
                    else:
                        has_live_data = True
                        current_price = f"${live_data['straddle_mid']:.2f}"
                        # Yellow if using 'last' price fallback (no live bid/ask)
                        if live_data.get('used_last'):
                            current_price_short = f"\033[93m${live_data['straddle_mid']:.2f}\033[0m"
                        else:
                            current_price_short = f"${live_data['straddle_mid']:.2f}"

                        if live_data['pnl_pct'] is not None:
                            pnl_pct = live_data['pnl_pct']
                            pnl_color = '\033[92m' if pnl_pct >= 0 else '\033[91m'
                            pnl_str = f"{pnl_pct:+.1f}%"

                            if live_data['unrealized_pnl'] is not None:
                                total_unrealized_pnl += live_data['unrealized_pnl']

            # Add warning indicator for partial fills and exiting status
            status_display = trade.status
            if trade.status == 'partial':
                status_display = "PARTIAL!"
            elif trade.status == 'exiting':
                status_display = "EXIT"
            # 'filled' just shows as 'filled' - expiration date indicates when it exits
            # Positions move to EXITING section after 14:00 ET on exit day

            if compact:
                # Compact: single line with key info
                import json
                try:
                    strikes = json.loads(trade.strikes) if trade.strikes else []
                except (json.JSONDecodeError, TypeError):
                    strikes = []
                strike_str = f"{strikes[0]:.0f}" if strikes else "?"
                expiry_short = trade.expiration[5:] if trade.expiration else "?"  # MM-DD
                age_str = format_age_compact(trade.entry_datetime, now)
                edge_str = f"{trade.edge_q75*100:.0f}%" if trade.edge_q75 else "."
                impl_str = f"{trade.implied_move*100:.0f}%" if trade.implied_move else "."

                # Entry spread %
                if trade.entry_quoted_bid and trade.entry_quoted_ask and trade.entry_quoted_mid:
                    spread = trade.entry_quoted_ask - trade.entry_quoted_bid
                    spread_pct = (spread / trade.entry_quoted_mid) * 100
                    sprd_str = f"{spread_pct:.0f}%"
                else:
                    sprd_str = "."

                # LLM check result with color
                llm_check = llm_check_map.get(trade.trade_id)
                if llm_check:
                    llm_decision = llm_check['decision']
                    llm_colors = {'PASS': '\033[92m', 'WARN': '\033[93m', 'NO_TRADE': '\033[91m'}
                    llm_color = llm_colors.get(llm_decision, '')
                    llm_str = f"{llm_color}{llm_decision[:4]}{reset_color()}"
                else:
                    llm_str = "."

                # Truncate symbol if too long
                sym = trade.ticker[:5]

                print(f"  {sym:<5} {status_color}{status_display:<7}{reset_color()} "
                      f"{age_str:<5} {strike_str:<7} {expiry_short:<5} "
                      f"{entry_price_short:<8} {current_price_short:<8} {pnl_color}{pnl_str:<8}{reset_color()} "
                      f"{edge_str:<5} {impl_str:<5} {sprd_str:<5} {llm_str:<4}")


                # Only show errors/warnings on second line if critical
                if trade.status == 'partial' and trade.notes:
                    print(f"         \033[91m^ {trade.notes}\033[0m")
            else:
                # Full format: multiple lines per position
                # LLM check result with color
                llm_check = llm_check_map.get(trade.trade_id)
                if llm_check:
                    llm_decision = llm_check['decision']
                    llm_colors = {'PASS': '\033[92m', 'WARN': '\033[93m', 'NO_TRADE': '\033[91m'}
                    llm_color = llm_colors.get(llm_decision, '')
                    llm_str = f"{llm_color}{llm_decision[:4]}{reset_color()}"
                else:
                    llm_str = "."

                age_str = format_age_compact(trade.entry_datetime, now)
                print(f"  {trade.ticker:<8} {trade.earnings_date:<12} "
                      f"{age_str:<6} {status_color}{status_display:<10}{reset_color()} "
                      f"{entry_price:<10} {current_price:<10} {pnl_color}{pnl_str:<10}{reset_color()} {llm_str:<4}")

                # Show partial fill warning with details
                if trade.status == 'partial' and trade.notes:
                    print(f"           \033[91mWARNING: {trade.notes}\033[0m")

                # Show live data error if any
                if live_error:
                    print(f"           \033[93m[No quote: {live_error}]\033[0m")

                # Show exit order info for exiting positions
                if trade.status == 'exiting' and trade.exit_limit_price:
                    exit_info = f"Exit limit: ${trade.exit_limit_price:.2f}"
                    if trade.exit_quoted_mid:
                        exit_info += f" (mid: ${trade.exit_quoted_mid:.2f})"
                    print(f"           \033[96m{exit_info}\033[0m")

                # Show order details on second line
                if trade.structure:
                    timing = trade.earnings_timing or "?"
                    strikes = trade.strikes or "?"
                    expiry = trade.expiration or "?"

                    # Time context
                    time_info = ""
                    if trade.entry_datetime:
                        time_info = f" | Placed: {format_time_since(trade.entry_datetime, now)}"

                    # Time until earnings
                    if trade.earnings_date and trade.earnings_timing:
                        try:
                            earn_date = datetime.strptime(trade.earnings_date, "%Y-%m-%d")
                            if trade.earnings_timing == 'BMO':
                                # BMO: earnings at ~8 AM ET
                                earn_dt = earn_date.replace(hour=8, minute=0)
                            else:
                                # AMC: earnings at ~4:30 PM ET
                                earn_dt = earn_date.replace(hour=16, minute=30)
                            time_info += f" | Earnings: {format_time_until(earn_dt, now)}"
                        except:
                            pass

                    print(f"           {trade.structure} | Strike: {strikes} | Exp: {expiry} | {timing}{time_info}")

                # Show edge, implied move, and spread
                spread_str = ""
                if trade.entry_quoted_bid and trade.entry_quoted_ask and trade.entry_quoted_mid:
                    spread = trade.entry_quoted_ask - trade.entry_quoted_bid
                    spread_pct = (spread / trade.entry_quoted_mid) * 100
                    spread_str = f" | Spread: {spread_pct:.1f}%"

                if trade.edge_q75 is not None:
                    print(f"           Edge: {format_pct(trade.edge_q75)} | "
                          f"Pred Q75: {format_pct(trade.predicted_q75)} | "
                          f"Implied: {format_pct(trade.implied_move)}{spread_str}")
                elif trade.implied_move is not None:
                    print(f"           Implied Move: {format_pct(trade.implied_move)}{spread_str}")

                # Show intraday sparkline if snapshots available
                try:
                    snapshots = logger.get_snapshots(trade.trade_id)
                    if snapshots and len(snapshots) >= 2:
                        values = [s.straddle_mid for s in snapshots if s.straddle_mid]
                        if len(values) >= 2:
                            sparkline = make_sparkline(values, width=20)
                            start_val = values[0]
                            end_val = values[-1]
                            change_pct = (end_val / start_val - 1) * 100 if start_val else 0
                            change_color = '\033[92m' if change_pct >= 0 else '\033[91m'
                            print(f"           Intraday: {sparkline} {change_color}{change_pct:+.1f}%{reset_color()} ({len(values)} pts)")
                except Exception:
                    pass  # Don't fail if snapshots unavailable
                print()

        # Show total unrealized P&L and risk summary
        print("  " + "-" * 88)

        # Calculate total capital at risk
        total_at_risk = sum(t.premium_paid or 0 for t in open_trades)
        total_max_loss = sum(t.max_loss or 0 for t in open_trades)

        if has_live_data and total_unrealized_pnl != 0:
            pnl_color = '\033[92m' if total_unrealized_pnl >= 0 else '\033[91m'
            print(f"  {'UNREALIZED P&L:':<20} {pnl_color}{format_currency(total_unrealized_pnl)}{reset_color():<20}"
                  f"  CAPITAL AT RISK: {format_currency(total_at_risk)}")
        else:
            print(f"  CAPITAL AT RISK: {format_currency(total_at_risk):<15} MAX LOSS: {format_currency(total_max_loss)}")

        print()
    else:
        print("  No open positions")
        print()

    # === Exiting Positions ===
    if exiting_trades:
        print(bold("  EXITING POSITIONS"))

        if compact:
            print(f"  {'Sym':<6}{'Status':<8}{'C Exit':<10}{'P Exit':<10}{'Realized':<10}{'Unreal':<12}{'Est P&L':<10}{'Exp'}")
            print("  " + "-" * 74)
        else:
            print(f"  {'Symbol':<8} {'Status':<12} {'Call Exit':<12} {'Put Exit':<12} {'Realized':<12} {'Unrealized':<10}")
            print("  " + "-" * 80)

        for trade in exiting_trades:
            # Determine status display and color
            if trade.status == 'partial_exit':
                status_display = "partial"
                status_color = '\033[93m'  # Yellow
            elif trade.status == 'exiting':
                status_display = "pending"
                status_color = '\033[96m'  # Cyan
            else:  # 'filled' after 14:00 ET without exit order = overdue
                status_display = "overdue"
                status_color = '\033[91m'  # Red

            # Call exit display
            if trade.call_exit_fill_price is not None:
                call_str = f"${trade.call_exit_fill_price:.2f} \033[92m✓\033[0m"
                call_str_len = len(f"${trade.call_exit_fill_price:.2f} ✓")  # Length without ANSI
            elif trade.exit_call_order_id:
                call_str = "pending"
                call_str_len = 7
            else:
                call_str = "-"
                call_str_len = 1

            # Put exit display
            if trade.put_exit_fill_price is not None:
                put_str = f"${trade.put_exit_fill_price:.2f} \033[92m✓\033[0m"
                put_str_len = len(f"${trade.put_exit_fill_price:.2f} ✓")
            elif trade.exit_put_order_id:
                put_str = "pending"
                put_str_len = 7
            else:
                put_str = "-"
                put_str_len = 1

            # Get per-leg entry prices (use 50/50 split as fallback)
            call_entry = trade.call_entry_fill_price
            put_entry = trade.put_entry_fill_price
            if call_entry is None or put_entry is None:
                # Fallback: 50/50 split of total entry price
                if trade.entry_fill_price:
                    call_entry = trade.entry_fill_price / 2
                    put_entry = trade.entry_fill_price / 2
                else:
                    call_entry = None
                    put_entry = None

            contracts = trade.contracts or 1

            # Realized P&L (from filled exit legs)
            realized_pnl = 0
            has_realized = False

            if trade.call_exit_fill_price is not None and call_entry is not None:
                call_exit_value = trade.call_exit_fill_price * contracts * 100
                call_entry_value = call_entry * contracts * 100
                realized_pnl += call_exit_value - call_entry_value
                has_realized = True

            if trade.put_exit_fill_price is not None and put_entry is not None:
                put_exit_value = trade.put_exit_fill_price * contracts * 100
                put_entry_value = put_entry * contracts * 100
                realized_pnl += put_exit_value - put_entry_value
                has_realized = True

            realized_str = format_currency(realized_pnl) if has_realized else "-"
            realized_color = '\033[92m' if realized_pnl >= 0 else '\033[91m'

            # Unrealized estimate (unfilled legs)
            unrealized_pnl = 0
            unrealized_str = "?"
            has_unrealized = False

            # Get live data for estimates
            live_data = None
            if live_data_map and trade.trade_id in live_data_map:
                live_data = live_data_map.get(trade.trade_id)
                if live_data and 'error' in live_data:
                    live_data = None

            # Check call leg (if not exited)
            if trade.call_exit_fill_price is None and call_entry is not None:
                call_entry_value = call_entry * contracts * 100
                if live_data and live_data.get('call_mid'):
                    call_current_value = live_data['call_mid'] * contracts * 100
                    unrealized_pnl += call_current_value - call_entry_value
                    has_unrealized = True
                elif trade.put_exit_fill_price is not None:
                    # Put filled, call has no quote - assume worthless (deep OTM)
                    unrealized_pnl += 0 - call_entry_value
                    has_unrealized = True

            # Check put leg (if not exited)
            if trade.put_exit_fill_price is None and put_entry is not None:
                put_entry_value = put_entry * contracts * 100
                if live_data and live_data.get('put_mid'):
                    put_current_value = live_data['put_mid'] * contracts * 100
                    unrealized_pnl += put_current_value - put_entry_value
                    has_unrealized = True
                elif trade.call_exit_fill_price is not None:
                    # Call filled, put has no quote - assume worthless (deep OTM)
                    unrealized_pnl += 0 - put_entry_value
                    has_unrealized = True

            if has_unrealized:
                unrealized_str = f"~{format_currency(unrealized_pnl)}"

            # Estimated total P&L
            if has_realized or has_unrealized:
                est_total = realized_pnl + unrealized_pnl
                est_pnl_str = format_currency(est_total)
                est_pnl_color = '\033[92m' if est_total >= 0 else '\033[91m'
            else:
                est_pnl_str = "?"
                est_pnl_color = ''

            # Expiration urgency
            exp_str = "?"
            if trade.expiration:
                try:
                    exp_date = datetime.strptime(trade.expiration, '%Y%m%d').date()
                    today = date.today()
                    days_to_exp = (exp_date - today).days
                    if days_to_exp <= 0:
                        exp_str = "\033[91mTODAY\033[0m"
                    elif days_to_exp == 1:
                        exp_str = "\033[93mtmrw\033[0m"
                    else:
                        exp_str = f"{days_to_exp}d"
                except (ValueError, TypeError):
                    pass

            if compact:
                sym = trade.ticker[:5]
                # Pad strings manually to account for ANSI codes in checkmarks
                call_padded = call_str + ' ' * max(0, 10 - call_str_len)
                put_padded = put_str + ' ' * max(0, 10 - put_str_len)
                # Build line with consistent spacing
                line = f"  {sym:<6}"
                line += f"{status_color}{status_display:<8}{reset_color()}"
                line += call_padded
                line += put_padded
                line += f"{realized_color if has_realized else ''}{realized_str:<10}{reset_color()}"
                line += f"{unrealized_str:<12}"
                line += f"{est_pnl_color}{est_pnl_str:<10}{reset_color()}"
                line += exp_str
                print(line)
            else:
                call_padded = call_str + ' ' * (12 - call_str_len)
                put_padded = put_str + ' ' * (12 - put_str_len)
                print(f"  {trade.ticker:<8} {status_color}{status_display:<12}{reset_color()}"
                      f"{call_padded}{put_padded}"
                      f"{realized_color if has_realized else ''}{realized_str:<12}{reset_color()}"
                      f"{unrealized_str:<10}")
                # Show expiration on second line for non-compact
                print(f"           Expiration: {trade.expiration} ({exp_str})")
                print()

        print()

    # === Completed Trades ===
    if completed_trades or show_all:
        print(bold("  COMPLETED TRADES"))
        if completed_trades:
            total_pnl = 0
            wins = 0
            losses = 0
            print(f"  {'Symbol':<8} {'Date':<12} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}")

            for trade in completed_trades[:10]:  # Most recent 10 (ordered by entry_datetime DESC)
                entry = trade.premium_paid or 0
                exit_val = trade.exit_pnl or 0
                pnl = exit_val
                ret = (pnl / entry * 100) if entry > 0 else 0

                total_pnl += pnl
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1

                pnl_color = '\033[92m' if pnl >= 0 else '\033[91m'

                print(f"  {trade.ticker:<8} {trade.earnings_date:<12} "
                      f"{format_currency(entry):<10} "
                      f"{format_currency(entry + pnl):<10} "
                      f"{pnl_color}{format_currency(pnl):<12}{reset_color()} "
                      f"{pnl_color}{ret:+.1f}%{reset_color()}")
            win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            pnl_color = '\033[92m' if total_pnl >= 0 else '\033[91m'
            print(f"  {'TOTAL':<8} {'':<12} {'':<10} {'':<10} "
                  f"{pnl_color}{format_currency(total_pnl):<12}{reset_color()} "
                  f"Win: {win_rate:.0f}%")
        else:
            print("  No completed trades")

    # === Summary Stats ===
    print(bold("  SUMMARY"))
    stats = logger.get_summary_stats()
    metrics = logger.get_execution_metrics()
    summary_parts = [f"Trades: {stats['total_trades']} ({stats['completed_trades']} done, {len(open_trades)} open)"]
    if stats['completed_trades'] > 0:
        summary_parts.append(f"P&L: {format_currency(stats['total_pnl'])} ({stats['avg_pnl_pct']*100:.1f}% avg)")
    print("  " + "  |  ".join(summary_parts))

    # Performance by timing (BMO vs AMC)
    if completed_trades:
        from collections import defaultdict
        timing_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
        for t in completed_trades:
            timing = t.earnings_timing or 'UNK'
            timing_stats[timing]['count'] += 1
            pnl = t.exit_pnl or 0
            timing_stats[timing]['pnl'] += pnl
            if pnl > 0:
                timing_stats[timing]['wins'] += 1

        timing_parts = []
        for timing in ['BMO', 'AMC']:
            if timing in timing_stats:
                s = timing_stats[timing]
                win_rate = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
                color = '\033[92m' if s['pnl'] >= 0 else '\033[91m'
                timing_parts.append(f"{timing}: {color}{format_currency(s['pnl'])}{reset_color()} ({s['count']} trades, {win_rate:.0f}% win)")
        if timing_parts:
            print("  " + "  |  ".join(timing_parts))

    # Next exit and screen countdown
    next_exit = get_next_exit_time()
    next_screen = get_next_screen_time()

    schedule_str = ""
    if open_trades and next_exit:
        schedule_str += f"Next Exit: {format_time_until(next_exit, now)}  |  "
    if next_screen:
        schedule_str += f"Next Screen: {format_time_until(next_screen, now)}"
    if schedule_str:
        print(f"  {schedule_str}")

    # === Candidate Preview ===
    try:
        # Determine screening date perspective based on market hours
        # Use datetime.now(ET) instead of now.astimezone() to avoid naive datetime issues
        et_now = datetime.now(ET)
        market_closed = et_now.hour >= 16

        # After market close, show NEXT screening session's targets (tomorrow's perspective)
        if market_closed:
            screening_date = date.today() + timedelta(days=1)
            amc_label = "Tmrw AMC:"   # Tomorrow's AMC (next screening)
            bmo_label = "Day+2 BMO:"  # Day after tomorrow's BMO
        else:
            screening_date = date.today()
            amc_label = "Today AMC:"
            bmo_label = "Tmrw BMO:"

        bmo_tomorrow, amc_today = get_tradeable_candidates(
            days_ahead=4,  # Look further ahead for after-hours preview
            trade_logger=None,
            fill_timing=True,
            verify_dates=False,
            screening_date=screening_date,
        )

        # Filter out symbols that already have open positions
        open_symbols = {t.ticker for t in open_trades} if open_trades else set()
        bmo_tomorrow = [c for c in bmo_tomorrow if c.symbol not in open_symbols]
        amc_today = [c for c in amc_today if c.symbol not in open_symbols]

        if bmo_tomorrow or amc_today:
            print(bold("  UPCOMING CANDIDATES"))

            # Show AMC first (more urgent)
            if amc_today:
                print(f"  {dim(amc_label)} ", end="")
                symbols = [c.symbol for c in amc_today[:10]]
                print(", ".join(symbols) + (f" (+{len(amc_today)-10})" if len(amc_today) > 10 else ""))

            # Then BMO
            if bmo_tomorrow:
                print(f"  {dim(bmo_label)}  ", end="")
                symbols = [c.symbol for c in bmo_tomorrow[:10]]
                print(", ".join(symbols) + (f" (+{len(bmo_tomorrow)-10})" if len(bmo_tomorrow) > 10 else ""))
    except Exception:
        pass  # Don't fail dashboard if candidate fetch fails

    # === Edge Realization (for completed trades) ===
    if completed_trades:
        edge_hits = 0
        edge_total = 0
        for t in completed_trades:
            if t.realized_move_pct is not None and t.implied_move is not None:
                edge_total += 1
                if t.realized_move_pct > t.implied_move:
                    edge_hits += 1
        if edge_total > 0:
            edge_hit_rate = edge_hits / edge_total * 100
            recent_with_data = [t for t in completed_trades[:10]
                               if t.realized_move_pct is not None and t.implied_move is not None]
            recent_str = ""
            for t in recent_with_data[:8]:
                recent_str += "\033[92m✓\033[0m" if t.realized_move_pct > t.implied_move else "\033[91m✗\033[0m"
            print(f"  Edge: {edge_hits}/{edge_total} ({edge_hit_rate:.0f}%) hit  Recent: {recent_str}")

    # === Execution Quality ===
    if metrics.total_orders >= 1:
        slip_str = f" | Slip: {metrics.avg_slippage_bps:.0f}bps" if metrics.avg_slippage_bps != 0 else ""
        if metrics.total_orders >= 5:
            recent_trades = [t for t in all_trades if t.status in ('filled', 'exited')][-10:]
            if recent_trades:
                recent_fills = sum(1 for t in recent_trades if t.status in ('filled', 'exited'))
                recent_fill_rate = recent_fills / len(recent_trades) * 100
                trend = ""
                if recent_fill_rate > metrics.fill_rate * 100 + 5:
                    trend = " \033[92m↑\033[0m"
                elif recent_fill_rate < metrics.fill_rate * 100 - 5:
                    trend = " \033[91m↓\033[0m"
                print(f"  Exec: {metrics.fill_rate*100:.0f}% fill ({recent_fill_rate:.0f}% recent{trend}){slip_str}")
            else:
                print(f"  Exec: {metrics.fill_rate*100:.0f}% fill{slip_str}")
        else:
            print(f"  Exec: {metrics.fill_rate*100:.0f}% fill{slip_str}")

    # === Equity Curve ===
    if len(completed_trades) >= 3:
        sorted_trades = sorted(
            [t for t in completed_trades if t.exit_pnl is not None],
            key=lambda t: t.exit_datetime or ""
        )
        if sorted_trades:
            cumulative = []
            running = 0
            for t in sorted_trades:
                running += t.exit_pnl
                cumulative.append(running)
            sparkline = make_sparkline(cumulative, width=30)
            final_pnl = cumulative[-1] if cumulative else 0
            pnl_color = '\033[92m' if final_pnl >= 0 else '\033[91m'

            # Weekly breakdown
            from collections import defaultdict
            weekly_pnl = defaultdict(float)
            for t in sorted_trades:
                if t.exit_datetime:
                    try:
                        exit_dt = datetime.fromisoformat(t.exit_datetime)
                        week_start = exit_dt - timedelta(days=exit_dt.weekday())
                        week_key = week_start.strftime("%m/%d")
                        weekly_pnl[week_key] += t.exit_pnl
                    except:
                        pass
            week_str = ""
            if weekly_pnl:
                for wk in sorted(weekly_pnl.keys())[-3:]:
                    pnl = weekly_pnl[wk]
                    color = '\033[92m' if pnl >= 0 else '\033[91m'
                    week_str += f" {wk}:{color}{format_currency(pnl)}{reset_color()}"
            print(f"  Curve: {sparkline} {pnl_color}{format_currency(final_pnl)}{reset_color()} |{week_str}")

    # === Recent Warnings/Errors ===
    # === Warnings/Errors (compact) ===
    warnings_errors = get_recent_warnings_errors(LOG_PATH)
    if warnings_errors:
        print(bold("  RECENT ISSUES"))
        for level, message in warnings_errors[:5]:  # Only show 5
            color = '\033[91m' if level == 'ERROR' else '\033[93m'
            print(f"  {color}{level[0]}{reset_color()} {message[:75]}")

    # === Counterfactual Summary (Non-Trades) ===
    non_trades = logger.get_non_trades()
    if non_trades:
        with_cf = [nt for nt in non_trades if nt.counterfactual_pnl is not None]
        if with_cf:
            total_cf_pnl = sum(float(nt.counterfactual_pnl or 0) for nt in with_cf)
            profitable_cf = sum(1 for nt in with_cf if float(nt.counterfactual_pnl or 0) > 0)
            cf_win_rate = profitable_cf / len(with_cf) * 100
            cf_color = '\033[92m' if total_cf_pnl >= 0 else '\033[91m'
            with_spread = [nt for nt in non_trades if nt.counterfactual_pnl_with_spread is not None]
            spread_pnl = sum(float(nt.counterfactual_pnl_with_spread or 0) for nt in with_spread) if with_spread else 0
            spread_color = '\033[92m' if spread_pnl >= 0 else '\033[91m'
            from collections import Counter
            recent_reasons = Counter(nt.rejection_reason for nt in non_trades[:20] if nt.rejection_reason)
            reason_str = ", ".join(f"{r[:15]}:{c}" for r, c in recent_reasons.most_common(2))
            print(f"  Counterfactual: {cf_color}{format_currency(total_cf_pnl)}{reset_color()} ({cf_win_rate:.0f}% win) | w/spread: {spread_color}{format_currency(spread_pnl)}{reset_color()} | {reason_str}")


def get_open_positions(logger: TradeLogger) -> list:
    """Get open positions that can be closed."""
    all_trades = logger.get_trades()
    return [t for t in all_trades if t.status in ('pending', 'submitted', 'filled', 'partial', 'exiting', 'partial_exit')]


def show_llm_details_interactive(logger: TradeLogger):
    """Interactive LLM check details viewer."""
    all_trades = logger.get_trades()
    if not all_trades:
        print("\n  No trades to show LLM details for.")
        input("  Press Enter to continue...")
        return

    # Build list of trades with LLM checks
    trades_with_llm = []
    for trade in all_trades:
        if trade.entry_datetime:
            llm_check = logger.get_llm_check_for_trade(trade.ticker, trade.entry_datetime)
            if llm_check:
                trades_with_llm.append((trade, llm_check))

    if not trades_with_llm:
        print("\n  No LLM checks found for any trades.")
        input("  Press Enter to continue...")
        return

    print("\n" + bold("  LLM SANITY CHECK DETAILS"))
    print("  " + "-" * 60)
    print()

    # List trades with LLM status
    for i, (trade, llm_check) in enumerate(trades_with_llm, 1):
        decision = llm_check['decision']
        llm_colors = {'PASS': '\033[92m', 'WARN': '\033[93m', 'NO_TRADE': '\033[91m'}
        llm_color = llm_colors.get(decision, '')
        print(f"  {i}. {trade.ticker:<8} {llm_color}{decision:<8}{reset_color()} "
              f"{trade.earnings_date} ({trade.earnings_timing or '?'})")

    print()
    print("  0. Cancel")
    print()

    try:
        choice = input("  Select trade to view LLM details: ").strip()
        if not choice or choice == '0':
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(trades_with_llm):
            print("  Invalid selection.")
            input("  Press Enter to continue...")
            return

        trade, llm_check = trades_with_llm[idx]

        # Display full LLM check details
        print()
        print(bold(f"  LLM Check for {trade.ticker}"))
        print("  " + "-" * 60)
        print()

        decision = llm_check['decision']
        llm_colors = {'PASS': '\033[92m', 'WARN': '\033[93m', 'NO_TRADE': '\033[91m'}
        llm_color = llm_colors.get(decision, '')
        print(f"  Decision:    {llm_color}{decision}{reset_color()}")
        print(f"  Model:       {llm_check.get('model', 'N/A')}")
        print(f"  Latency:     {llm_check.get('latency_ms', 0)}ms")
        print(f"  Timestamp:   {llm_check.get('ts', 'N/A')}")
        print()

        risk_flags = llm_check.get('risk_flags', [])
        if risk_flags:
            print(f"  Risk Flags:")
            for flag in risk_flags:
                print(f"    - {flag}")
            print()

        reasons = llm_check.get('reasons', [])
        if reasons:
            print(f"  Reasons:")
            for reason in reasons:
                # Word wrap long reasons
                import textwrap
                wrapped = textwrap.wrap(reason, width=70)
                for j, line in enumerate(wrapped):
                    if j == 0:
                        print(f"    - {line}")
                    else:
                        print(f"      {line}")
            print()

        search_queries = llm_check.get('search_queries', [])
        if search_queries:
            print(f"  Search Queries:")
            for query in search_queries:
                print(f"    - {query}")
            print()

        input("  Press Enter to continue...")

    except ValueError:
        print("  Invalid input.")
        input("  Press Enter to continue...")
    except KeyboardInterrupt:
        pass


def close_position_interactive(logger: TradeLogger):
    """Interactive position closing with fill tracking."""
    global _ib

    open_trades = get_open_positions(logger)
    if not open_trades:
        print("\n  No open positions to close.")
        input("  Press Enter to continue...")
        return

    print("\n" + bold("  CLOSE POSITION"))
    print("  " + "-" * 50)
    print()

    # List positions with numbers
    for i, trade in enumerate(open_trades, 1):
        status_color = get_status_color(trade.status)
        print(f"  {i}. {trade.ticker:<8} {status_color}{trade.status:<10}{reset_color()} "
              f"Strike: {trade.strikes} Exp: {trade.expiration}")

    print()
    print("  0. Cancel")
    print()

    try:
        choice = input("  Select position to close: ").strip()
        if not choice or choice == '0':
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(open_trades):
            print("  Invalid selection.")
            input("  Press Enter to continue...")
            return

        trade = open_trades[idx]

        # Connect to IB if needed
        # We need a synchronous connection here
        ib = connect_ib()
        if not ib:
            print("  Could not connect to IBKR.")
            input("  Press Enter to continue...")
            return

        # Parse trade details
        import json
        try:
            strikes = json.loads(trade.strikes) if trade.strikes else []
        except (json.JSONDecodeError, TypeError):
            strikes = []
        strike = strikes[0] if strikes else 0
        expiry = trade.expiration

        # Get current positions from IBKR
        positions = ib.positions()
        symbol_positions = [p for p in positions if p.contract.symbol == trade.ticker]

        if not symbol_positions:
            print(f"  No IBKR positions found for {trade.ticker}")
            input("  Press Enter to continue...")
            return

        print()
        print(f"  Found {len(symbol_positions)} position(s) for {trade.ticker}:")
        for p in symbol_positions:
            c = p.contract
            print(f"    {c.right} strike={c.strike} qty={int(p.position)} avgCost=${p.avgCost:.2f}")

        print()
        confirm = input(f"  Close ALL {trade.ticker} positions? (y/n): ").strip().lower()
        if confirm != 'y':
            print("  Cancelled.")
            input("  Press Enter to continue...")
            return

        # Close each position and track fills
        from ib_insync import Option, LimitOrder, Stock

        placed_orders = []  # Track (order_trade, contract, qty, price) for fill checking
        total_exit_value = 0.0
        total_contracts = 0

        for pos in symbol_positions:
            if pos.position == 0:
                continue

            # Create a fresh contract with exchange set
            c = pos.contract
            contract = Option(
                c.symbol, c.lastTradeDateOrContractMonth, c.strike, c.right,
                'SMART', tradingClass=c.tradingClass
            )
            ib.qualifyContracts(contract)

            qty = abs(int(pos.position))
            action = 'SELL' if pos.position > 0 else 'BUY'

            # Get current quote
            ticker = None
            price = None
            try:
                ticker = ib.reqMktData(contract, '', False, False)
                ib.sleep(1)

                # Use bid for sells, ask for buys
                def valid_price(p):
                    return p is not None and not math.isnan(p) and p > 0

                if action == 'SELL':
                    if valid_price(ticker.bid):
                        price = ticker.bid
                    elif valid_price(ticker.last):
                        price = ticker.last
                    else:
                        price = None
                else:
                    if valid_price(ticker.ask):
                        price = ticker.ask
                    elif valid_price(ticker.last):
                        price = ticker.last
                    else:
                        price = None
            finally:
                # Always cancel market data subscriptions to prevent leaks
                if ticker is not None:
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass

            if not price:
                print(f"    No price available for {c.right} (market closed?)")
                print(f"    Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")
                manual = input("    Enter price manually (or press Enter to skip): ").strip()
                if manual:
                    try:
                        price = float(manual)
                    except ValueError:
                        print("    Invalid price, skipping.")
                        continue
                else:
                    continue

            print(f"    Closing {qty} {c.right} @ ${price:.2f}...")

            order = LimitOrder(action, qty, price)
            order_trade = ib.placeOrder(contract, order)
            placed_orders.append((order_trade, contract, qty, price, c.right))
            ib.sleep(1)

        if not placed_orders:
            print()
            print(f"  No orders placed for {trade.ticker}. Status unchanged.")
            input("  Press Enter to continue...")
            return

        # Wait for fills (up to 30 seconds)
        print()
        print("  Waiting for fills...")
        max_wait = 30
        start_time = time.time()
        all_filled = False

        while time.time() - start_time < max_wait:
            ib.sleep(1)
            all_filled = all(ot.orderStatus.status == 'Filled' for ot, _, _, _, _ in placed_orders)
            filled_count = sum(1 for ot, _, _, _, _ in placed_orders if ot.orderStatus.status == 'Filled')
            print(f"    {filled_count}/{len(placed_orders)} orders filled...", end='\r')
            if all_filled:
                break

        print()

        # Calculate total exit value from fills
        total_exit_value = 0.0
        for order_trade, contract, qty, limit_price, right in placed_orders:
            status = order_trade.orderStatus.status
            if status == 'Filled':
                fill_price = order_trade.orderStatus.avgFillPrice
                exit_value = fill_price * qty * 100
                total_exit_value += exit_value
                print(f"    {right}: Filled {qty} @ ${fill_price:.2f} = ${exit_value:.2f}")
            else:
                print(f"    {right}: {status} (not filled)")

        # Calculate P&L
        entry_value = trade.premium_paid or 0
        exit_pnl = total_exit_value - entry_value

        # Calculate exit fill price (per straddle)
        contracts = trade.contracts or 1
        exit_fill_price = total_exit_value / contracts / 100 if contracts > 0 else 0

        # Get spot price at exit for realized move calculation
        spot_at_exit = None
        realized_move = None
        realized_move_pct = None
        try:
            stock = Stock(trade.ticker, 'SMART', 'USD')
            ib.qualifyContracts(stock)
            ticker = ib.reqMktData(stock, '', False, False)
            ib.sleep(1)
            spot_at_exit = ticker.marketPrice()
            if spot_at_exit is None or math.isnan(spot_at_exit) or spot_at_exit <= 0:
                spot_at_exit = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            ib.cancelMktData(stock)

            if spot_at_exit and spot_at_exit > 0 and trade.spot_at_entry:
                realized_move = abs(spot_at_exit - trade.spot_at_entry)
                realized_move_pct = realized_move / trade.spot_at_entry
        except Exception as e:
            print(f"    Warning: Could not fetch spot price: {e}")

        # Update database with exit info
        exit_datetime = datetime.now().isoformat()
        pnl_pct = exit_pnl / entry_value if entry_value > 0 else 0

        logger.update_trade(
            trade.trade_id,
            status='exited',
            exit_datetime=exit_datetime,
            exit_fill_price=exit_fill_price,
            exit_pnl=exit_pnl,
            exit_pnl_pct=pnl_pct,
            spot_at_exit=spot_at_exit,
            realized_move=realized_move,
            realized_move_pct=realized_move_pct,
            notes='Manually closed via dashboard'
        )

        # Display result
        pnl_color = '\033[92m' if exit_pnl >= 0 else '\033[91m'
        print()
        print(f"  {trade.ticker} CLOSED")
        print(f"    Entry:  {format_currency(entry_value)}")
        print(f"    Exit:   {format_currency(total_exit_value)}")
        print(f"    P&L:    {pnl_color}{format_currency(exit_pnl)} ({pnl_pct*100:+.1f}%){reset_color()}")
        if realized_move_pct is not None:
            print(f"    Move:   {realized_move_pct*100:.1f}% (implied: {trade.implied_move*100:.1f}%)" if trade.implied_move else f"    Move:   {realized_move_pct*100:.1f}%")

        if not all_filled:
            print()
            print("  \033[93mWARNING: Some orders may not have filled. Check TWS.\033[0m")

    except ValueError:
        print("  Invalid input.")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    input("  Press Enter to continue...")


def check_keyboard_input(timeout: float = 0.1) -> Optional[str]:
    """Check for keyboard input without blocking (Unix only)."""
    if sys.platform == 'win32':
        # Windows - use msvcrt
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', errors='ignore')
        return None
    else:
        # Unix - use select
        import termios
        import tty

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return None


def main():
    parser = argparse.ArgumentParser(description='Earnings Trading Dashboard')
    parser.add_argument('--watch', '-w', action='store_true', help='Auto-refresh every 5 seconds')
    parser.add_argument('--all', '-a', action='store_true', help='Show all trades including completed')
    parser.add_argument('--live', '-l', action='store_true', help='Connect to IBKR for live prices')
    parser.add_argument('--compact', '-c', action='store_true', help='Compact 1-line per position format')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--sound', '-s', action='store_true', help='Play sound on fills (watch mode)')
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("No trades have been recorded yet.")
        return

    logger = TradeLogger(db_path=DB_PATH)

    async def run_dashboard():
        if args.watch:
            try:
                # Use cached data for flickering prevention
                live_data_cache = {}
                # Track trade states for sound alerts
                prev_trade_states = {}

                while True:
                    # In live mode with cached data, clear screen ONLY before new render

                    if args.live:
                         # 1. Fetch data (async, parallel)
                         trades = logger.get_trades()
                         open_trades = [t for t in trades if t.status in ('pending', 'submitted', 'filled', 'partial', 'exiting', 'partial_exit')]

                         new_live_data = {}
                         if open_trades:
                             ib = await connect_ib_async()
                             if ib:
                                 tasks = []
                                 for trade in open_trades:
                                     if trade.status not in ('filled', 'partial', 'exiting', 'partial_exit'):
                                         continue
                                     tasks.append((trade.trade_id, get_position_live_value_async(ib, trade)))

                                 if tasks:
                                     results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
                                     for i, res in enumerate(results):
                                         if not isinstance(res, Exception):
                                             new_live_data[tasks[i][0]] = res

                         # Update cache
                         live_data_cache.update(new_live_data)

                         # 3. Clear and Render
                         clear_screen()
                         render_dashboard(logger, show_all=args.all, live=True, compact=args.compact, live_data_map=live_data_cache)

                    else:
                        clear_screen()
                        render_dashboard(logger, show_all=args.all, live=False, compact=args.compact)

                    # Sound alerts on state changes (fills, exits)
                    if args.sound:
                        trades = logger.get_trades()
                        current_states = {t.trade_id: t.status for t in trades}
                        for trade_id, status in current_states.items():
                            prev_status = prev_trade_states.get(trade_id)
                            if prev_status and prev_status != status:
                                # State changed - check if it's a fill or exit
                                if status in ('filled', 'exited'):
                                    print('\a', end='', flush=True)  # Terminal bell
                        prev_trade_states.update(current_states)

                    mode = "LIVE" if args.live else "DB only"
                    compact_str = " | COMPACT" if args.compact else ""
                    sound_str = " | SOUND" if args.sound else ""
                    print(f"  [{mode}{compact_str}{sound_str} | Refreshing every {args.interval}s | c=close, l=llm, r=refresh, q=quit]")

                    # Wait for interval...
                    start = time.time()
                    while time.time() - start < args.interval:
                        key = check_keyboard_input(0.1)
                        if key:
                            if key.lower() == 'q':
                                raise KeyboardInterrupt
                            elif key.lower() == 'c':
                                close_position_interactive(logger) # This is sync but interactive
                                break  # Refresh after closing
                            elif key.lower() == 'l':
                                show_llm_details_interactive(logger)
                                break  # Refresh after viewing
                            elif key.lower() == 'r':
                                break  # Refresh now
                        await asyncio.sleep(0.01) # Yield to event loop if needed

            except KeyboardInterrupt:
                disconnect_ib()
                print("\n  Dashboard closed.")
        else:
            if args.live:
                # One-shot live fetch
                ib = await connect_ib_async()
                live_data_cache = {}
                if ib:
                     trades = logger.get_trades()
                     open_trades = [t for t in trades if t.status in ('pending', 'submitted', 'filled', 'partial', 'exiting', 'partial_exit')]
                     tasks = []
                     for trade in open_trades:
                         if trade.status not in ('filled', 'partial', 'exiting', 'partial_exit'):
                             continue
                         tasks.append((trade.trade_id, get_position_live_value_async(ib, trade)))
                     if tasks:
                         results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
                         for i, res in enumerate(results):
                             if not isinstance(res, Exception):
                                 live_data_cache[tasks[i][0]] = res

                render_dashboard(logger, show_all=args.all, live=True, compact=args.compact, live_data_map=live_data_cache)
            else:
                render_dashboard(logger, show_all=args.all, live=False, compact=args.compact)
            disconnect_ib()


    if args.live or args.watch:
        try:
             asyncio.run(run_dashboard())
        except KeyboardInterrupt:
            pass
    else:
        # For non-watch, non-live mode, we can run sync or async.
        # But since we wrapped logic in run_dashboard, use it.
        try:
             asyncio.run(run_dashboard())
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
