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
import os
import select
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

from trading.earnings.logging import TradeLogger
from trading.earnings.screener import fetch_upcoming_earnings

DB_PATH = PROJECT_ROOT / 'data' / 'earnings_trades.db'
LOG_PATH = PROJECT_ROOT / 'logs' / 'daemon.log'

# Global IB connection for live mode
_ib = None


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def format_currency(value: float) -> str:
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
        now = datetime.now()

    if target_dt.tzinfo:
        target_dt = target_dt.replace(tzinfo=None)

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
        now = datetime.now()

    if isinstance(past_dt, str):
        past_dt = datetime.fromisoformat(past_dt)

    if past_dt.tzinfo:
        past_dt = past_dt.replace(tzinfo=None)

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


def get_next_screen_time() -> Optional[datetime]:
    """Get next screening time (3:00 PM ET on weekdays)."""
    now_et = datetime.now(ET)
    screen_time = now_et.replace(hour=15, minute=0, second=0, microsecond=0)

    # If past 3 PM today, next screen is tomorrow
    if now_et.hour >= 15:
        screen_time += timedelta(days=1)

    # Skip weekends
    while screen_time.weekday() >= 5:
        screen_time += timedelta(days=1)

    # Return as naive datetime in local time for comparison
    # Convert ET to local time
    return screen_time.astimezone().replace(tzinfo=None)


def get_next_exit_time() -> Optional[datetime]:
    """Get next exit time (2:45 PM ET on weekdays)."""
    now_et = datetime.now(ET)
    exit_time = now_et.replace(hour=14, minute=45, second=0, microsecond=0)

    # If past 2:45 PM today, next exit is tomorrow
    if now_et.hour > 14 or (now_et.hour == 14 and now_et.minute >= 45):
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
    """Connect to IB Gateway for live prices."""
    global _ib
    if _ib is not None and _ib.isConnected():
        return _ib

    try:
        from ib_insync import IB
        _ib = IB()
        _ib.connect('127.0.0.1', 4002, clientId=client_id)
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


def get_live_option_price(ib, symbol: str, expiry: str, strike: float, right: str) -> Optional[dict]:
    """Get live bid/ask for an option."""
    if ib is None:
        return None

    try:
        from ib_insync import Option
        opt = Option(symbol, expiry, strike, right, 'SMART', tradingClass=symbol)
        qualified = ib.qualifyContracts(opt)

        if not qualified or not opt.conId:
            # Try without tradingClass
            opt = Option(symbol, expiry, strike, right, 'SMART')
            qualified = ib.qualifyContracts(opt)
            if not qualified or not opt.conId:
                return {'error': 'contract not found'}

        ticker = ib.reqMktData(opt, '', False, False)

        # Wait longer for data, check incrementally
        for _ in range(4):
            ib.sleep(0.3)
            if ticker.bid and ticker.bid > 0:
                break

        # IBKR returns -1.0 when no quote is available (market closed, no liquidity)
        # Also check for NaN (value != value)
        def valid_price(p):
            return p is not None and p == p and p > 0

        result = {
            'bid': ticker.bid if valid_price(ticker.bid) else None,
            'ask': ticker.ask if valid_price(ticker.ask) else None,
            'last': ticker.last if valid_price(ticker.last) else None,
        }

        ib.cancelMktData(opt)
        return result
    except Exception as e:
        return {'error': str(e)}


def get_position_live_value(ib, trade) -> Optional[dict]:
    """Get live value for a position."""
    if ib is None or not trade.strikes or not trade.expiration:
        return {'error': 'missing trade data'}

    try:
        import json
        strikes = json.loads(trade.strikes) if isinstance(trade.strikes, str) else trade.strikes
        strike = strikes[0] if isinstance(strikes, list) else float(strikes)
        expiry = trade.expiration.replace('-', '')

        call_data = get_live_option_price(ib, trade.ticker, expiry, strike, 'C')
        put_data = get_live_option_price(ib, trade.ticker, expiry, strike, 'P')

        # Check for errors from option price fetch
        if call_data and 'error' in call_data:
            return {'error': f"call: {call_data['error']}"}
        if put_data and 'error' in put_data:
            return {'error': f"put: {put_data['error']}"}

        if call_data and put_data:
            call_mid = None
            put_mid = None

            if call_data.get('bid') and call_data.get('ask'):
                call_mid = (call_data['bid'] + call_data['ask']) / 2
            elif call_data.get('last'):
                call_mid = call_data['last']

            if put_data.get('bid') and put_data.get('ask'):
                put_mid = (put_data['bid'] + put_data['ask']) / 2
            elif put_data.get('last'):
                put_mid = put_data['last']

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


def render_dashboard(logger: TradeLogger, show_all: bool = False, live: bool = False, compact: bool = False):
    """Render the dashboard.

    Args:
        compact: If True, show condensed 1-line per position format
    """
    now = datetime.now()

    # Market status
    market_status, market_color = get_market_status()

    print(bold("=" * 120))
    print(bold(f"  EARNINGS TRADING DASHBOARD - {now.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(f"  Market: {market_color}{market_status}{reset_color()}")
    print(bold("=" * 120))
    print()

    # Get all trades
    all_trades = logger.get_trades()

    if not all_trades:
        print("  No trades recorded yet.")
        print()
        return

    # Categorize trades
    open_trades = [t for t in all_trades if t.status in ('pending', 'submitted', 'filled', 'partial', 'exiting')]
    completed_trades = [t for t in all_trades if t.status in ('completed', 'exited')]

    # === Open Positions ===
    print(bold("  OPEN POSITIONS"))
    print("  " + "-" * 116)

    # Connect to IB if live mode
    ib = None
    if live:
        ib = connect_ib()
        if ib:
            print("  [Connected to IBKR for live prices]")

            # Cross-check DB vs IBKR positions
            ibkr_positions = ib.positions()
            ibkr_symbols = set(p.contract.symbol for p in ibkr_positions if p.position != 0)
            db_symbols = set(t.ticker for t in open_trades if t.status == 'filled')

            # Check for mismatches
            in_db_not_ibkr = db_symbols - ibkr_symbols
            in_ibkr_not_db = ibkr_symbols - db_symbols

            if in_db_not_ibkr or in_ibkr_not_db:
                print(f"  \033[93m⚠ POSITION MISMATCH: DB={len(db_symbols)} symbols, IBKR={len(ibkr_symbols)} symbols\033[0m")
                if in_db_not_ibkr:
                    print(f"  \033[93m  In DB but not IBKR: {', '.join(sorted(in_db_not_ibkr))}\033[0m")
        print()

    total_unrealized_pnl = 0
    has_live_data = False

    if open_trades:
        if compact:
            # Compact format: 1 line per position
            print(f"  {'Symbol':<6} {'Status':<8} {'Strike':<8} {'Exp':<8} {'Time':<5} {'Entry':<9} {'Curr':<9} {'P&L':<9} {'Edge':<6} {'Impl':<6} {'Sprd':<5}")
            print("  " + "-" * 116)
        else:
            print(f"  {'Symbol':<8} {'Earnings':<12} {'Status':<10} {'Entry':<10} {'Current':<10} {'P&L':<12}")
            print("  " + "-" * 116)

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

            if live and ib:
                live_data = get_position_live_value(ib, trade)
                if live_data:
                    if 'error' in live_data:
                        live_error = live_data['error']
                    else:
                        has_live_data = True
                        current_price = f"${live_data['straddle_mid']:.2f}"
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

            if compact:
                # Compact: single line with key info
                import json
                strikes = json.loads(trade.strikes) if trade.strikes else []
                strike_str = f"{strikes[0]:.0f}" if strikes else "?"
                expiry_short = trade.expiration[-5:] if trade.expiration else "?"  # MM-DD
                timing = (trade.earnings_timing or "?")[:3]
                edge_str = f"{trade.edge_q75*100:.0f}%" if trade.edge_q75 else "-"
                impl_str = f"{trade.implied_move*100:.0f}%" if trade.implied_move else "-"

                # Calculate spread %
                spread_str = "-"
                if trade.entry_quoted_bid and trade.entry_quoted_ask and trade.entry_quoted_mid:
                    spread = trade.entry_quoted_ask - trade.entry_quoted_bid
                    spread_pct = (spread / trade.entry_quoted_mid) * 100
                    spread_str = f"{spread_pct:.0f}%"

                print(f"  {trade.ticker:<6} {status_color}{status_display:<8}{reset_color()} "
                      f"{strike_str:<8} {expiry_short:<8} {timing:<5} "
                      f"{entry_price_short:<9} {current_price_short:<9} {pnl_color}{pnl_str:<9}{reset_color()} "
                      f"{edge_str:<6} {impl_str:<6} {spread_str:<5}")

                # Only show errors/warnings on second line if critical
                if trade.status == 'partial' and trade.notes:
                    print(f"         \033[91m^ {trade.notes}\033[0m")
            else:
                # Full format: multiple lines per position
                print(f"  {trade.ticker:<8} {trade.earnings_date:<12} "
                      f"{status_color}{status_display:<10}{reset_color()} "
                      f"{entry_price:<10} {current_price:<10} {pnl_color}{pnl_str:<12}{reset_color()}")

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
                print()

        # Show total unrealized P&L and risk summary
        print("  " + "-" * 116)

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

    # === Completed Trades ===
    if completed_trades or show_all:
        print(bold("  COMPLETED TRADES"))
        print("  " + "-" * 116)

        if completed_trades:
            total_pnl = 0
            wins = 0
            losses = 0

            print(f"  {'Symbol':<8} {'Date':<12} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}")
            print("  " + "-" * 116)

            for trade in completed_trades[-10:]:  # Last 10
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

            print("  " + "-" * 116)
            win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            pnl_color = '\033[92m' if total_pnl >= 0 else '\033[91m'
            print(f"  {'TOTAL':<8} {'':<12} {'':<10} {'':<10} "
                  f"{pnl_color}{format_currency(total_pnl):<12}{reset_color()} "
                  f"Win: {win_rate:.0f}%")
        else:
            print("  No completed trades")
        print()

    # === Summary Stats ===
    print(bold("  SUMMARY"))
    print("  " + "-" * 116)

    stats = logger.get_summary_stats()
    metrics = logger.get_execution_metrics()

    print(f"  Total Trades: {stats['total_trades']:<10} "
          f"Completed: {stats['completed_trades']:<10} "
          f"Open: {len(open_trades)}")

    if stats['completed_trades'] > 0:
        print(f"  Total P&L: {format_currency(stats['total_pnl']):<12} "
              f"Avg P&L: {stats['avg_pnl_pct']*100:.1f}%")

    if metrics.total_orders > 0:
        print(f"  Fill Rate: {metrics.fill_rate*100:.1f}%  "
              f"Avg Slippage: {metrics.avg_slippage_bps:.1f} bps")

    # Next exit and screen countdown
    next_exit = get_next_exit_time()
    next_screen = get_next_screen_time()

    schedule_str = ""
    if open_trades and next_exit:
        schedule_str += f"Next Exit: {format_time_until(next_exit, now)}  |  "
    if next_screen:
        schedule_str += f"Next Screen: {format_time_until(next_screen, now)}"

        # Fetch upcoming earnings count
        try:
            tomorrow = date.today() + timedelta(days=1)
            events = fetch_upcoming_earnings(days_ahead=3)

            # Count tradeable candidates (BMO tomorrow + AMC today)
            bmo_tomorrow = sum(1 for e in events if e.earnings_date == tomorrow and e.timing == 'BMO')
            amc_today = sum(1 for e in events if e.earnings_date == date.today() and e.timing == 'AMC')
            total_candidates = bmo_tomorrow + amc_today

            if total_candidates > 0:
                schedule_str += f"  |  Candidates: {total_candidates} ({bmo_tomorrow} BMO tmrw, {amc_today} AMC today)"
            else:
                schedule_str += f"  |  No candidates today"
        except Exception:
            pass  # Don't fail dashboard if earnings fetch fails

    if schedule_str:
        print(f"  {schedule_str}")

    print()

    # === Edge Realization (for completed trades) ===
    if completed_trades:
        # Calculate edge hit rate: did realized move exceed implied?
        edge_hits = 0
        edge_total = 0
        for t in completed_trades:
            if t.realized_move_pct is not None and t.implied_move is not None:
                edge_total += 1
                if t.realized_move_pct > t.implied_move:
                    edge_hits += 1

        if edge_total > 0:
            edge_hit_rate = edge_hits / edge_total * 100
            print(bold("  EDGE REALIZATION"))
            print("  " + "-" * 116)
            print(f"  Realized > Implied: {edge_hits}/{edge_total} ({edge_hit_rate:.0f}%)")

            # Show recent edge performance
            recent_with_data = [t for t in completed_trades[-10:]
                               if t.realized_move_pct is not None and t.implied_move is not None]
            if recent_with_data:
                recent_str = ""
                for t in recent_with_data[-8:]:
                    if t.realized_move_pct > t.implied_move:
                        recent_str += "\033[92m✓\033[0m"  # Green check
                    else:
                        recent_str += "\033[91m✗\033[0m"  # Red X
                print(f"  Recent (last 8): {recent_str}")
            print()

    # === Execution Quality ===
    if metrics.total_orders >= 5:
        print(bold("  EXECUTION QUALITY"))
        print("  " + "-" * 116)

        # Fill rate trend (last 10 vs all-time)
        recent_trades = [t for t in all_trades if t.status in ('filled', 'exited', 'cancelled')][-10:]
        if recent_trades:
            recent_fills = sum(1 for t in recent_trades if t.status in ('filled', 'exited'))
            recent_fill_rate = recent_fills / len(recent_trades) * 100
            trend = ""
            if recent_fill_rate > metrics.fill_rate * 100 + 5:
                trend = " \033[92m↑\033[0m"
            elif recent_fill_rate < metrics.fill_rate * 100 - 5:
                trend = " \033[91m↓\033[0m"

            print(f"  Fill Rate: {metrics.fill_rate*100:.0f}% all-time | {recent_fill_rate:.0f}% recent{trend}")

        # Average slippage detail
        if metrics.avg_slippage_bps != 0:
            print(f"  Slippage: avg {metrics.avg_slippage_bps:.1f} bps | max {metrics.max_slippage_bps:.1f} bps")

        print()

    # === Equity Curve ===
    if len(completed_trades) >= 3:
        print(bold("  EQUITY CURVE"))
        print("  " + "-" * 116)

        # Build cumulative P&L series
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

            sparkline = make_sparkline(cumulative, width=40)
            final_pnl = cumulative[-1] if cumulative else 0
            pnl_color = '\033[92m' if final_pnl >= 0 else '\033[91m'

            print(f"  {sparkline}  {pnl_color}{format_currency(final_pnl)}{reset_color()}")

            # Weekly breakdown (last 4 weeks)
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

            if weekly_pnl:
                recent_weeks = sorted(weekly_pnl.keys())[-4:]
                week_str = "  Weekly: "
                for wk in recent_weeks:
                    pnl = weekly_pnl[wk]
                    color = '\033[92m' if pnl >= 0 else '\033[91m'
                    week_str += f"{wk}: {color}{format_currency(pnl)}{reset_color()}  "
                print(week_str)

        print()

    # === Recent Warnings/Errors ===
    warnings_errors = get_recent_warnings_errors(LOG_PATH)
    if warnings_errors:
        print(bold("  RECENT WARNINGS/ERRORS"))
        print("  " + "-" * 116)
        for level, message in warnings_errors:
            if level == 'ERROR':
                color = '\033[91m'  # Red
            else:
                color = '\033[93m'  # Yellow
            print(f"  {color}{level:<7}{reset_color()} {message}")
        print()

    # === Counterfactual Summary (Non-Trades) ===
    non_trades = logger.get_non_trades()
    if non_trades:
        # Calculate counterfactual stats for trades with realized data
        with_cf = [nt for nt in non_trades if nt.counterfactual_pnl is not None]

        print(bold("  COUNTERFACTUAL ANALYSIS (Rejected Candidates)"))
        print("  " + "-" * 116)

        if with_cf:
            avg_cf_pnl = sum(nt.counterfactual_pnl for nt in with_cf) / len(with_cf)
            total_cf_pnl = sum(nt.counterfactual_pnl for nt in with_cf)
            profitable_cf = sum(1 for nt in with_cf if nt.counterfactual_pnl > 0)
            cf_win_rate = profitable_cf / len(with_cf) * 100

            cf_color = '\033[92m' if total_cf_pnl >= 0 else '\033[91m'
            print(f"  If we traded all rejections: {cf_color}{format_currency(total_cf_pnl)}{reset_color()} "
                  f"(avg: {format_currency(avg_cf_pnl)}, win rate: {cf_win_rate:.0f}%)")

            # With spread costs
            with_spread = [nt for nt in non_trades if nt.counterfactual_pnl_with_spread is not None]
            if with_spread:
                total_cf_spread = sum(nt.counterfactual_pnl_with_spread for nt in with_spread)
                spread_color = '\033[92m' if total_cf_spread >= 0 else '\033[91m'
                print(f"  With spread costs:           {spread_color}{format_currency(total_cf_spread)}{reset_color()}")
        else:
            print(f"  {len(non_trades)} candidates rejected (counterfactual data pending backfill)")

        # Show recent rejections breakdown by reason
        from collections import Counter
        recent_reasons = Counter(nt.rejection_reason for nt in non_trades[:20] if nt.rejection_reason)
        if recent_reasons:
            reason_str = ", ".join(f"{r}: {c}" for r, c in recent_reasons.most_common(3))
            print(f"  Recent reasons: {reason_str}")

        print()


def get_open_positions(logger: TradeLogger) -> list:
    """Get open positions that can be closed."""
    all_trades = logger.get_trades()
    return [t for t in all_trades if t.status in ('pending', 'submitted', 'filled', 'partial', 'exiting')]


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
        ib = connect_ib()
        if not ib:
            print("  Could not connect to IBKR.")
            input("  Press Enter to continue...")
            return

        # Parse trade details
        import json
        strikes = json.loads(trade.strikes) if trade.strikes else []
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
        from ib_insync import Option, LimitOrder

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
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(1)

            # Use bid for sells, ask for buys
            def valid_price(p):
                return p is not None and p == p and p > 0

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

            ib.cancelMktData(contract)

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
            notes='Manually closed via dashboard'
        )

        # Display result
        pnl_color = '\033[92m' if exit_pnl >= 0 else '\033[91m'
        print()
        print(f"  {trade.ticker} CLOSED")
        print(f"    Entry:  {format_currency(entry_value)}")
        print(f"    Exit:   {format_currency(total_exit_value)}")
        print(f"    P&L:    {pnl_color}{format_currency(exit_pnl)} ({pnl_pct*100:+.1f}%){reset_color()}")

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
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("No trades have been recorded yet.")
        return

    logger = TradeLogger(db_path=DB_PATH)

    if args.watch:
        try:
            while True:
                clear_screen()
                render_dashboard(logger, show_all=args.all, live=args.live, compact=args.compact)
                mode = "LIVE" if args.live else "DB only"
                compact_str = " | COMPACT" if args.compact else ""
                print(f"  [{mode}{compact_str} | Refreshing every {args.interval}s | c=close, r=refresh, q=quit]")

                # Wait for interval, checking for keyboard input
                start = time.time()
                while time.time() - start < args.interval:
                    key = check_keyboard_input(0.1)
                    if key:
                        if key.lower() == 'q':
                            raise KeyboardInterrupt
                        elif key.lower() == 'c':
                            close_position_interactive(logger)
                            break  # Refresh after closing
                        elif key.lower() == 'r':
                            break  # Refresh now

        except KeyboardInterrupt:
            disconnect_ib()
            print("\n  Dashboard closed.")
    else:
        render_dashboard(logger, show_all=args.all, live=args.live, compact=args.compact)
        disconnect_ib()


if __name__ == '__main__':
    main()
