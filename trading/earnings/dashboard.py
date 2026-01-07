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
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

from trading.earnings.logging import TradeLogger

DB_PATH = PROJECT_ROOT / 'data' / 'earnings_trades.db'

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
        'completed': '\033[94m',    # Blue
        'cancelled': '\033[90m',    # Gray
        'error': '\033[91m',        # Red
    }
    return colors.get(status.lower(), '\033[0m')


def reset_color() -> str:
    return '\033[0m'


def bold(text: str) -> str:
    return f'\033[1m{text}\033[0m'


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
        ib.qualifyContracts(opt)
        ticker = ib.reqMktData(opt, '', False, False)
        ib.sleep(0.5)

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
    except Exception:
        return None


def get_position_live_value(ib, trade) -> Optional[dict]:
    """Get live value for a position."""
    if ib is None or not trade.strikes or not trade.expiration:
        return None

    try:
        import json
        strikes = json.loads(trade.strikes) if isinstance(trade.strikes, str) else trade.strikes
        strike = strikes[0] if isinstance(strikes, list) else float(strikes)
        expiry = trade.expiration.replace('-', '')

        call_data = get_live_option_price(ib, trade.ticker, expiry, strike, 'C')
        put_data = get_live_option_price(ib, trade.ticker, expiry, strike, 'P')

        if call_data and put_data:
            call_mid = None
            put_mid = None

            if call_data['bid'] and call_data['ask']:
                call_mid = (call_data['bid'] + call_data['ask']) / 2
            elif call_data['last']:
                call_mid = call_data['last']

            if put_data['bid'] and put_data['ask']:
                put_mid = (put_data['bid'] + put_data['ask']) / 2
            elif put_data['last']:
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
    except Exception as e:
        pass

    return None


def render_dashboard(logger: TradeLogger, show_all: bool = False, live: bool = False):
    """Render the dashboard."""
    now = datetime.now()

    print(bold("=" * 70))
    print(bold(f"  EARNINGS TRADING DASHBOARD - {now.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(bold("=" * 70))
    print()

    # Get all trades
    all_trades = logger.get_trades()

    if not all_trades:
        print("  No trades recorded yet.")
        print()
        return

    # Categorize trades
    open_trades = [t for t in all_trades if t.status in ('pending', 'submitted', 'filled', 'partial')]
    completed_trades = [t for t in all_trades if t.status in ('completed', 'exited')]

    # === Open Positions ===
    print(bold("  OPEN POSITIONS"))
    print("  " + "-" * 66)

    # Connect to IB if live mode
    ib = None
    if live:
        ib = connect_ib()
        if ib:
            print("  [Connected to IBKR for live prices]")
        print()

    total_unrealized_pnl = 0
    has_live_data = False

    if open_trades:
        print(f"  {'Symbol':<8} {'Earnings':<12} {'Status':<10} {'Entry':<10} {'Current':<10} {'P&L':<12}")
        print("  " + "-" * 66)

        for trade in open_trades:
            status_color = get_status_color(trade.status)

            # Parse entry info
            entry_price = "N/A"
            if trade.entry_fill_price:
                entry_price = format_currency(trade.entry_fill_price)
            elif trade.entry_quoted_mid:
                entry_price = f"${trade.entry_quoted_mid:.2f}"

            # Get live prices if available
            current_price = "N/A"
            pnl_str = "N/A"
            pnl_color = reset_color()

            if live and ib:
                live_data = get_position_live_value(ib, trade)
                if live_data:
                    has_live_data = True
                    current_price = f"${live_data['straddle_mid']:.2f}"

                    if live_data['pnl_pct'] is not None:
                        pnl_pct = live_data['pnl_pct']
                        pnl_color = '\033[92m' if pnl_pct >= 0 else '\033[91m'
                        pnl_str = f"{pnl_pct:+.1f}%"

                        if live_data['unrealized_pnl'] is not None:
                            total_unrealized_pnl += live_data['unrealized_pnl']

            # Add warning indicator for partial fills
            status_display = trade.status
            if trade.status == 'partial':
                status_display = "PARTIAL!"

            print(f"  {trade.ticker:<8} {trade.earnings_date:<12} "
                  f"{status_color}{status_display:<10}{reset_color()} "
                  f"{entry_price:<10} {current_price:<10} {pnl_color}{pnl_str:<12}{reset_color()}")

            # Show partial fill warning with details
            if trade.status == 'partial' and trade.notes:
                print(f"           \033[91mWARNING: {trade.notes}\033[0m")

            # Show order details on second line
            if trade.structure:
                timing = trade.earnings_timing or "?"
                strikes = trade.strikes or "?"
                expiry = trade.expiration or "?"
                print(f"           {trade.structure} | Strike: {strikes} | Exp: {expiry} | {timing}")

            # Show edge and implied move
            if trade.edge_q75 is not None:
                print(f"           Edge: {format_pct(trade.edge_q75)} | "
                      f"Pred Q75: {format_pct(trade.predicted_q75)} | "
                      f"Implied: {format_pct(trade.implied_move)}")
            elif trade.implied_move is not None:
                print(f"           Implied Move: {format_pct(trade.implied_move)}")
            print()

        # Show total unrealized P&L
        if has_live_data and total_unrealized_pnl != 0:
            pnl_color = '\033[92m' if total_unrealized_pnl >= 0 else '\033[91m'
            print("  " + "-" * 66)
            print(f"  {'TOTAL UNREALIZED P&L:':<52} {pnl_color}{format_currency(total_unrealized_pnl)}{reset_color()}")
            print()
    else:
        print("  No open positions")
        print()

    # === Completed Trades ===
    if completed_trades or show_all:
        print(bold("  COMPLETED TRADES"))
        print("  " + "-" * 66)

        if completed_trades:
            total_pnl = 0
            wins = 0
            losses = 0

            print(f"  {'Symbol':<8} {'Date':<12} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}")
            print("  " + "-" * 66)

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

            print("  " + "-" * 66)
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
    print("  " + "-" * 66)

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

    print()

    # === Non-Trades (Rejections) ===
    non_trades = logger.get_non_trades()
    if non_trades:
        print(bold("  RECENT REJECTIONS"))
        print("  " + "-" * 66)
        for nt in non_trades[:5]:
            reason = (nt.rejection_reason or 'Unknown')[:40]
            symbol = nt.ticker or '?'
            print(f"  {symbol:<8} {reason}")
        print()


def get_open_positions(logger: TradeLogger) -> list:
    """Get open positions that can be closed."""
    all_trades = logger.get_trades()
    return [t for t in all_trades if t.status in ('pending', 'submitted', 'filled', 'partial')]


def close_position_interactive(logger: TradeLogger):
    """Interactive position closing."""
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

        # Close each position
        from ib_insync import Option, LimitOrder

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
            # Check for valid price (not NaN, not -1, positive)
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
            ib.sleep(2)

            status = order_trade.orderStatus.status
            print(f"    Order status: {status}")

            if status in ('Submitted', 'Filled', 'PreSubmitted'):
                closed_count = closed_count + 1 if 'closed_count' in dir() else 1

        # Update trade status in DB only if we placed orders
        if 'closed_count' in dir() and closed_count > 0:
            logger.update_trade(trade.trade_id, status='exited', notes='Manually closed via dashboard')
            print()
            print(f"  {trade.ticker}: {closed_count} close order(s) placed. Status updated to 'exited'.")
        else:
            print()
            print(f"  No orders placed for {trade.ticker}. Status unchanged.")

    except ValueError:
        print("  Invalid input.")
    except Exception as e:
        print(f"  Error: {e}")

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
                render_dashboard(logger, show_all=args.all, live=args.live)
                mode = "LIVE" if args.live else "DB only"
                print(f"  [{mode} | Refreshing every {args.interval}s | c=close, r=refresh, q=quit]")

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
        render_dashboard(logger, show_all=args.all, live=args.live)
        disconnect_ib()


if __name__ == '__main__':
    main()
