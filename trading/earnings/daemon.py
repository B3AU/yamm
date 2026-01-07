#!/usr/bin/env python3
"""Phase 0 Trading Daemon

Automated earnings volatility trading with scheduled execution.

Schedule (all times ET):
- 09:25: Connect to IB Gateway, load positions to exit
- 14:45: Exit positions from previous day
- 15:00: Screen upcoming earnings AND place new orders
- 15:50: Final fill check, cancel unfilled orders
- 16:05: Disconnect (after market close)

Exit positions are handled the next trading day at 15:45.

Usage:
    python -m trading.earnings.daemon

Or via systemd:
    systemctl start yamm-trading
"""
from __future__ import annotations

import os
import sys
import signal
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ib_insync import IB

from trading.earnings.screener import (
    fetch_upcoming_earnings,
    screen_all_candidates,
    ScreenedCandidate,
)
from trading.earnings.executor import Phase0Executor, close_position
from trading.earnings.logging import TradeLogger
from trading.earnings.ml_predictor import get_predictor, EarningsPredictor

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Timezone
ET = pytz.timezone('US/Eastern')

# Check if paper trading mode
PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'

# Configuration - use paper or live parameters based on mode
CONFIG = {
    'ib_host': os.getenv('IB_HOST', '127.0.0.1'),
    'ib_port': int(os.getenv('IB_PORT', '4002')),  # Paper trading port
    'ib_client_id': int(os.getenv('IB_CLIENT_ID', '1')),

    'paper_mode': PAPER_MODE,

    # Use paper or live thresholds based on mode
    'spread_threshold': float(os.getenv(
        'PAPER_SPREAD_THRESHOLD' if PAPER_MODE else 'SPREAD_THRESHOLD', '15.0'
    )),
    'edge_threshold': float(os.getenv(
        'PAPER_EDGE_THRESHOLD' if PAPER_MODE else 'EDGE_THRESHOLD', '0.05'
    )),
    'max_daily_trades': int(os.getenv(
        'PAPER_MAX_DAILY_TRADES' if PAPER_MODE else 'MAX_DAILY_TRADES', '5'
    )),

    'max_contracts': int(os.getenv('MAX_CONTRACTS', '1')),
    'limit_aggression': float(os.getenv('LIMIT_AGGRESSION', '0.3')),
    'max_candidates_to_screen': int(os.getenv('MAX_CANDIDATES_TO_SCREEN', '50')),

    'db_path': PROJECT_ROOT / 'data' / 'earnings_trades.db',
    'log_path': PROJECT_ROOT / 'logs' / 'daemon.log',

    'dry_run': os.getenv('DRY_RUN', 'false').lower() == 'true',
}

# Setup logging
def setup_logging():
    log_dir = CONFIG['log_path'].parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG['log_path']),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger('earnings_daemon')

logger = setup_logging()


class TradingDaemon:
    """Main trading daemon."""

    def __init__(self):
        self.ib: Optional[IB] = None
        self.trade_logger = TradeLogger(db_path=CONFIG['db_path'])
        self.executor: Optional[Phase0Executor] = None
        self.scheduler = BackgroundScheduler(timezone=ET)

        # ML predictor
        self.predictor: Optional[EarningsPredictor] = None

        # Daily state
        self.todays_candidates: list[ScreenedCandidate] = []
        self.todays_trades: list[str] = []  # trade_ids
        self.positions_to_exit: list[dict] = []  # Positions from previous day

        # Track connection state
        self.connected = False

    def connect(self):
        """Connect to IB Gateway."""
        if self.ib and self.ib.isConnected():
            logger.info("Already connected to IB")
            return True

        logger.info(f"Connecting to IB Gateway at {CONFIG['ib_host']}:{CONFIG['ib_port']}...")

        try:
            self.ib = IB()
            self.ib.connect(
                CONFIG['ib_host'],
                CONFIG['ib_port'],
                clientId=CONFIG['ib_client_id'],
            )

            # Request LIVE market data (1 = live, 3 = delayed)
            # Must have IBKR market data subscription for live quotes
            self.ib.reqMarketDataType(1)
            logger.info("Requested LIVE market data (type 1)")

            self.executor = Phase0Executor(
                ib=self.ib,
                trade_logger=self.trade_logger,
                max_contracts=CONFIG['max_contracts'],
                limit_aggression=CONFIG['limit_aggression'],
            )

            self.connected = True
            logger.info("Connected to IB Gateway")

            # Log extensive account info
            self._log_account_summary()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
        self.connected = False
        self.ib = None
        self.executor = None

    def ensure_connected(self) -> bool:
        """Ensure IB connection is alive, reconnect if needed."""
        if self.ib and self.ib.isConnected():
            return True

        logger.warning("IB connection lost, reconnecting...")
        return self.connect()

    # ==================== Scheduled Tasks ====================

    def task_morning_connect(self):
        """9:25 AM - Connect before market open."""
        logger.info("=== MORNING CONNECT ===")

        # Reset daily state
        self.todays_candidates = []
        self.todays_trades = []

        # Load ML predictor
        try:
            self.predictor = get_predictor()
            logger.info(f"ML predictor loaded: {len(self.predictor.models)} models")
        except Exception as e:
            logger.error(f"Failed to load ML predictor: {e}")
            self.predictor = None

        # Load positions that need to be exited today
        self._load_positions_to_exit()

        if not self.connect():
            logger.error("Morning connect failed - will retry at screening time")

    def task_screen_candidates(self):
        """3:00 PM - Screen upcoming earnings."""
        logger.info("=== SCREENING CANDIDATES ===")

        if not self.ensure_connected():
            logger.error("Cannot screen - not connected")
            return

        try:
            # Fetch earnings for tomorrow (T+1)
            events = fetch_upcoming_earnings(days_ahead=3)

            # Filter to events happening tomorrow or day after
            tomorrow = date.today() + timedelta(days=1)
            day_after = date.today() + timedelta(days=2)

            # For BMO earnings, we enter day before (today for tomorrow's BMO)
            # For AMC earnings, we enter same day (today for today's AMC)
            relevant_events = []
            for e in events:
                if e.earnings_date == tomorrow and e.timing == 'BMO':
                    relevant_events.append(e)  # Enter today for tomorrow BMO
                elif e.earnings_date == date.today() and e.timing == 'AMC':
                    relevant_events.append(e)  # Enter today for today AMC

            logger.info(f"Found {len(relevant_events)} earnings events to consider")

            if not relevant_events:
                logger.info("No earnings events requiring entry today")
                return

            # Screen candidates
            passed, rejected = screen_all_candidates(
                self.ib,
                relevant_events,
                spread_threshold=CONFIG['spread_threshold'],
                max_candidates=CONFIG['max_candidates_to_screen'],
            )

            logger.info(f"Liquidity screening: {len(passed)} passed, {len(rejected)} rejected")

            # Log rejected candidates (liquidity fails)
            for candidate in rejected:
                self.executor.log_non_trade(candidate)

            # Apply ML edge filter
            if self.predictor and passed:
                edge_threshold = CONFIG['edge_threshold']
                logger.info(f"Applying ML edge filter (threshold={edge_threshold:.0%})...")

                ml_passed = []
                for candidate in passed:
                    prediction = self.predictor.predict(
                        symbol=candidate.symbol,
                        earnings_date=candidate.earnings_date,
                        timing=candidate.timing,
                    )

                    if prediction is None:
                        # Get detailed reason why prediction failed
                        status = self.predictor.get_prediction_status(
                            candidate.symbol, candidate.earnings_date
                        )
                        logger.info(f"  {candidate.symbol}: No ML prediction - {status}")
                        candidate.rejection_reason = status
                        self.executor.log_non_trade(candidate)
                        continue

                    # Add ML fields to candidate
                    candidate.pred_q75 = prediction.pred_q75
                    candidate.hist_move_mean = prediction.hist_move_mean
                    candidate.edge_q75 = prediction.edge_q75

                    if prediction.edge_q75 >= edge_threshold:
                        candidate.passes_edge = True
                        ml_passed.append(candidate)
                        logger.info(
                            f"  {candidate.symbol}: PASS edge={prediction.edge_q75:.1%} "
                            f"(pred_q75={prediction.pred_q75:.1%}, hist={prediction.hist_move_mean:.1%})"
                        )
                    else:
                        candidate.rejection_reason = f"Edge {prediction.edge_q75:.1%} < {edge_threshold:.0%}"
                        self.executor.log_non_trade(candidate)
                        logger.info(
                            f"  {candidate.symbol}: FAIL edge={prediction.edge_q75:.1%} < {edge_threshold:.0%}"
                        )

                passed = ml_passed
                logger.info(f"ML edge filter: {len(passed)} passed")
            else:
                logger.warning("ML predictor not available - using liquidity-only screening")

            # Store passed candidates for order placement
            self.todays_candidates = passed[:CONFIG['max_daily_trades']]

            if self.todays_candidates:
                logger.info("Final candidates for today:")
                for c in self.todays_candidates:
                    edge_str = f", edge {c.edge_q75:.1%}" if c.edge_q75 is not None else ""
                    logger.info(f"  {c.symbol}: earnings {c.earnings_date} ({c.timing}), "
                              f"spread {c.spread_pct:.1f}%, implied move {c.implied_move_pct:.1f}%{edge_str}")

                # Place orders immediately after screening
                self._place_orders_for_candidates()

        except Exception as e:
            logger.exception(f"Screening failed: {e}")

    def _place_orders_for_candidates(self):
        """Place orders on screened candidates (called from screening task)."""
        logger.info("=== PLACING ORDERS ===")

        if not self.ensure_connected():
            logger.error("Cannot place orders - not connected")
            return

        if not self.todays_candidates:
            logger.info("No candidates to trade today")
            return

        if CONFIG['dry_run']:
            logger.info("DRY RUN - would place orders on:")
            for c in self.todays_candidates:
                logger.info(f"  {c.symbol}: {CONFIG['max_contracts']} contracts")
            return

        try:
            for candidate in self.todays_candidates:
                logger.info(f"Placing order for {candidate.symbol}...")

                order_pair = self.executor.place_straddle(candidate)

                if order_pair:
                    self.todays_trades.append(order_pair.trade_id)
                    logger.info(f"  Order placed: {order_pair.trade_id}")
                else:
                    logger.error(f"  Order failed for {candidate.symbol}")

                # Small delay between orders
                self.ib.sleep(1)

            logger.info(f"Placed {len(self.todays_trades)} orders")

        except Exception as e:
            logger.exception(f"Order placement failed: {e}")

    def task_place_orders(self):
        """Manual trigger for placing orders (if needed separately)."""
        self._place_orders_for_candidates()

    def task_check_fills(self):
        """3:50 PM - Final fill check before close.

        Also cancels unfilled orders to prevent orphan positions.
        """
        logger.info("=== CHECKING FILLS (NEAR CLOSE) ===")

        if not self.ensure_connected():
            logger.error("Cannot check fills - not connected")
            return

        try:
            # First, check current fill status
            filled = self.executor.check_fills()

            logger.info(f"Fill status: {len(filled)} filled, "
                       f"{self.executor.get_active_count()} still pending")

            for order in filled:
                logger.info(f"  {order.symbol}: Call @ ${order.call_fill_price:.2f}, "
                          f"Put @ ${order.put_fill_price:.2f}")

            # Check for partial fills - these are orphan risks
            partials = self.executor.get_partial_fills()
            if partials:
                logger.warning(f"=== PARTIAL FILLS DETECTED: {len(partials)} ===")
                for order in partials:
                    call_qty = order.call_trade.orderStatus.filled if order.call_trade else 0
                    put_qty = order.put_trade.orderStatus.filled if order.put_trade else 0
                    call_total = order.call_trade.order.totalQuantity if order.call_trade else 0
                    put_total = order.put_trade.order.totalQuantity if order.put_trade else 0
                    logger.warning(
                        f"  {order.symbol}: Call {int(call_qty)}/{int(call_total)}, "
                        f"Put {int(put_qty)}/{int(put_total)}"
                    )

            # Cancel all unfilled orders near close to prevent orphans
            pending_count = self.executor.get_active_count()
            if pending_count > 0 or partials:
                logger.warning("=== CANCELLING UNFILLED ORDERS (NEAR CLOSE) ===")
                for trade_id, order_pair in list(self.executor.active_orders.items()):
                    if order_pair.status in ('pending', 'partial'):
                        result = self.executor.cancel_unfilled_orders(trade_id)
                        if result.get('cancelled'):
                            logger.info(f"  {order_pair.symbol}: Cancelled unfilled orders")

            # Log metrics
            metrics = self.trade_logger.get_execution_metrics()
            logger.info(f"Execution metrics: fill_rate={metrics.fill_rate*100:.1f}%, "
                       f"avg_slippage={metrics.avg_slippage_bps:.1f}bps")

        except Exception as e:
            logger.exception(f"Fill check failed: {e}")

    def task_exit_positions(self):
        """3:45 PM - Exit positions from previous day."""
        logger.info("=== EXITING POSITIONS ===")

        if not self.ensure_connected():
            logger.error("Cannot exit positions - not connected")
            return

        if not self.positions_to_exit:
            logger.info("No positions to exit today")
            return

        if CONFIG['dry_run']:
            logger.info("DRY RUN - would exit positions:")
            for pos in self.positions_to_exit:
                logger.info(f"  {pos['symbol']}")
            return

        try:
            for pos in self.positions_to_exit:
                logger.info(f"Exiting {pos['symbol']}...")

                success = close_position(
                    self.ib,
                    self.trade_logger,
                    trade_id=pos['trade_id'],
                    symbol=pos['symbol'],
                    expiry=pos['expiry'],
                    strike=pos['strike'],
                    contracts=pos['contracts'],
                    limit_aggression=CONFIG['limit_aggression'],
                )

                if success:
                    logger.info(f"  Exit order placed for {pos['symbol']}")
                else:
                    logger.error(f"  Exit failed for {pos['symbol']}")

                self.ib.sleep(1)

        except Exception as e:
            logger.exception(f"Position exit failed: {e}")

    def task_evening_disconnect(self):
        """4:05 PM - Disconnect after market close."""
        logger.info("=== EVENING DISCONNECT ===")

        # Final fill check
        if self.executor:
            self.executor.check_fills()

        # Log daily summary
        stats = self.trade_logger.get_summary_stats()
        logger.info(f"Daily summary: {stats['total_trades']} total trades, "
                   f"{stats['completed_trades']} completed, P&L: ${stats['total_pnl']:.2f}")

        self.disconnect()

    def _log_account_summary(self):
        """Log extensive account information for monitoring."""
        logger.info("=" * 50)
        logger.info("ACCOUNT SUMMARY")
        logger.info("=" * 50)

        try:
            # Account ID first
            accounts = self.ib.managedAccounts()
            logger.info(f"Account(s): {accounts}")

            # Account values - collect all, prefer USD but take any
            account_values = {}
            all_values = self.ib.accountValues()
            logger.info(f"Total account values received: {len(all_values)}")

            for av in all_values:
                # Store values, prefer BASE currency (EUR for your account), then USD
                if av.tag not in account_values or av.currency in ('BASE', 'EUR', 'USD'):
                    account_values[av.tag] = (av.value, av.currency or 'BASE')

            # Key metrics
            key_tags = [
                'NetLiquidation', 'TotalCashValue', 'BuyingPower',
                'AvailableFunds', 'ExcessLiquidity', 'GrossPositionValue',
                'MaintMarginReq', 'InitMarginReq', 'UnrealizedPnL', 'RealizedPnL',
                'CashBalance', 'AccruedCash', 'EquityWithLoanValue'
            ]

            logger.info("\nKEY METRICS:")
            for tag in key_tags:
                if tag in account_values:
                    val, currency = account_values[tag]
                    try:
                        val_float = float(val)
                        logger.info(f"  {tag}: ${val_float:,.2f} {currency or ''}")
                    except (ValueError, TypeError):
                        logger.info(f"  {tag}: {val} {currency or ''}")

            # Current positions
            positions = self.ib.positions()
            if positions:
                logger.info(f"\nOPEN POSITIONS ({len(positions)}):")
                for pos in positions:
                    logger.info(f"  {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f}")
            else:
                logger.info("\nNo open positions")

            # Open orders
            orders = self.ib.openOrders()
            if orders:
                logger.info(f"\nOPEN ORDERS ({len(orders)}):")
                for order in orders:
                    logger.info(f"  {order.orderId}: {order.action} {order.totalQuantity}")
            else:
                logger.info("\nNo open orders")

            # Test live market data with SPY
            logger.info("\nTESTING LIVE MARKET DATA...")
            self._test_live_market_data()

        except Exception as e:
            logger.exception(f"Error logging account summary: {e}")

        logger.info("=" * 50)

    def _test_live_market_data(self):
        """Test that we're getting live (not delayed) market data."""
        from ib_insync import Stock

        try:
            spy = Stock('SPY', 'SMART', 'USD')
            self.ib.qualifyContracts(spy)
            logger.info(f"  SPY contract qualified: conId={spy.conId}")

            # Request market data type info
            # 1 = live, 2 = frozen, 3 = delayed, 4 = delayed-frozen
            logger.info("  Requesting market data (type 1 = live)...")

            ticker = self.ib.reqMktData(spy, '', False, False)

            # Wait longer and check incrementally
            for i in range(5):
                self.ib.sleep(1)
                if ticker.bid and ticker.bid == ticker.bid:  # has valid bid
                    break
                logger.info(f"  Waiting for data... ({i+1}/5)")

            # Get all available data
            bid = ticker.bid
            ask = ticker.ask
            last = ticker.last
            close = ticker.close
            high = ticker.high
            low = ticker.low
            volume = ticker.volume

            # Check market data type returned
            # ticker.marketDataType: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
            mkt_data_type = getattr(ticker, 'marketDataType', None)

            self.ib.cancelMktData(spy)

            logger.info(f"  SPY Quote: bid=${bid}, ask=${ask}, last=${last}, close=${close}")
            logger.info(f"  SPY OHLV: high=${high}, low=${low}, volume={volume}")
            logger.info(f"  Market data type: {mkt_data_type} (1=live, 2=frozen, 3=delayed, 4=delayed-frozen)")

            # Check if we got valid data
            has_bid = bid is not None and bid == bid and bid > 0  # nan check
            has_ask = ask is not None and ask == ask and ask > 0
            has_last = last is not None and last == last and last > 0

            if mkt_data_type == 3:
                logger.warning("  ✗ DELAYED DATA DETECTED (type 3)")
                logger.warning("    You need to subscribe to US Equity Real-Time Data in IBKR Account Management")
                logger.warning("    Go to: Account Management > Settings > Market Data Subscriptions")
            elif mkt_data_type == 1:
                logger.info("  ✓ LIVE DATA TYPE CONFIRMED (type 1)")
            elif mkt_data_type == 2:
                logger.info("  Market is closed - using frozen data (type 2)")
            elif mkt_data_type == 4:
                logger.warning("  ✗ DELAYED-FROZEN DATA (type 4) - need subscription")

            # Check for competing session error (10197)
            # This happens when another session is using the market data lines
            if not has_bid and not has_ask and not has_last:
                logger.warning("  NOTE: If you see Error 10197 above, it means another session")
                logger.warning("    (TWS, another IB Gateway, or API client) is using your market data.")
                logger.warning("    Close other sessions or increase market data lines in IBKR settings.")

            if has_bid and has_ask:
                spread = ask - bid
                spread_pct = spread / ((bid + ask) / 2) * 100
                logger.info(f"  SPY Spread: ${spread:.2f} ({spread_pct:.3f}%)")

                if spread_pct < 0.1:
                    logger.info("  ✓ TIGHT SPREAD - quotes look live")
                elif spread_pct < 1.0:
                    logger.info("  Spread OK for market conditions")
                else:
                    logger.warning("  ? Spread wider than expected")
            elif has_last:
                logger.info("  Got last price but no bid/ask - market may be closed")
            else:
                logger.warning("  ✗ NO QUOTE DATA RECEIVED")
                logger.warning("    Possible causes:")
                logger.warning("    - No market data subscription for US equities")
                logger.warning("    - IB Gateway not connected properly")
                logger.warning("    - Paper account needs delayed data enabled")

        except Exception as e:
            logger.exception(f"  Market data test failed: {e}")

    def _log_upcoming_earnings(self):
        """Log upcoming earnings events for visibility."""
        logger.info("=" * 50)
        logger.info("UPCOMING EARNINGS (Next 7 Days)")
        logger.info("=" * 50)

        try:
            events = fetch_upcoming_earnings(days_ahead=7)

            if not events:
                logger.info("No upcoming earnings found")
                return

            # Group by date
            from collections import defaultdict
            by_date = defaultdict(list)
            for e in events:
                by_date[e.earnings_date].append(e)

            # Sort dates
            today = date.today()
            tomorrow = today + timedelta(days=1)

            for earn_date in sorted(by_date.keys()):
                events_on_date = by_date[earn_date]
                bmo = [e for e in events_on_date if e.timing == 'BMO']
                amc = [e for e in events_on_date if e.timing == 'AMC']
                unknown = [e for e in events_on_date if e.timing == 'unknown']

                # Highlight today/tomorrow
                if earn_date == today:
                    date_label = f"{earn_date} (TODAY)"
                elif earn_date == tomorrow:
                    date_label = f"{earn_date} (TOMORROW)"
                else:
                    date_label = str(earn_date)

                logger.info(f"\n{date_label}: {len(events_on_date)} earnings")

                if bmo:
                    bmo_symbols = [e.symbol for e in bmo[:15]]
                    more = f" +{len(bmo)-15} more" if len(bmo) > 15 else ""
                    logger.info(f"  BMO ({len(bmo)}): {', '.join(bmo_symbols)}{more}")

                if amc:
                    amc_symbols = [e.symbol for e in amc[:15]]
                    more = f" +{len(amc)-15} more" if len(amc) > 15 else ""
                    logger.info(f"  AMC ({len(amc)}): {', '.join(amc_symbols)}{more}")

                if unknown:
                    unk_symbols = [e.symbol for e in unknown[:10]]
                    more = f" +{len(unknown)-10} more" if len(unknown) > 10 else ""
                    logger.info(f"  Unknown ({len(unknown)}): {', '.join(unk_symbols)}{more}")

            # Summary of what we'd trade today
            logger.info("\n" + "-" * 50)
            logger.info("TRADEABLE TODAY:")

            # BMO tomorrow = enter today
            tomorrows_bmo = [e for e in events if e.earnings_date == tomorrow and e.timing == 'BMO']
            # AMC today = enter today
            todays_amc = [e for e in events if e.earnings_date == today and e.timing == 'AMC']

            if tomorrows_bmo:
                symbols = [e.symbol for e in tomorrows_bmo[:20]]
                more = f" +{len(tomorrows_bmo)-20} more" if len(tomorrows_bmo) > 20 else ""
                logger.info(f"  Tomorrow BMO (enter today): {len(tomorrows_bmo)} - {', '.join(symbols)}{more}")
            else:
                logger.info("  Tomorrow BMO: None")

            if todays_amc:
                symbols = [e.symbol for e in todays_amc[:20]]
                more = f" +{len(todays_amc)-20} more" if len(todays_amc) > 20 else ""
                logger.info(f"  Today AMC (enter today): {len(todays_amc)} - {', '.join(symbols)}{more}")
            else:
                logger.info("  Today AMC: None")

            total_tradeable = len(tomorrows_bmo) + len(todays_amc)
            logger.info(f"  TOTAL CANDIDATES TO SCREEN: {total_tradeable}")

        except Exception as e:
            logger.exception(f"Error fetching upcoming earnings: {e}")

        logger.info("=" * 50)

    def _load_positions_to_exit(self):
        """Load positions that need to be exited today."""
        # Get trades that are 'filled' status and were entered yesterday
        # They should be exited today at T+1 close
        trades = self.trade_logger.get_trades(status='filled')

        self.positions_to_exit = []
        yesterday = date.today() - timedelta(days=1)

        for trade in trades:
            # Parse entry date
            try:
                entry_date = datetime.fromisoformat(trade.entry_datetime).date()
            except:
                continue

            # Check if this should be exited today
            # (entered yesterday or earlier, not yet exited)
            if entry_date <= yesterday and not trade.exit_datetime:
                self.positions_to_exit.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.ticker,
                    'expiry': trade.expiration,
                    'strike': float(trade.strikes.strip('[]')),
                    'contracts': trade.contracts,
                })

        if self.positions_to_exit:
            logger.info(f"Loaded {len(self.positions_to_exit)} positions to exit today")

    # ==================== Scheduler Setup ====================

    def setup_schedule(self):
        """Configure the job schedule."""

        # Market hours schedule (Monday-Friday)
        # All times in ET

        # Morning connect
        self.scheduler.add_job(
            self.task_morning_connect,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=25),
            id='morning_connect',
            name='Morning Connect',
        )

        # Exit positions BEFORE placing new ones
        self.scheduler.add_job(
            self.task_exit_positions,
            CronTrigger(day_of_week='mon-fri', hour=14, minute=45),
            id='exit_positions',
            name='Exit Positions',
        )

        # Screen candidates AND place orders (after exits are done)
        self.scheduler.add_job(
            self.task_screen_candidates,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=0),
            id='screen_and_place',
            name='Screen & Place Orders',
        )

        # Final fill check
        self.scheduler.add_job(
            self.task_check_fills,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=50),
            id='check_fills',
            name='Check Fills',
        )

        # Evening disconnect
        self.scheduler.add_job(
            self.task_evening_disconnect,
            CronTrigger(day_of_week='mon-fri', hour=16, minute=5),
            id='evening_disconnect',
            name='Evening Disconnect',
        )

        logger.info("Schedule configured:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  {job.name}: {job.trigger}")

    def run(self):
        """Start the daemon."""
        logger.info("=" * 60)
        logger.info("EARNINGS TRADING DAEMON STARTING")
        logger.info("=" * 60)

        mode_str = "PAPER" if CONFIG['paper_mode'] else "LIVE"
        logger.info(f"Mode: {mode_str}")
        logger.info(f"Config: spread_threshold={CONFIG['spread_threshold']}%, "
                   f"edge_threshold={CONFIG['edge_threshold']:.0%}, "
                   f"max_contracts={CONFIG['max_contracts']}, "
                   f"max_daily_trades={CONFIG['max_daily_trades']}, "
                   f"dry_run={CONFIG['dry_run']}")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Setup schedule
        self.setup_schedule()

        # Always connect immediately on start to verify connection and log account
        logger.info("Connecting to IB Gateway on startup...")
        if self.connect():
            logger.info("Startup connection successful")

            # Load ML predictor
            try:
                self.predictor = get_predictor()
                logger.info(f"ML predictor loaded: {len(self.predictor.models)} models, {len(self.predictor.feature_cols)} features")
            except Exception as e:
                logger.error(f"Failed to load ML predictor: {e}")
                self.predictor = None

            # Show upcoming earnings
            self._log_upcoming_earnings()

            # Recover any pending orders from previous run
            if self.executor:
                recovered = self.executor.recover_orders()
                if recovered > 0:
                    logger.info(f"Recovered {recovered} pending orders from database")
        else:
            logger.error("STARTUP CONNECTION FAILED - check IB Gateway is running")
            logger.error("Daemon will continue and retry at scheduled times")

        # Check if we should run screening immediately (missed the scheduled time)
        self._run_screening_if_needed()

        # Start scheduler (background - doesn't block)
        logger.info("Starting scheduler...")
        self.scheduler.start()

        # Log next scheduled jobs
        for job in self.scheduler.get_jobs():
            logger.info(f"  {job.name}: next run at {job.next_run_time}")

        # Keep main thread alive with IB event loop
        logger.info("Entering main loop (Ctrl+C to stop)...")
        try:
            while True:
                if self.ib and self.ib.isConnected():
                    self.ib.sleep(1)  # Process IB events
                else:
                    time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Daemon stopped")
        finally:
            self.scheduler.shutdown()
            self.disconnect()

    def _run_screening_if_needed(self):
        """Run screening immediately if we're in the trading window and missed the scheduled time."""
        if not self.connected:
            logger.info("Not connected - skipping startup screening")
            return

        now = datetime.now(ET)
        screen_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # If we're between 15:00 and 16:00 ET on a weekday
        if now.weekday() < 5 and screen_time <= now < market_close:
            # Run screening (which now also places orders immediately)
            logger.info("=" * 60)
            logger.info("STARTUP: Running screening (missed 15:00 scheduled time)")
            logger.info("=" * 60)
            self.task_screen_candidates()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.scheduler.shutdown(wait=False)
        self.disconnect()
        sys.exit(0)


def main():
    """Entry point."""
    daemon = TradingDaemon()
    daemon.run()


if __name__ == '__main__':
    main()
