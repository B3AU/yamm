#!/usr/bin/env python3
"""Phase 0 Trading Daemon (Async)

Automated earnings volatility trading with scheduled execution.

Schedule (all times ET):
- 09:25: Connect to IB Gateway, load positions to exit
- 14:45: Exit positions from previous day
- 14:00: Screen upcoming earnings AND place new orders
- 15:50: Final fill check, cancel unfilled orders
- 16:05: Disconnect (after market close)

Exit positions are handled the next trading day at 14:45.

Usage:
    python -m trading.earnings.daemon
"""
from __future__ import annotations

import os
import sys
import signal
import logging
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use nest_asyncio to allow nested loops if needed (e.g. ib.run inside async)
# But we aim for pure async.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from ib_insync import IB

from trading.earnings.screener import (
    fetch_upcoming_earnings,
    screen_all_candidates,
    ScreenedCandidate,
)
from trading.earnings.executor import (
    Phase0Executor, close_position, ExitComboOrder, check_exit_fills
)
from trading.earnings.logging import TradeLogger
from trading.earnings.ml_predictor import get_predictor, EarningsPredictor
from trading.earnings.counterfactual import backfill_counterfactuals

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Timezone
ET = pytz.timezone('US/Eastern')

# Check if paper trading mode
PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'

# Configuration
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
    # Silence ib_insync noise
    logging.getLogger('ib_insync').setLevel(logging.WARNING)
    return logging.getLogger('earnings_daemon')

logger = setup_logging()


class TradingDaemon:
    """Main trading daemon (Async)."""

    def __init__(self):
        self.ib = IB()
        self.trade_logger = TradeLogger(db_path=CONFIG['db_path'])

        # Async scheduler with ET timezone
        self.scheduler = AsyncIOScheduler(timezone=ET)

        self.executor: Optional[Phase0Executor] = None

        # ML predictor
        self.predictor: Optional[EarningsPredictor] = None

        # Daily state
        self.todays_candidates: list[ScreenedCandidate] = []
        self.todays_trades: list[str] = []  # trade_ids
        self.positions_to_exit: list[dict] = []  # Positions from previous day
        self.active_exit_orders: dict[str, ExitComboOrder] = {}  # Exit orders being tracked

        # Track connection state
        self.connected = False

    async def connect_async(self):
        """Connect to IB Gateway asynchronously."""
        if self.ib.isConnected():
            logger.info("Already connected to IB")
            return True

        logger.info(f"Connecting to IB Gateway at {CONFIG['ib_host']}:{CONFIG['ib_port']}...")

        try:
            # Use connectAsync
            await self.ib.connectAsync(
                CONFIG['ib_host'],
                CONFIG['ib_port'],
                clientId=CONFIG['ib_client_id'],
            )

            # Request LIVE market data (1 = live, 3 = delayed)
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

            # Log account info (runs async methods internally)
            await self._log_account_summary()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
        self.connected = False
        self.executor = None

    async def ensure_connected(self) -> bool:
        """Ensure IB connection is alive, reconnect if needed."""
        if self.ib.isConnected():
            return True

        logger.warning("IB connection lost, reconnecting with linear backoff...")

        # Simple linear backoff: 3 attempts with 2s delay
        for attempt in range(1, 4):
            logger.info(f"Reconnection attempt {attempt}/3...")
            if await self.connect_async():
                return True
            await asyncio.sleep(2)

        logger.error("Failed to reconnect after 3 attempts")
        return False

    # ==================== Scheduled Tasks (Async) ====================

    async def task_morning_connect(self):
        """9:25 AM ET - Connect before market open."""
        logger.info("=== MORNING CONNECT ===")

        try:
            # Reset daily state
            self.todays_candidates = []
            self.todays_trades = []

            # Cleanup stale orders
            self._cleanup_stale_orders()

            # Load any trades already placed today (to prevent duplicates on restart)
            self._load_todays_activity()

            # Load ML predictor
            try:
                self.predictor = get_predictor()
                logger.info(f"ML predictor loaded: {len(self.predictor.models)} models")
            except Exception as e:
                logger.error(f"Failed to load ML predictor: {e}")
                self.predictor = None

            # Load positions that need to be exited today
            self._load_positions_to_exit()

            # Recover active exit orders
            if self.executor:
                self._recover_exit_orders()

            await self.connect_async()

        except Exception as e:
            logger.exception(f"Morning connect failed: {e}")

    async def task_screen_candidates(self):
        """2:00 PM ET - Screen upcoming earnings."""
        logger.info("=== SCREENING CANDIDATES ===")

        if not await self.ensure_connected():
            logger.error("Cannot screen - not connected")
            return

        try:
            # Check if we already traded today (duplicate check)
            self._load_todays_activity()
            already_placed = len(self.todays_trades)
            remaining_slots = max(0, CONFIG['max_daily_trades'] - already_placed)

            if remaining_slots <= 0:
                logger.info(f"Daily trade limit reached ({already_placed} placed). Skipping screening.")
                return

            # Fetch earnings for tomorrow (T+1)
            # Nasdaq API call is sync, run in executor to not block loop
            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, lambda: fetch_upcoming_earnings(days_ahead=3))

            # Filter to events happening tomorrow or day after
            tomorrow = date.today() + timedelta(days=1)

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

            # Screen candidates (Async)
            passed, rejected = await screen_all_candidates(
                self.ib,
                relevant_events,
                spread_threshold=CONFIG['spread_threshold'],
                max_candidates=CONFIG['max_candidates_to_screen'],
            )

            logger.info(f"Liquidity screening: {len(passed)} passed, {len(rejected)} rejected")

            # Log rejected candidates
            for candidate in rejected:
                self.executor.log_non_trade(candidate)

            # Apply ML edge filter
            if not self.predictor:
                # Try reload
                try:
                    self.predictor = get_predictor()
                except Exception as e:
                    logger.error(f"Failed to load ML predictor: {e}")

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
                        status = self.predictor.get_prediction_status(
                            candidate.symbol, candidate.earnings_date
                        )
                        logger.info(f"  {candidate.symbol}: No ML prediction - {status}")
                        candidate.rejection_reason = status
                        self.executor.log_non_trade(candidate)
                        continue

                    # Add ML fields
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

            # Limits again (in case changed)
            if already_placed > 0:
                logger.info(f"Already placed {already_placed} orders today, {remaining_slots} slots remaining")

            self.todays_candidates = passed[:remaining_slots]

            if self.todays_candidates:
                logger.info("Final candidates for today:")
                for c in self.todays_candidates:
                    edge_str = f", edge {c.edge_q75:.1%}" if c.edge_q75 is not None else ""
                    logger.info(f"  {c.symbol}: earnings {c.earnings_date} ({c.timing}), "
                              f"spread {c.spread_pct:.1f}%, implied move {c.implied_move_pct:.1f}%{edge_str}")

                # Place orders (Async)
                await self._place_orders_for_candidates()

        except Exception as e:
            logger.exception(f"Screening failed: {e}")

    async def _place_orders_for_candidates(self):
        """Place orders on screened candidates (Async)."""
        logger.info("=== PLACING ORDERS ===")

        if not await self.ensure_connected():
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
                # Double check we haven't already traded this symbol today
                # (Paranoid check for duplicate trades)
                # Parse ticker from trade_id to avoid substring matches (e.g. GO vs GOOG)
                if any(t.split('_')[0] == candidate.symbol for t in self.todays_trades):
                    logger.warning(f"Skipping {candidate.symbol} - already traded today")
                    continue

                logger.info(f"Placing order for {candidate.symbol}...")

                # Use async place_straddle
                order_pair = await self.executor.place_straddle(candidate)

                if order_pair:
                    self.todays_trades.append(order_pair.trade_id)
                    logger.info(f"  Order placed: {order_pair.trade_id}")
                else:
                    logger.error(f"  Order failed for {candidate.symbol}")

                await asyncio.sleep(1)

            logger.info(f"Placed {len(self.todays_trades)} orders")

        except Exception as e:
            logger.exception(f"Order placement failed: {e}")

    async def task_cancel_unfilled(self):
        """3:50 PM ET - Cancel unfilled orders."""
        logger.info("=== CANCELLING UNFILLED ORDERS ===")

        if not await self.ensure_connected():
            return

        if not self.executor:
            return

        try:
            active_count = self.executor.get_active_count()
            if active_count == 0:
                logger.info("No active orders to cancel")
                return

            logger.info(f"Checking {active_count} active orders for cancellation...")

            for trade_id, combo_order in list(self.executor.active_orders.items()):
                # Handle partial fills
                if combo_order.status == 'partial':
                    logger.info(f"  {combo_order.symbol}: Order is partially filled, keeping open until close.")
                    continue

                if combo_order.status == 'pending':
                    result = self.executor.cancel_unfilled_orders(trade_id)
                    if result.get('cancelled'):
                        logger.info(f"  {combo_order.symbol}: Cancelled unfilled order")

        except Exception as e:
            logger.exception(f"Cancel unfilled failed: {e}")

    async def task_check_fills(self):
        """3:50 PM ET - Fill check & Cancel."""
        logger.info("=== CHECKING FILLS (NEAR CLOSE) ===")

        if not await self.ensure_connected():
            return

        try:
            # Sync check_fills is safe (checks internal state)
            filled = self.executor.check_fills()

            if filled:
                logger.info(f"Found {len(filled)} late fills, scheduling markouts...")
                for order in filled:
                    self._schedule_markouts(order)

            logger.info(f"Fill status (active): {self.executor.get_active_count()} pending")

            # Cancel unfilled
            pending_count = self.executor.get_active_count()
            if pending_count > 0:
                logger.warning("=== CANCELLING UNFILLED ORDERS (NEAR CLOSE) ===")
                for trade_id, combo_order in list(self.executor.active_orders.items()):
                    if combo_order.status == 'pending':
                        result = self.executor.cancel_unfilled_orders(trade_id)
                        if result.get('cancelled'):
                            logger.info(f"  {combo_order.symbol}: Cancelled unfilled order")

            # Check exit order fills (Sync check)
            if self.active_exit_orders:
                logger.info(f"Checking {len(self.active_exit_orders)} exit orders...")
                exit_filled = check_exit_fills(self.active_exit_orders, self.trade_logger)
                logger.info(f"Exit fills: {len(exit_filled)} completed")

                for exit_pair in exit_filled:
                    self.active_exit_orders.pop(exit_pair.trade_id, None)

            # Log metrics
            metrics = self.trade_logger.get_execution_metrics()
            logger.info(f"Execution metrics: fill_rate={metrics.fill_rate*100:.1f}%, "
                       f"avg_slippage={metrics.avg_slippage_bps:.1f}bps")

        except Exception as e:
            logger.exception(f"Fill check failed: {e}")

    async def task_monitor_fills(self):
        """Monitor for new fills and schedule markouts (runs every min)."""
        if not self.ib.isConnected():
            return

        if not self.executor:
            return

        try:
            filled = self.executor.check_fills()
            for order in filled:
                logger.info(f"New fill detected for {order.symbol}, scheduling markouts...")
                self._schedule_markouts(order)
        except Exception as e:
            logger.error(f"Monitor fills failed: {e}")

    def _schedule_markouts(self, order):
        """Schedule 1m, 5m, 30m markout checks for a filled order."""
        for delay_m in [1, 5, 30]:
            run_time = datetime.now(ET) + timedelta(minutes=delay_m)

            self.scheduler.add_job(
                self.task_record_markout,
                'date',
                run_date=run_time,
                args=[order.trade_id, order.symbol, order.expiry, order.strike, delay_m],
                name=f"Markout {delay_m}m {order.symbol}"
            )

    async def task_record_markout(self, trade_id: str, symbol: str, expiry: str, strike: float, delay_min: int):
        """Record post-fill markout price (Async)."""
        logger.info(f"Recording {delay_min}m markout for {symbol}...")

        if not await self.ensure_connected():
            return

        price = await self._fetch_straddle_price(symbol, expiry, strike)

        if price:
            field_name = f"markout_{delay_min}min"
            self.trade_logger.update_trade(trade_id, **{field_name: price})
            logger.info(f"  {symbol}: {delay_min}m markout = ${price:.2f}")
        else:
            logger.warning(f"  {symbol}: Failed to fetch {delay_min}m markout price")

    async def _fetch_straddle_price(self, symbol: str, expiry: str, strike: float) -> Optional[float]:
        """Fetch current mid price for a straddle (Async)."""
        try:
            from ib_insync import Option

            ib_expiry = expiry.replace('-', '')
            call = Option(symbol, ib_expiry, strike, 'C', 'SMART')
            put = Option(symbol, ib_expiry, strike, 'P', 'SMART')

            await self.ib.qualifyContractsAsync(call, put)

            call_ticker = self.ib.reqMktData(call, '', False, False)
            put_ticker = self.ib.reqMktData(put, '', False, False)

            # Wait for data
            for _ in range(20):
                if (call_ticker.bid > 0 and call_ticker.ask > 0 and
                    put_ticker.bid > 0 and put_ticker.ask > 0):
                    break
                await asyncio.sleep(0.1)

            def get_mid(t):
                if t.bid > 0 and t.ask > 0:
                    return (t.bid + t.ask) / 2
                return None

            call_mid = get_mid(call_ticker)
            put_mid = get_mid(put_ticker)

            self.ib.cancelMktData(call)
            self.ib.cancelMktData(put)

            if call_mid is not None and put_mid is not None:
                return call_mid + put_mid

            return None

        except Exception as e:
            logger.error(f"Error fetching straddle price for {symbol}: {e}")
            return None

    async def task_improve_prices(self, aggression: float = 0.5):
        """Walk up limit prices (Async)."""
        logger.info(f"=== PRICE IMPROVEMENT (aggression={aggression:.0%}) ===")

        if not await self.ensure_connected():
            return

        if not self.executor:
            return

        try:
            pending_count = self.executor.get_active_count()
            if pending_count == 0:
                logger.info("No pending orders to improve")
                return

            logger.info(f"Checking {pending_count} pending orders...")

            improved = 0
            for trade_id, combo_order in list(self.executor.active_orders.items()):
                if combo_order.status != 'pending' or not combo_order.trade:
                    continue

                symbol = combo_order.symbol
                current_limit = combo_order.trade.order.lmtPrice

                try:
                    from ib_insync import Option
                    import json

                    trade_log = self.trade_logger.get_trade(trade_id)
                    if not trade_log or not trade_log.strikes:
                        continue

                    strikes = json.loads(trade_log.strikes)
                    strike = strikes[0] if strikes else None
                    expiry = trade_log.expiration.replace('-', '') if trade_log.expiration else None

                    if not strike or not expiry:
                        continue

                    call = Option(symbol, expiry, strike, 'C', 'SMART')
                    put = Option(symbol, expiry, strike, 'P', 'SMART')
                    await self.ib.qualifyContractsAsync(call, put)

                    call_ticker = self.ib.reqMktData(call, '', False, False)
                    put_ticker = self.ib.reqMktData(put, '', False, False)

                    for _ in range(10):
                        if call_ticker.bid > 0 and put_ticker.bid > 0:
                            break
                        await asyncio.sleep(0.1)

                    def valid(p):
                        return p is not None and p == p and p > 0

                    call_mid = (call_ticker.bid + call_ticker.ask) / 2 if valid(call_ticker.bid) and valid(call_ticker.ask) else None
                    put_mid = (put_ticker.bid + put_ticker.ask) / 2 if valid(put_ticker.bid) and valid(put_ticker.ask) else None

                    self.ib.cancelMktData(call)
                    self.ib.cancelMktData(put)

                    if not call_mid or not put_mid:
                        continue

                    straddle_mid = call_mid + put_mid
                    call_spread = (call_ticker.ask - call_ticker.bid) if valid(call_ticker.bid) else 0
                    put_spread = (put_ticker.ask - put_ticker.bid) if valid(put_ticker.bid) else 0
                    total_spread = call_spread + put_spread

                    new_limit = round(straddle_mid + aggression * total_spread, 2)

                    if new_limit <= current_limit:
                        logger.info(f"  {symbol}: Current ${current_limit:.2f} >= new ${new_limit:.2f}")
                        continue

                    combo_order.trade.order.lmtPrice = new_limit
                    self.ib.placeOrder(combo_order.trade.contract, combo_order.trade.order)

                    logger.info(f"  {symbol}: Improved ${current_limit:.2f} -> ${new_limit:.2f}")
                    improved += 1

                except Exception as e:
                    logger.error(f"  {symbol}: Price improvement error: {e}")

            logger.info(f"Price improvement complete: {improved}/{pending_count} modified")

        except Exception as e:
            logger.exception(f"Price improvement failed: {e}")

    async def task_exit_positions(self):
        """2:45 PM ET - Exit positions."""
        logger.info("=== EXITING POSITIONS ===")

        if not await self.ensure_connected():
            return

        self._load_positions_to_exit()

        if not self.positions_to_exit:
            logger.info("No positions to exit today")
            return

        if CONFIG['dry_run']:
            logger.info("DRY RUN - would exit positions")
            return

        try:
            for pos in self.positions_to_exit:
                logger.info(f"Exiting {pos['symbol']}...")

                # Use async close_position
                exit_pair = await close_position(
                    self.ib,
                    self.trade_logger,
                    trade_id=pos['trade_id'],
                    symbol=pos['symbol'],
                    expiry=pos['expiry'],
                    strike=pos['strike'],
                    contracts=pos['contracts'],
                    limit_aggression=CONFIG['limit_aggression'],
                    entry_fill_price=pos.get('entry_fill_price'),
                )

                if exit_pair:
                    logger.info(f"  Exit order placed for {pos['symbol']}")
                    self.active_exit_orders[pos['trade_id']] = exit_pair
                else:
                    logger.error(f"  Exit failed for {pos['symbol']}")

                await asyncio.sleep(1)

            logger.info(f"Tracking {len(self.active_exit_orders)} exit orders")

        except Exception as e:
            logger.exception(f"Position exit failed: {e}")

    async def task_evening_disconnect(self):
        """4:05 PM ET - Disconnect."""
        logger.info("=== EVENING DISCONNECT ===")

        try:
            await self.task_monitor_fills()

            if self.active_exit_orders:
                logger.info(f"Checking {len(self.active_exit_orders)} exit orders...")
                filled = check_exit_fills(self.active_exit_orders, self.trade_logger)
                for exit_pair in filled:
                    self.active_exit_orders.pop(exit_pair.trade_id, None)

            self.disconnect()
        except Exception as e:
            logger.exception(f"Evening disconnect failed: {e}")

    def _cleanup_stale_orders(self):
        """Clean up stale unfilled orders."""
        try:
            cancelled = self.trade_logger.cleanup_stale_orders(max_age_hours=24)
            if cancelled:
                logger.warning(f"Cleaned up {len(cancelled)} stale orders")
        except Exception as e:
            logger.error(f"Failed to cleanup stale orders: {e}")

    def _load_positions_to_exit(self):
        """Load positions to exit."""
        # Check both filled and partial trades
        filled_trades = self.trade_logger.get_trades(status='filled')
        try:
            partial_trades = self.trade_logger.get_trades(status='partial')
        except Exception:
            partial_trades = []

        trades = filled_trades + partial_trades

        self.positions_to_exit = []
        yesterday = date.today() - timedelta(days=1)

        for trade in trades:
            try:
                entry_date = datetime.fromisoformat(trade.entry_datetime).date()
            except Exception:
                continue

            if entry_date <= yesterday and not trade.exit_datetime:
                # Safe strike parsing
                strike = 0.0
                try:
                    import json
                    if trade.strikes:
                        strikes_list = json.loads(trade.strikes)
                        if strikes_list:
                            strike = float(strikes_list[0])
                except Exception as e:
                    logger.error(f"Error parsing strikes for {trade.ticker}: {e}")

                contracts = trade.contracts
                entry_fill_price = trade.entry_fill_price

                # For partials, we need to find actual filled quantity
                if trade.status == 'partial':
                    try:
                        last_event = self.trade_logger.get_latest_order_event(trade.trade_id)
                        if last_event and last_event.get('filled'):
                            contracts = int(last_event['filled'])
                            logger.info(f"{trade.ticker}: Resuming PARTIAL position - exiting {contracts} contracts (of {trade.contracts} planned)")
                        else:
                            logger.warning(f"{trade.ticker}: Partial status but no fill info - skipping exit for safety")
                            continue
                    except Exception as e:
                        logger.error(f"{trade.ticker}: Error checking partial fills: {e}")
                        continue

                self.positions_to_exit.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.ticker,
                    'expiry': trade.expiration,
                    'strike': strike,
                    'contracts': contracts,
                    'entry_fill_price': entry_fill_price,
                })

        if self.positions_to_exit:
            logger.info(f"Loaded {len(self.positions_to_exit)} positions to exit")

    def _load_todays_activity(self):
        """Load trades placed today to prevent duplicates on restart."""
        self.todays_trades = []
        today_str = date.today().isoformat()

        # Get all trades from today
        try:
            import sqlite3
            conn = sqlite3.connect(CONFIG['db_path'])
            c = conn.cursor()
            c.execute("SELECT trade_id, ticker FROM trades WHERE entry_datetime LIKE ?", (f"{today_str}%",))
            rows = c.fetchall()
            for tid, ticker in rows:
                self.todays_trades.append(tid)

            if rows:
                logger.info(f"Loaded {len(rows)} trades placed today")
        except Exception as e:
            logger.error(f"Failed to load today's activity: {e}")

    def _recover_exit_orders(self):
        """Recover active exit orders from database after restart."""
        try:
            # Find trades that are exiting/exited but we want to track
            # 'exited' might be useful if we want to confirm fill details,
            # but mainly we care about 'exiting' (placed but not confirmed filled by daemon)
            # OR 'exited' if we just restarted and want to catch up on fill events.
            # For simplicity, look for trades with exit order IDs.

            trades = self.trade_logger.get_trades()
            recovered_count = 0

            # Get open orders from IB to map orderId -> Trade object
            open_orders = {t.order.orderId: t for t in self.ib.openTrades()}

            for trade in trades:
                # We only care if:
                # 1. It has an exit order ID
                # 2. It's not fully finalized in our memory (though here we start fresh)
                # 3. Status is 'exiting' OR 'exited' (to catch up)
                if not trade.exit_call_order_id:
                    continue

                if trade.trade_id in self.active_exit_orders:
                    continue

                # Check if this exit order is still open in IB
                order_id = trade.exit_call_order_id
                ib_trade = open_orders.get(order_id)

                if not ib_trade:
                    # Not in open orders. Check if it's filled or cancelled according to DB
                    # If DB says 'exiting' but IB doesn't have it, it might be filled or cancelled.
                    # We can't easily reconstruct the ExitComboOrder without the IB Trade object
                    # unless we create a dummy one, which is risky.
                    # For now, skip if not in open orders.
                    # Ideally we should query execution details, but that's complex for combo.
                    continue

                # Reconstruct ExitComboOrder
                # Need to parse strikes/expiry from trade log
                try:
                    import json
                    strikes = json.loads(trade.strikes) if trade.strikes else []
                    strike = strikes[0] if strikes else 0.0
                except Exception:
                    strike = 0.0

                exit_order = ExitComboOrder(
                    trade_id=trade.trade_id,
                    symbol=trade.ticker,
                    expiry=trade.expiration,
                    strike=strike,
                    contracts=trade.contracts,
                    order_id=order_id,
                    trade=ib_trade,
                    entry_fill_price=trade.entry_fill_price,
                    status=ib_trade.orderStatus.status.lower()
                )

                self.active_exit_orders[trade.trade_id] = exit_order
                recovered_count += 1

            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} active exit orders")

        except Exception as e:
            logger.error(f"Failed to recover exit orders: {e}")

    async def _log_account_summary(self):
        """Log account info (Async)."""
        logger.info("=" * 50)
        logger.info("ACCOUNT SUMMARY")
        logger.info("=" * 50)

        # managedAccounts is sync and cached
        accounts = self.ib.managedAccounts()
        logger.info(f"Account(s): {accounts}")

        # accountValues is sync and cached (updates in background)
        all_values = self.ib.accountValues()
        for av in all_values:
            if av.tag in ('NetLiquidation', 'BuyingPower', 'UnrealizedPnL', 'RealizedPnL') and av.currency == 'USD':
                logger.info(f"  {av.tag}: {av.value} {av.currency}")

        positions = self.ib.positions()
        if positions:
            logger.info(f"OPEN POSITIONS: {len(positions)}")
            for pos in positions:
                logger.info(f"  {pos.contract.symbol}: {pos.position}")

        # Test live data using async
        await self._test_live_market_data_async()

    async def _test_live_market_data_async(self):
        """Test live data (Async)."""
        from ib_insync import Stock
        spy = Stock('SPY', 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(spy)

        ticker = self.ib.reqMktData(spy, '', False, False)
        for _ in range(10):
            if ticker.bid and ticker.bid > 0:
                break
            await asyncio.sleep(0.5)

        self.ib.cancelMktData(spy)
        logger.info(f"  SPY: bid={ticker.bid}, type={getattr(ticker, 'marketDataType', '?')}")

    def setup_schedule(self):
        """Configure the job schedule (Timezone Aware)."""

        # NOTE: When timezone=ET is set on the scheduler, triggers should use ET hours.
        # 09:25 ET = 9, 25
        # 14:00 ET = 14, 0

        self.scheduler.add_job(
            self.task_morning_connect,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=25, timezone=ET),
            id='morning_connect',
            name='Morning Connect (09:25 ET)',
        )

        self.scheduler.add_job(
            self.task_exit_positions,
            CronTrigger(day_of_week='mon-fri', hour=14, minute=45, timezone=ET),
            id='exit_positions',
            name='Exit Positions (14:45 ET)',
        )

        self.scheduler.add_job(
            self.task_screen_candidates,
            CronTrigger(day_of_week='mon-fri', hour=14, minute=0, timezone=ET),
            id='screen_and_place',
            name='Screen & Place Orders (14:00 ET)',
        )

        self.scheduler.add_job(
            self.task_monitor_fills,
            CronTrigger(day_of_week='mon-fri', hour='14-16', minute='*', timezone=ET),
            id='monitor_fills',
            name='Monitor Fills (Every min)',
        )

        # Price improvements 14:10 - 14:40
        for m, agg in [(10, 0.4), (20, 0.5), (30, 0.6), (40, 0.7)]:
            self.scheduler.add_job(
                lambda a=agg: asyncio.create_task(self.task_improve_prices(a)),
                CronTrigger(day_of_week='mon-fri', hour=14, minute=m, timezone=ET),
                id=f'improve_{m}',
                name=f'Price Improvement (14:{m} ET)',
            )

        self.scheduler.add_job(
            self.task_cancel_unfilled,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=58, timezone=ET),
            id='cancel_unfilled',
            name='Cancel Unfilled (15:58 ET)',
        )

        self.scheduler.add_job(
            self.task_check_fills,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=55, timezone=ET),
            id='check_fills',
            name='Check Fills (15:55 ET)',
        )

        self.scheduler.add_job(
            self.task_evening_disconnect,
            CronTrigger(day_of_week='mon-fri', hour=16, minute=5, timezone=ET),
            id='disconnect',
            name='Disconnect (16:05 ET)',
        )

        # Backfill - 16:30 ET
        self.scheduler.add_job(
            self.run_backfill_task,
            CronTrigger(day_of_week='mon-fri', hour=16, minute=30, timezone=ET),
            id='backfill',
            name='Backfill (16:30 ET)',
        )

    async def run_backfill_task(self):
        """Async wrapper for synchronous backfill task."""
        logger.info("Starting daily counterfactual backfill...")
        try:
            # Backfill for today (handles BMO earnings today)
            stats_today = await asyncio.to_thread(backfill_counterfactuals, self.trade_logger, date.today())
            logger.info(f"Backfill (Today): {stats_today}")

            # Backfill for yesterday (handles AMC earnings yesterday which exit today)
            yesterday = date.today() - timedelta(days=1)
            stats_yesterday = await asyncio.to_thread(backfill_counterfactuals, self.trade_logger, yesterday)
            logger.info(f"Backfill (Yesterday): {stats_yesterday}")

        except Exception as e:
            logger.exception(f"Backfill task failed: {e}")

    async def run_async(self):
        """Async entry point."""
        logger.info("DAEMON STARTING (Async)...")

        # Setup signals
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        self.setup_schedule()

        # Connect
        await self.connect_async()

        # Initial cleanup & load
        self._cleanup_stale_orders()
        self._load_todays_activity()

        # Recover any active orders (entry or exit) from DB if daemon restarted
        if self.executor:
            self.executor.recover_orders()
            self._recover_exit_orders()

        # Startup checks
        await self._run_screening_if_needed()

        # Start scheduler
        self.scheduler.start()
        logger.info("Scheduler started")

        # Keep alive
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.scheduler.shutdown()
        self.disconnect()
        # Cancel all tasks?
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.get_running_loop().stop()

    async def _run_screening_if_needed(self):
        """Run screening if started during window."""
        if not self.ib.isConnected():
            logger.error("STARTUP CONNECTION FAILED - check IB Gateway is running")
            logger.error("Daemon will continue and retry at scheduled times")
            logger.info("Not connected - skipping startup screening")
            return

        now = datetime.now(ET)

        # Original schedule: screen at 14:00.
        # But we'll allow catch-up if restarted between 14:00 and 21:00 (market hours + after hours)
        # to ensure we don't miss anything if daemon crashes.
        # CRITICAL: Must rely on _load_todays_activity() to prevent duplicates!

        start = now.replace(hour=14, minute=0, second=0)
        end = now.replace(hour=21, minute=0, second=0)

        # If between 14:00 and 21:00 ET on a weekday
        if now.weekday() < 5 and start <= now < end:
            # Check for existing trades loaded from DB
            if self.todays_trades:
                logger.info(f"STARTUP: Already placed {len(self.todays_trades)} trades today.")
                # If we've already hit the daily max, definitely skip
                if len(self.todays_trades) >= CONFIG['max_daily_trades']:
                     logger.info("STARTUP: Daily trade limit reached. Skipping startup screening.")
                     return
                else:
                     logger.info("STARTUP: Partial trades placed. Checking for additional candidates...")

            # Check for existing log entries to see if screening already ran?
            # Ideally we check if we already have non_trades for this earnings date too.
            # For now, relying on todays_trades matches + logic in task_screen_candidates is better.

            logger.info(f"STARTUP: Running screening (missed 14:00 scheduled time)")
            await self.task_screen_candidates()
        else:
            logger.info("STARTUP: Outside screening window (14:00-21:00 ET), waiting for next schedule.")

    def run(self):
        """Entry point."""
        try:
            asyncio.run(self.run_async())
        except (KeyboardInterrupt, SystemExit):
            pass

def main():
    daemon = TradingDaemon()
    daemon.run()

if __name__ == '__main__':
    main()
