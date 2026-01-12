#!/usr/bin/env python3
"""Phase 0 Trading Daemon (Async)

Automated earnings volatility trading with scheduled execution.

Schedule (all times ET):
- 09:25: Connect to IB Gateway, load positions to exit
- 14:00: Exit positions from previous day (free up capital first)
- 14:15: Screen upcoming earnings AND place new orders
- 14:25-14:55: Price improvements (every 10 min)
- 15:55: Final fill check, reprice unfilled exits to bid
- 15:58: Cancel unfilled entry orders, force exit with market orders
- 16:05: Disconnect (after market close)

Exit positions are handled the next trading day at 14:00.

Usage:
    python -m trading.earnings.daemon
"""
from __future__ import annotations

import os
import sys
import signal
import logging
import asyncio
import math
from datetime import datetime, date, timedelta
from functools import partial
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
    get_tradeable_candidates,
    screen_all_candidates,
    ScreenedCandidate,
    is_valid_price,
)
from trading.earnings.executor import (
    Phase0Executor, close_position, close_position_market, ExitComboOrder,
    check_exit_fills, reprice_exit_to_bid, convert_exit_to_market
)
from trading.earnings.logging import TradeLogger, SnapshotLog
from trading.earnings.ml_predictor import get_predictor, EarningsPredictor
from trading.earnings.counterfactual import backfill_counterfactuals

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Timezone
ET = pytz.timezone('US/Eastern')


def today_et() -> date:
    """Get today's date in Eastern Time."""
    return datetime.now(ET).date()

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

    # Position sizing: target equal dollar entry amounts
    'target_entry_amount': float(os.getenv('TARGET_ENTRY_AMOUNT', '2000')),  # target $ per trade
    'min_contracts': int(os.getenv('MIN_CONTRACTS', '1')),
    'max_contracts': int(os.getenv('MAX_CONTRACTS', '5')),  # safety cap
    'limit_aggression': float(os.getenv('LIMIT_AGGRESSION', '0.3')),
    'max_candidates_to_screen': int(os.getenv('MAX_CANDIDATES_TO_SCREEN', '50')),

    'db_path': PROJECT_ROOT / 'data' / 'earnings_trades.db',
    'log_path': PROJECT_ROOT / 'logs' / 'daemon.log',

    'dry_run': os.getenv('DRY_RUN', 'false').lower() == 'true',

    # LLM sanity check threshold: PASS, WARN, NO_TRADE, or DISABLED
    # - PASS: Only trade if LLM returns PASS (strictest)
    # - WARN: Trade on PASS or WARN, block on NO_TRADE (default for live)
    # - NO_TRADE: Trade on anything except NO_TRADE (permissive, good for paper)
    # - DISABLED: Skip LLM check entirely
    'llm_sanity_threshold': os.getenv(
        'PAPER_LLM_SANITY_THRESHOLD' if PAPER_MODE else 'LLM_SANITY_THRESHOLD',
        'NO_TRADE' if PAPER_MODE else 'WARN'
    ),
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

        # Async locks for shared state protection
        self._trades_lock = asyncio.Lock()  # Protects todays_trades
        self._exit_orders_lock = asyncio.Lock()  # Protects active_exit_orders

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
                await self._recover_exit_orders()

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

            # Fetch tradeable candidates using unified function
            # Handles: Nasdaq fetch, timing fill from yfinance, date verification against FMP
            loop = asyncio.get_running_loop()
            logger.info("Fetching tradeable candidates (with timing fill + date verification)...")
            bmo_tomorrow, amc_today = await loop.run_in_executor(
                None,
                lambda: get_tradeable_candidates(
                    days_ahead=3,
                    trade_logger=self.trade_logger,
                    fill_timing=True,
                    verify_dates=True,
                )
            )
            relevant_events = bmo_tomorrow + amc_today
            logger.info(f"Found {len(relevant_events)} earnings events to consider ({len(bmo_tomorrow)} BMO tmrw, {len(amc_today)} AMC today)")

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

                    # Add ML fields (all quantiles for calibration tracking)
                    candidate.pred_q50 = prediction.pred_q50
                    candidate.pred_q75 = prediction.pred_q75
                    candidate.pred_q90 = prediction.pred_q90
                    candidate.pred_q95 = prediction.pred_q95
                    candidate.hist_move_mean = prediction.hist_move_mean
                    candidate.edge_q75 = prediction.edge_q75
                    candidate.edge_q90 = prediction.edge_q90
                    candidate.news_count = prediction.news_count

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
            elif not passed:
                logger.info("No candidates passed liquidity screening - skipping ML filter")
            else:
                logger.warning("ML predictor not available - using liquidity-only screening")

            # LLM sanity check (if not disabled)
            # Thresholds: DISABLED (skip), LOG_ONLY (run but don't block), WARN, NO_TRADE, PASS
            llm_threshold = CONFIG['llm_sanity_threshold']
            if llm_threshold != 'DISABLED' and passed:
                from trading.earnings.llm_sanity_check import check_with_llm, build_sanity_packet

                log_only = llm_threshold == 'LOG_ONLY'
                logger.info(f"Running LLM sanity checks (threshold={llm_threshold}{', log-only mode' if log_only else ''})...")
                llm_passed = []

                for candidate in passed:
                    # Need to get prediction for this candidate to build packet
                    prediction = self.predictor.predict(
                        symbol=candidate.symbol,
                        earnings_date=candidate.earnings_date,
                        timing=candidate.timing,
                    ) if self.predictor else None

                    if not prediction:
                        # No prediction, skip LLM check but allow trade
                        llm_passed.append(candidate)
                        continue

                    packet = build_sanity_packet(candidate, prediction)
                    result = await check_with_llm(packet, self.trade_logger, ticker=candidate.symbol)

                    # LOG_ONLY mode: log results but always allow trade
                    if log_only:
                        logger.info(f"  {candidate.symbol}: LLM {result.decision} (log-only, proceeding) - {result.risk_flags}")
                        llm_passed.append(candidate)
                    elif result.decision == "NO_TRADE":
                        candidate.rejection_reason = f"LLM NO_TRADE: {result.reasons[0] if result.reasons else 'Unknown'}"
                        self.executor.log_non_trade(candidate)
                        logger.warning(f"  {candidate.symbol}: LLM NO_TRADE - {result.risk_flags}")
                    elif result.decision == "WARN" and llm_threshold == "PASS":
                        # Strict mode: block on WARN
                        candidate.rejection_reason = f"LLM WARN: {result.reasons[0] if result.reasons else 'Unknown'}"
                        self.executor.log_non_trade(candidate)
                        logger.warning(f"  {candidate.symbol}: LLM WARN (blocked in strict mode) - {result.risk_flags}")
                    else:
                        # PASS or WARN with permissive threshold
                        if result.decision == "WARN":
                            logger.warning(f"  {candidate.symbol}: LLM WARN (proceeding) - {result.risk_flags}")
                        else:
                            logger.info(f"  {candidate.symbol}: LLM PASS")
                        llm_passed.append(candidate)

                passed = llm_passed
                logger.info(f"LLM sanity check: {len(passed)} passed")

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

                # Use async place_straddle with position sizing config
                order_pair = await self.executor.place_straddle(
                    candidate,
                    target_entry_amount=CONFIG['target_entry_amount'],
                    min_contracts=CONFIG['min_contracts'],
                    max_contracts=CONFIG['max_contracts'],
                )

                if order_pair:
                    async with self._trades_lock:
                        self.todays_trades.append(order_pair.trade_id)
                    logger.info(f"  Order placed: {order_pair.trade_id}")
                else:
                    logger.error(f"  Order failed for {candidate.symbol}")

                await asyncio.sleep(1)

            logger.info(f"Placed {len(self.todays_trades)} orders")

        except Exception as e:
            logger.exception(f"Order placement failed: {e}")

    async def task_cancel_unfilled(self):
        """3:58 PM ET - Cancel unfilled entry orders and force exit positions."""
        logger.info("=== CANCELLING UNFILLED ORDERS ===")

        if not await self.ensure_connected():
            return

        if not self.executor:
            return

        try:
            # Cancel unfilled entry orders
            active_count = self.executor.get_active_count()
            if active_count == 0:
                logger.info("No active entry orders to cancel")
            else:
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

            # Force exit unfilled positions with market orders
            await self.task_force_exit_market()

        except Exception as e:
            logger.exception(f"Cancel unfilled failed: {e}")

    async def task_check_fills(self):
        """3:50 PM ET - Fill check & Cancel."""
        logger.info("=== CHECKING FILLS (NEAR CLOSE) ===")

        if not await self.ensure_connected():
            return

        try:
            # check_fills is async to properly handle the fill lock
            filled = await self.executor.check_fills()

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

            # Check exit order fills (with IB for spot price)
            if self.active_exit_orders:
                logger.info(f"Checking {len(self.active_exit_orders)} exit orders...")
                exit_filled = await check_exit_fills(self.active_exit_orders, self.trade_logger, self.ib)
                logger.info(f"Exit fills: {len(exit_filled)} completed")

                for exit_pair in exit_filled:
                    self.active_exit_orders.pop(exit_pair.trade_id, None)

                # Reprice unfilled exit orders to bid
                unfilled_exits = [
                    (tid, eo) for tid, eo in self.active_exit_orders.items()
                    if eo.status not in ('filled', 'cancelled')
                ]
                if unfilled_exits:
                    logger.warning(f"=== REPRICING {len(unfilled_exits)} UNFILLED EXIT ORDERS TO BID ===")
                    for trade_id, exit_order in unfilled_exits:
                        repriced = await reprice_exit_to_bid(exit_order, self.ib, self.trade_logger)
                        if repriced:
                            self.active_exit_orders[trade_id] = repriced
                        else:
                            logger.error(f"{exit_order.symbol}: Failed to reprice exit order")

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

        # Monitor entry fills
        if self.executor.get_active_count() > 0:
            try:
                filled = await self.executor.check_fills()
                for order in filled:
                    logger.info(f"New fill detected for {order.symbol}, scheduling markouts...")
                    self._schedule_markouts(order)
            except Exception as e:
                logger.error(f"Monitor entry fills failed: {e}")

        # Monitor exit fills
        if self.active_exit_orders:
            try:
                exit_filled = await check_exit_fills(self.active_exit_orders, self.trade_logger, self.ib)
                if exit_filled:
                    logger.info(f"Exit fills detected: {[e.trade_id for e in exit_filled]}")
            except Exception as e:
                logger.error(f"Monitor exit fills failed: {e}")

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
        call = None
        put = None
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

            if call_mid is not None and put_mid is not None:
                return call_mid + put_mid

            return None

        except Exception as e:
            logger.error(f"Error fetching straddle price for {symbol}: {e}")
            return None
        finally:
            # Always cancel market data subscriptions to prevent leaks
            if call is not None:
                try:
                    self.ib.cancelMktData(call)
                except Exception:
                    pass
            if put is not None:
                try:
                    self.ib.cancelMktData(put)
                except Exception:
                    pass

    async def _fetch_spot_price(self, symbol: str) -> Optional[float]:
        """Fetch current stock price (Async)."""
        stock = None
        try:
            from ib_insync import Stock
            stock = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(stock)

            ticker = self.ib.reqMktData(stock, '', False, False)

            # Wait for data
            for _ in range(20):
                if is_valid_price(ticker.last):
                    break
                if is_valid_price(ticker.close):
                    break
                await asyncio.sleep(0.1)

            spot = ticker.marketPrice()
            if not is_valid_price(spot):
                spot = ticker.last if is_valid_price(ticker.last) else ticker.close

            return spot if is_valid_price(spot) else None

        except Exception as e:
            logger.error(f"Error fetching spot price for {symbol}: {e}")
            return None
        finally:
            if stock is not None:
                try:
                    self.ib.cancelMktData(stock)
                except Exception:
                    pass

    def _get_open_positions(self) -> list[dict]:
        """Get positions to snapshot (open + exited today for counterfactual tracking)."""
        filled_trades = self.trade_logger.get_trades(status='filled')
        exited_trades = self.trade_logger.get_trades(status='exited')
        positions = []

        for trade in filled_trades + exited_trades:
            # Skip if exited before today (only track same-day exits for counterfactual)
            if trade.exit_datetime:
                try:
                    exit_date = datetime.fromisoformat(trade.exit_datetime).date()
                    if exit_date < today_et():
                        continue  # Exited on previous day, don't track
                except Exception:
                    continue

            # Parse strike from JSON
            try:
                import json
                strikes = json.loads(trade.strikes) if trade.strikes else []
                strike = strikes[0] if strikes else 0.0
            except Exception:
                strike = 0.0

            positions.append({
                'trade_id': trade.trade_id,
                'symbol': trade.ticker,
                'expiry': trade.expiration,
                'strike': strike,
                'contracts': trade.contracts,
                'entry_fill_price': trade.entry_fill_price,
                'spot_at_entry': trade.spot_at_entry,
            })

        return positions

    async def task_snapshot_positions(self):
        """Take price snapshot of all open positions (runs every 5 min during market hours)."""
        if not self.ib.isConnected():
            return

        positions = self._get_open_positions()
        if not positions:
            return

        now = datetime.now(ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = int((now - market_open).total_seconds() / 60)

        logger.debug(f"Snapshotting {len(positions)} positions at T+{minutes_since_open}min")

        for pos in positions:
            try:
                # Fetch straddle price (returns combined mid)
                straddle_mid = await self._fetch_straddle_price(
                    pos['symbol'], pos['expiry'], pos['strike']
                )

                if not straddle_mid:
                    continue

                # Calculate unrealized P&L
                unrealized_pnl = None
                unrealized_pnl_pct = None
                entry_fill = pos.get('entry_fill_price')
                if entry_fill and entry_fill > 0:
                    unrealized_pnl = (straddle_mid - entry_fill) * pos['contracts'] * 100
                    unrealized_pnl_pct = straddle_mid / entry_fill - 1

                # Fetch spot price
                spot = await self._fetch_spot_price(pos['symbol'])

                snapshot = SnapshotLog(
                    trade_id=pos['trade_id'],
                    ts=now.isoformat(),
                    minutes_since_open=minutes_since_open,
                    straddle_mid=straddle_mid,
                    call_mid=None,  # Could parse from _fetch_straddle_price if needed
                    put_mid=None,
                    spot_price=spot,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                )
                self.trade_logger.log_snapshot(snapshot)

            except Exception as e:
                logger.error(f"Snapshot error for {pos['symbol']}: {e}")

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
                        return p is not None and not math.isnan(p) and p > 0

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
                    spot_at_entry=pos.get('spot_at_entry'),
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

    async def task_force_exit_market(self):
        """3:58 PM ET - Force close remaining positions with market orders."""
        logger.info("=== FORCE EXIT (MARKET ORDERS) ===")

        if not await self.ensure_connected():
            return

        try:
            # First check if any active exit orders filled since last check
            if self.active_exit_orders:
                exit_filled = await check_exit_fills(self.active_exit_orders, self.trade_logger, self.ib)
                for exit_pair in exit_filled:
                    self.active_exit_orders.pop(exit_pair.trade_id, None)

            # Convert remaining unfilled active exit orders to market orders
            unfilled_exits = [
                (tid, eo) for tid, eo in self.active_exit_orders.items()
                if eo.status not in ('filled', 'cancelled')
            ]

            if unfilled_exits:
                logger.warning(f"Converting {len(unfilled_exits)} unfilled exits to MARKET ORDERS")
                for trade_id, exit_order in unfilled_exits:
                    converted = await convert_exit_to_market(exit_order, self.ib, self.trade_logger)
                    if converted:
                        self.active_exit_orders[trade_id] = converted
                    else:
                        logger.error(f"{exit_order.symbol}: Failed to convert to market order - MANUAL CLOSE REQUIRED")

            # CRITICAL: Also check for orphaned "exiting" positions in DB
            # These are positions where exit orders were placed but daemon restarted
            # and couldn't recover them (e.g., orders in limbo state)
            await self._force_exit_orphaned_positions()

        except Exception as e:
            logger.exception(f"Force exit failed: {e}")

    async def _force_exit_orphaned_positions(self):
        """Force exit positions that are stuck in 'exiting' status."""
        import json

        try:
            # Get actual positions from IBKR
            positions = self.ib.positions()
            position_symbols = {
                p.contract.symbol for p in positions
                if p.contract.secType == 'OPT' and p.position != 0
            }

            if not position_symbols:
                return

            # Check for trades marked 'exiting' that still have positions
            exiting_trades = self.trade_logger.get_trades(status='exiting')

            for trade in exiting_trades:
                # Skip if we're already tracking this exit
                if trade.trade_id in self.active_exit_orders:
                    continue

                # Check if position actually exists
                if trade.ticker not in position_symbols:
                    # Position gone but status not updated - mark as exited
                    logger.info(f"{trade.ticker}: Position closed (manually?), updating status")
                    self.trade_logger.update_trade(trade.trade_id, status='exited')
                    continue

                # Position exists but we're not tracking it - force exit with market order
                logger.warning(f"{trade.ticker}: Orphaned position found - forcing market exit")

                # Parse strike
                strike = 0.0
                try:
                    if trade.strikes:
                        strikes_list = json.loads(trade.strikes)
                        if strikes_list:
                            strike = float(strikes_list[0])
                except Exception:
                    pass

                # Use market order exit (no quotes needed)
                exit_order = await close_position_market(
                    self.ib, self.trade_logger, trade.trade_id,
                    trade.ticker, trade.expiration, strike, trade.contracts,
                    entry_fill_price=trade.entry_fill_price,
                    spot_at_entry=trade.spot_at_entry
                )

                if exit_order:
                    self.active_exit_orders[trade.trade_id] = exit_order
                    logger.info(f"{trade.ticker}: Market exit order placed")
                else:
                    logger.error(f"{trade.ticker}: FAILED to place market exit - MANUAL CLOSE REQUIRED")

        except Exception as e:
            logger.error(f"Failed to force exit orphaned positions: {e}")

    async def task_evening_disconnect(self):
        """4:05 PM ET - Disconnect."""
        logger.info("=== EVENING DISCONNECT ===")

        try:
            await self.task_monitor_fills()

            if self.active_exit_orders:
                logger.info(f"Checking {len(self.active_exit_orders)} exit orders...")
                filled = await check_exit_fills(self.active_exit_orders, self.trade_logger, self.ib)
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
        yesterday = today_et() - timedelta(days=1)

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
                    'spot_at_entry': trade.spot_at_entry,
                })

        if self.positions_to_exit:
            logger.info(f"Loaded {len(self.positions_to_exit)} positions to exit")

    def _load_todays_activity(self):
        """Load trades placed today to prevent duplicates on restart."""
        self.todays_trades = []
        today_str = today_et().isoformat()

        # Get all trades from today
        try:
            import sqlite3
            with sqlite3.connect(CONFIG['db_path']) as conn:
                c = conn.cursor()
                c.execute("SELECT trade_id, ticker FROM trades WHERE entry_datetime LIKE ?", (f"{today_str}%",))
                rows = c.fetchall()
                for tid, ticker in rows:
                    self.todays_trades.append(tid)

            if rows:
                logger.info(f"Loaded {len(rows)} trades placed today")
        except Exception as e:
            logger.error(f"Failed to load today's activity: {e}")

    async def _recover_exit_orders(self):
        """Recover active exit orders from database after restart."""
        try:
            # Find trades that are exiting/exited but we want to track
            # 'exiting' (placed but not confirmed filled by daemon)

            trades = self.trade_logger.get_trades()
            recovered_count = 0
            filled_count = 0
            new_exit_count = 0

            # Get open orders from IB to map orderId -> Trade object
            open_orders = {t.order.orderId: t for t in self.ib.openTrades()}

            # Get recent fills to check for already-filled exit orders
            fills = self.ib.fills()
            fills_by_order = {}
            for fill in fills:
                order_id = fill.execution.orderId
                if order_id not in fills_by_order:
                    fills_by_order[order_id] = []
                fills_by_order[order_id].append(fill)

            # Get current positions for fallback
            positions = self.ib.positions()

            for trade in trades:
                # We only care if:
                # 1. It has an exit order ID
                # 2. Status is 'exiting' (not yet confirmed filled)
                if not trade.exit_call_order_id:
                    continue

                if trade.status != 'exiting':
                    continue

                if trade.trade_id in self.active_exit_orders:
                    continue

                # Parse strike for later use
                try:
                    import json
                    strikes = json.loads(trade.strikes) if trade.strikes else []
                    strike = strikes[0] if strikes else 0.0
                except Exception:
                    strike = 0.0

                call_order_id = trade.exit_call_order_id
                put_order_id = trade.exit_put_order_id

                # Check if exit orders are still open in IB
                call_ib_trade = open_orders.get(call_order_id)
                put_ib_trade = open_orders.get(put_order_id)

                if call_ib_trade or put_ib_trade:
                    # At least one leg is still open - recover for monitoring
                    exit_order = ExitComboOrder(
                        trade_id=trade.trade_id,
                        symbol=trade.ticker,
                        expiry=trade.expiration,
                        strike=strike,
                        contracts=trade.contracts,
                        order_id=call_order_id,
                        trade=call_ib_trade,
                        put_trade=put_ib_trade,
                        entry_fill_price=trade.entry_fill_price,
                        spot_at_entry=trade.spot_at_entry,
                        status='pending'
                    )

                    self.active_exit_orders[trade.trade_id] = exit_order
                    recovered_count += 1
                else:
                    # Not in open orders - check if both legs filled
                    call_fills = fills_by_order.get(call_order_id, [])
                    put_fills = fills_by_order.get(put_order_id, [])

                    if call_fills and put_fills:
                        # Both legs filled - calculate P&L and update DB
                        call_shares = sum(f.execution.shares for f in call_fills)
                        put_shares = sum(f.execution.shares for f in put_fills)
                        if call_shares <= 0 or put_shares <= 0:
                            logger.warning(f"{trade.ticker}: Zero shares in exit fills (call={call_shares}, put={put_shares})")
                            continue
                        call_fill_price = sum(f.execution.price * f.execution.shares for f in call_fills) / call_shares
                        put_fill_price = sum(f.execution.price * f.execution.shares for f in put_fills) / put_shares
                        combined_fill = call_fill_price + put_fill_price

                        if trade.entry_fill_price:
                            exit_pnl = (combined_fill - trade.entry_fill_price) * trade.contracts * 100
                            exit_pnl_pct = (combined_fill - trade.entry_fill_price) / trade.entry_fill_price
                        else:
                            exit_pnl = None
                            exit_pnl_pct = None

                        self.trade_logger.update_trade(
                            trade.trade_id,
                            status='exited',
                            exit_fill_price=combined_fill,
                            exit_pnl=exit_pnl,
                            exit_pnl_pct=exit_pnl_pct,
                        )
                        logger.info(
                            f"Recovered filled exit for {trade.ticker}: "
                            f"Call ${call_fill_price:.2f} + Put ${put_fill_price:.2f} = ${combined_fill:.2f}, "
                            f"P&L: ${exit_pnl:.2f}" if exit_pnl else f"Recovered filled exit for {trade.ticker}"
                        )
                        filled_count += 1
                    else:
                        # Order not in open trades and not in fills
                        # Check if position still exists - if so, place new exit order
                        has_position = any(
                            p.contract.symbol == trade.ticker and
                            p.contract.secType == 'OPT' and
                            p.position != 0
                            for p in positions
                        )

                        if has_position:
                            logger.warning(
                                f"Exit order for {trade.ticker} not found but position exists - placing new exit order"
                            )
                            # Try limit order first (requires quotes)
                            exit_order = await close_position(
                                self.ib, self.trade_logger, trade.trade_id,
                                trade.ticker, trade.expiration, strike, trade.contracts,
                                limit_aggression=0.0,  # At bid for faster fill
                                entry_fill_price=trade.entry_fill_price,
                                spot_at_entry=trade.spot_at_entry
                            )
                            if exit_order:
                                self.active_exit_orders[trade.trade_id] = exit_order
                                new_exit_count += 1
                                logger.info(f"{trade.ticker}: New exit order placed successfully")
                            else:
                                # Quotes unavailable (market closed?) - track for later
                                # Create placeholder ExitComboOrder so force exit can find it
                                logger.warning(
                                    f"{trade.ticker}: No quotes available - will retry at next exit window"
                                )
                                # Add to positions_to_exit for next task_exit_positions run
                                self.positions_to_exit.append({
                                    'trade_id': trade.trade_id,
                                    'symbol': trade.ticker,
                                    'expiry': trade.expiration,
                                    'strike': strike,
                                    'contracts': trade.contracts,
                                    'entry_fill_price': trade.entry_fill_price,
                                    'spot_at_entry': trade.spot_at_entry,
                                    'orphaned': True,  # Flag for special handling
                                })
                        else:
                            # No position and no fills - position may have been closed manually
                            logger.warning(
                                f"Exit order for {trade.ticker} not found and no position exists "
                                f"(call_id={call_order_id}, put_id={put_order_id}) - may need manual DB update"
                            )

            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} active exit orders")
            if filled_count > 0:
                logger.info(f"Marked {filled_count} exit orders as filled from execution history")
            if new_exit_count > 0:
                logger.info(f"Placed {new_exit_count} new exit orders for orphaned positions")

        except Exception as e:
            logger.error(f"Failed to recover exit orders: {e}")

    async def _reconcile_positions(self):
        """Reconcile IBKR positions with database state on startup."""
        import json

        logger.info("=== POSITION RECONCILIATION ===")

        try:
            # Get actual positions from IBKR
            positions = self.ib.positions()
            ibkr_options = {}
            for p in positions:
                if p.contract.secType == 'OPT' and p.position != 0:
                    symbol = p.contract.symbol
                    if symbol not in ibkr_options:
                        ibkr_options[symbol] = []
                    ibkr_options[symbol].append({
                        'right': p.contract.right,
                        'strike': p.contract.strike,
                        'expiry': p.contract.lastTradeDateOrContractMonth,
                        'qty': abs(p.position),
                        'avg_cost': p.avgCost,
                    })

            if not ibkr_options:
                logger.info("No option positions in IBKR")
                return

            logger.info(f"IBKR option positions: {list(ibkr_options.keys())}")

            # Get trades from DB that should have positions
            filled_trades = self.trade_logger.get_trades(status='filled')
            exiting_trades = self.trade_logger.get_trades(status='exiting')
            db_positions = {t.ticker: t for t in filled_trades + exiting_trades}

            # Also check "exited" trades - they might have orphaned positions
            exited_trades = self.trade_logger.get_trades(status='exited')
            db_exited = {t.ticker: t for t in exited_trades}

            # Check for mismatches
            mismatches = []

            # 1. Positions in IBKR but not in DB (filled/exiting)
            for symbol in ibkr_options:
                if symbol not in db_positions:
                    # Check if it's marked as "exited" but still has position
                    if symbol in db_exited:
                        trade = db_exited[symbol]
                        logger.warning(
                            f"{symbol}: Marked 'exited' but IBKR still has position - reopening as 'exiting'"
                        )
                        self.trade_logger.update_trade(
                            trade.trade_id,
                            status='exiting',
                            notes='Reopened: position still exists in IBKR'
                        )
                        # Add to db_positions for further processing
                        db_positions[symbol] = trade
                    else:
                        mismatches.append(f"IBKR has {symbol} but no matching DB trade")

            # 2. DB says we have position but IBKR doesn't
            for ticker, trade in db_positions.items():
                if ticker not in ibkr_options:
                    mismatches.append(
                        f"DB trade {trade.trade_id} ({ticker}) is {trade.status} but no IBKR position"
                    )
                    # If exiting with no position, mark as exited
                    if trade.status == 'exiting':
                        logger.info(f"{ticker}: Marking as exited (position closed)")
                        self.trade_logger.update_trade(trade.trade_id, status='exited')

            if mismatches:
                logger.warning("POSITION MISMATCHES DETECTED:")
                for m in mismatches:
                    logger.warning(f"  - {m}")
            else:
                logger.info("Positions reconciled - no mismatches")

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

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
            CronTrigger(day_of_week='mon-fri', hour=14, minute=0, timezone=ET),
            id='exit_positions',
            name='Exit Positions (14:00 ET)',
        )

        self.scheduler.add_job(
            self.task_screen_candidates,
            CronTrigger(day_of_week='mon-fri', hour=14, minute=15, timezone=ET),
            id='screen_and_place',
            name='Screen & Place Orders (14:15 ET)',
        )

        self.scheduler.add_job(
            self.task_monitor_fills,
            CronTrigger(day_of_week='mon-fri', hour='14-16', minute='*', timezone=ET),
            id='monitor_fills',
            name='Monitor Fills (Every min)',
            misfire_grace_time=30,  # Allow 30s delay, skip if too late
            coalesce=True,  # Combine missed jobs into one
        )

        # Position snapshots every 5 minutes from 9:30-16:00 for intraday P&L tracking
        self.scheduler.add_job(
            self.task_snapshot_positions,
            CronTrigger(day_of_week='mon-fri', hour='9-15', minute='*/5', timezone=ET),
            id='snapshot_positions',
            name='Snapshot Positions (Every 5min)',
        )

        # Price improvements 14:25 - 14:55 (after 14:15 order placement)
        for m, agg in [(25, 0.4), (35, 0.5), (45, 0.6), (55, 0.7)]:
            self.scheduler.add_job(
                partial(self.task_improve_prices, agg),
                CronTrigger(day_of_week='mon-fri', hour=14, minute=m, timezone=ET),
                id=f'improve_{m}',
                name=f'Price Improvement (14:{m:02d} ET)',
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
            stats_today = await asyncio.to_thread(backfill_counterfactuals, self.trade_logger, today_et())
            logger.info(f"Backfill (Today): {stats_today}")

            # Backfill for yesterday (handles AMC earnings yesterday which exit today)
            yesterday = today_et() - timedelta(days=1)
            stats_yesterday = await asyncio.to_thread(backfill_counterfactuals, self.trade_logger, yesterday)
            logger.info(f"Backfill (Yesterday): {stats_yesterday}")

        except Exception as e:
            logger.exception(f"Backfill task failed: {e}")

    def _handle_signal(self):
        """Signal handler that safely triggers shutdown."""
        async def safe_shutdown():
            try:
                await self.shutdown()
            except Exception as e:
                logger.exception(f"Error during shutdown: {e}")
                asyncio.get_running_loop().stop()
        asyncio.create_task(safe_shutdown())

    async def run_async(self):
        """Async entry point."""
        logger.info("DAEMON STARTING (Async)...")

        # Setup signals with safe handler
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal)

        self.setup_schedule()

        # Connect
        await self.connect_async()

        # Initial cleanup & load
        self._cleanup_stale_orders()
        self._load_todays_activity()

        # Recover any active orders (entry or exit) from DB if daemon restarted
        if self.executor:
            self.executor.recover_orders()
            await self._recover_exit_orders()

        # Reconcile positions (detect IBKR vs DB mismatches)
        await self._reconcile_positions()

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
