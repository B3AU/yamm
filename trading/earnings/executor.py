"""Phase 0 executor - places and manages straddle orders.

Handles:
- Order placement with limit pricing strategy (using combo/BAG orders)
- Fill monitoring and logging
- Position tracking

Uses IBKR combo (BAG) orders for straddles to ensure atomic execution
of both legs, eliminating orphan leg risk.
"""
from __future__ import annotations

import logging
import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytz

from ib_insync import IB, Option, Stock, LimitOrder, MarketOrder, Trade, Contract, ComboLeg

from trading.earnings.screener import ScreenedCandidate
from trading.earnings.logging import (
    TradeLog, NonTradeLog, TradeLogger,
    generate_trade_id, generate_log_id
)

logger = logging.getLogger(__name__)


def create_straddle_contract(
    call: Option,
    put: Option,
    action: str = 'BUY',
) -> Contract:
    """Create a combo contract for a straddle (call + put)."""
    combo = Contract()
    combo.symbol = call.symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'

    combo.comboLegs = [
        ComboLeg(
            conId=call.conId,
            ratio=1,
            action=action,
            exchange='SMART',
        ),
        ComboLeg(
            conId=put.conId,
            ratio=1,
            action=action,
            exchange='SMART',
        ),
    ]
    return combo


@dataclass
class ComboOrder:
    """Tracks a straddle combo order (call + put as single atomic order)."""
    trade_id: str
    symbol: str
    expiry: str
    strike: float

    order_id: Optional[int] = None
    trade: Optional[Trade] = None

    # Individual leg conIds for reference
    call_conId: Optional[int] = None
    put_conId: Optional[int] = None

    fill_price: Optional[float] = None  # Combined straddle fill price
    placement_time: Optional[datetime] = None  # For fill latency tracking

    status: str = 'pending'  # pending, filled, cancelled


@dataclass
class ExitComboOrder:
    """Tracks exit orders for a straddle position (separate call/put legs)."""
    trade_id: str
    symbol: str
    expiry: str
    strike: float
    contracts: int

    order_id: Optional[int] = None  # Call order ID (primary)
    trade: Optional[Trade] = None  # Call trade (primary)
    put_trade: Optional[Trade] = None  # Put trade (secondary)

    call_conId: Optional[int] = None
    put_conId: Optional[int] = None

    fill_price: Optional[float] = None  # Combined fill price
    call_fill_price: Optional[float] = None
    put_fill_price: Optional[float] = None
    entry_fill_price: Optional[float] = None  # For P&L calculation
    spot_at_entry: Optional[float] = None  # For realized move calculation

    status: str = 'pending'  # pending, filled


class Phase0Executor:
    """Executes Phase 0 validation trades using combo orders."""

    def __init__(
        self,
        ib: IB,
        trade_logger: TradeLogger,
        limit_aggression: float = 0.3,  # How much above mid to place limit
    ):
        self.ib = ib
        self.logger = trade_logger
        self.limit_aggression = limit_aggression
        self.active_orders: dict[str, ComboOrder] = {}
        self._fill_lock = asyncio.Lock()  # Prevents race conditions in fill detection (async-safe)

    def _create_straddle_combo(
        self,
        call: Option,
        put: Option,
    ) -> Optional[Contract]:
        """Create a combo contract for a straddle (call + put)."""
        return create_straddle_contract(call, put, 'BUY')

    async def place_straddle(
        self,
        candidate: ScreenedCandidate,
        target_entry_amount: float = None,
        min_contracts: int = 1,
        max_contracts: int = 5,
    ) -> Optional[ComboOrder]:
        """
        Place a straddle combo order (buy call + put atomically).

        Uses a single combo/BAG order to ensure both legs fill together,
        eliminating orphan leg risk.

        Position sizing: If target_entry_amount is set, calculates contracts
        to achieve approximately that dollar entry (premium * contracts * 100).
        """
        # Calculate combined straddle limit price
        call_mid = (candidate.call_bid + candidate.call_ask) / 2
        put_mid = (candidate.put_bid + candidate.put_ask) / 2
        straddle_mid = call_mid + put_mid

        # Combined spread for straddle
        straddle_bid = candidate.call_bid + candidate.put_bid
        straddle_ask = candidate.call_ask + candidate.put_ask
        straddle_spread = straddle_ask - straddle_bid

        # Limit price: mid + aggression * spread
        straddle_limit = round(straddle_mid + self.limit_aggression * straddle_spread, 2)

        # Position sizing: target equal dollar entry amounts
        straddle_cost = straddle_mid * 100  # cost per contract in dollars
        if target_entry_amount and target_entry_amount > 0:
            contracts = int(target_entry_amount / straddle_cost)
            contracts = max(min_contracts, min(contracts, max_contracts))
        else:
            contracts = min_contracts

        if contracts < min_contracts:
            logger.info(f"{candidate.symbol}: Straddle too expensive (${straddle_cost:.0f}/contract vs ${target_entry_amount:.0f} target) - skipping")
            return None

        logger.info(f"{candidate.symbol}: Position size = {contracts} contracts (${straddle_cost:.0f}/contract, target ${target_entry_amount:.0f})")

        # Create and qualify individual option contracts first
        call = Option(
            candidate.symbol,
            candidate.expiry,
            candidate.atm_strike,
            'C',
            'SMART',
            tradingClass=candidate.symbol
        )
        put = Option(
            candidate.symbol,
            candidate.expiry,
            candidate.atm_strike,
            'P',
            'SMART',
            tradingClass=candidate.symbol
        )

        try:
            qualified = await self.ib.qualifyContractsAsync(call, put)
            if len(qualified) < 2:
                logger.error(f"{candidate.symbol}: Could not qualify options for order")
                return None
        except Exception as e:
            logger.error(f"{candidate.symbol}: Qualification error: {e}")
            return None

        # Create combo contract
        combo = self._create_straddle_combo(call, put)

        # Generate trade ID
        trade_id = generate_trade_id(candidate.symbol, str(candidate.earnings_date))

        # Capture timestamp for decision latency tracking (ET timezone)
        ET = pytz.timezone('US/Eastern')
        decision_time = datetime.now(ET)

        # Log the trade entry with all quantiles for calibration tracking
        trade_log = TradeLog(
            trade_id=trade_id,
            ticker=candidate.symbol,
            earnings_date=str(candidate.earnings_date),
            earnings_timing=candidate.timing,
            entry_datetime=decision_time.isoformat(),
            entry_quoted_bid=straddle_bid,
            entry_quoted_ask=straddle_ask,
            entry_quoted_mid=straddle_mid,
            entry_combo_bid=straddle_bid,  # Capture at moment of decision
            entry_combo_ask=straddle_ask,
            entry_limit_price=straddle_limit,
            structure='straddle_combo',
            strikes=str([candidate.atm_strike]),
            expiration=candidate.expiry,
            contracts=contracts,
            premium_paid=0,  # Updated on fill
            max_loss=straddle_limit * contracts * 100,
            predicted_q50=candidate.pred_q50,
            predicted_q75=candidate.pred_q75,
            predicted_q90=candidate.pred_q90,
            predicted_q95=candidate.pred_q95,
            edge_q75=candidate.edge_q75,
            edge_q90=candidate.edge_q90,
            implied_move=candidate.implied_move_pct / 100,
            spot_at_entry=candidate.spot_price,
            status='pending',
            news_count=candidate.news_count,
        )
        self.logger.log_trade(trade_log)

        # Place combo order
        order = LimitOrder('BUY', contracts, straddle_limit)

        try:
            trade = self.ib.placeOrder(combo, order)

            # Log initial placement event
            self.logger.log_order_event(
                trade_id=trade_id,
                ib_order_id=trade.order.orderId,
                event='placed',
                status='PendingSubmit', # Initial IB status
                filled=0,
                remaining=contracts,
                avg_fill_price=0.0,
                limit_price=straddle_limit,
                details={'type': 'entry', 'symbol': candidate.symbol}
            )

        except Exception as e:
            logger.error(f"{candidate.symbol}: Combo order placement error: {e}")
            self.logger.update_trade(trade_id, status='error', notes=str(e))
            return None

        combo_order = ComboOrder(
            trade_id=trade_id,
            symbol=candidate.symbol,
            expiry=candidate.expiry,
            strike=candidate.atm_strike,
            order_id=trade.order.orderId,
            trade=trade,
            call_conId=call.conId,
            put_conId=put.conId,
            placement_time=datetime.now(),  # For fill latency tracking
            status='pending',
        )

        self.active_orders[trade_id] = combo_order

        # Save order ID to DB for recovery after restart
        self.logger.update_trade(
            trade_id,
            call_order_id=trade.order.orderId,  # Using call_order_id for combo order ID
            notes=f"Combo order: call_conId={call.conId}, put_conId={put.conId}",
        )

        logger.info(
            f"{candidate.symbol}: Placed straddle COMBO order - "
            f"{contracts}x @ ${straddle_limit:.2f} (mid: ${straddle_mid:.2f})"
        )

        return combo_order

    async def check_fills(self) -> list[ComboOrder]:
        """Check status of all active combo orders and update logs.

        Async-safe: uses lock to prevent race conditions when called
        concurrently from multiple scheduler jobs.
        """
        async with self._fill_lock:
            filled = []

            for trade_id, combo_order in list(self.active_orders.items()):
                if not combo_order.trade:
                    continue

                # Defensive null check for trade.orderStatus
                if not combo_order.trade.orderStatus:
                    logger.warning(f"Trade object missing orderStatus for {combo_order.symbol}")
                    continue

                status = combo_order.trade.orderStatus.status
                filled_qty = combo_order.trade.orderStatus.filled
                total_qty = combo_order.trade.order.totalQuantity
                remaining_qty = combo_order.trade.orderStatus.remaining
                last_fill_price = combo_order.trade.orderStatus.lastFillPrice
                avg_fill_price = combo_order.trade.orderStatus.avgFillPrice

                # Log granular status update if status changed or filled qty increased
                prev_status = getattr(combo_order, '_last_log_status', None)
                prev_filled = getattr(combo_order, '_last_log_filled', 0)

                if status != prev_status or filled_qty > prev_filled:
                    self.logger.log_order_event(
                        trade_id=trade_id,
                        ib_order_id=combo_order.trade.order.orderId,
                        event='status_update' if filled_qty == prev_filled else 'fill',
                        status=status,
                        filled=filled_qty,
                        remaining=remaining_qty,
                        avg_fill_price=avg_fill_price,
                        limit_price=combo_order.trade.order.lmtPrice,
                        last_fill_price=last_fill_price,
                        last_fill_qty=filled_qty - prev_filled if filled_qty > prev_filled else 0
                    )
                    combo_order._last_log_status = status
                    combo_order._last_log_filled = filled_qty

                if status == 'Filled' and combo_order.fill_price is None:
                    combo_order.fill_price = avg_fill_price
                    combo_order.status = 'filled'
                    filled.append(combo_order)

                    # Calculate fill latency
                    fill_time = datetime.now()
                    fill_latency_seconds = None
                    if combo_order.placement_time:
                        fill_latency_seconds = (fill_time - combo_order.placement_time).total_seconds()

                    latency_str = f" (latency: {fill_latency_seconds:.0f}s)" if fill_latency_seconds else ""
                    logger.info(f"{combo_order.symbol}: Straddle filled @ ${combo_order.fill_price:.2f}{latency_str}")

                    # Update trade log with fill details and timing
                    self.logger.update_trade(
                        trade_id,
                        status='filled',
                        entry_fill_price=combo_order.fill_price,
                        entry_fill_time=fill_time.isoformat(),
                        entry_slippage=combo_order.fill_price - combo_order.trade.order.lmtPrice,
                        premium_paid=combo_order.fill_price * total_qty * 100,  # Use ordered quantity for max loss
                        fill_latency_seconds=fill_latency_seconds,
                    )

                elif status in ('Cancelled', 'Inactive'):
                    combo_order.status = 'cancelled'
                    self.logger.update_trade(trade_id, status='cancelled')
                    logger.info(f"{combo_order.symbol}: Order cancelled/inactive")

                elif filled_qty > 0 and filled_qty < total_qty:
                    # Partial fill on combo (rare but possible)
                    if combo_order.status != 'partial':
                        combo_order.status = 'partial'
                        self.logger.update_trade(trade_id, status='partial')

                    logger.warning(
                        f"{combo_order.symbol}: PARTIAL COMBO FILL - "
                        f"{int(filled_qty)}/{int(total_qty)} ({status})"
                    )

            return filled

    def get_partial_fills(self) -> list[ComboOrder]:
        """Get all orders with partial fills.

        With combo orders, partial fills are much rarer since both legs
        must execute together. This mainly catches edge cases.
        """
        partials = []
        for combo_order in self.active_orders.values():
            if not combo_order.trade:
                continue

            filled_qty = combo_order.trade.orderStatus.filled
            total_qty = combo_order.trade.order.totalQuantity

            if filled_qty > 0 and filled_qty < total_qty:
                partials.append(combo_order)

        return partials

    def cancel_unfilled_orders(self, trade_id: str) -> dict:
        """Cancel unfilled combo order.

        Returns dict with what was cancelled.
        """
        if trade_id not in self.active_orders:
            return {'cancelled': False, 'reason': 'not found'}

        combo_order = self.active_orders[trade_id]

        try:
            if combo_order.trade:
                status = combo_order.trade.orderStatus.status
                if status not in ('Filled', 'Cancelled', 'Inactive'):
                    self.ib.cancelOrder(combo_order.trade.order)
                    logger.info(f"{combo_order.symbol}: Cancelled unfilled combo order")

                    combo_order.status = 'cancelled'
                    self.logger.update_trade(trade_id, status='cancelled', notes='No fills before close')
                    return {'cancelled': True}

        except Exception as e:
            logger.error(f"Cancel error for {trade_id}: {e}")
            return {'cancelled': False, 'reason': str(e)}

        return {'cancelled': False, 'reason': 'already filled or inactive'}

    def cancel_order(self, trade_id: str) -> bool:
        """Cancel an active combo order."""
        if trade_id not in self.active_orders:
            return False

        combo_order = self.active_orders[trade_id]

        try:
            if combo_order.trade:
                self.ib.cancelOrder(combo_order.trade.order)
        except Exception as e:
            logger.error(f"Cancel error for {trade_id}: {e}")
            return False

        combo_order.status = 'cancelled'
        self.logger.update_trade(trade_id, status='cancelled')

        return True

    def cancel_all(self):
        """Cancel all active orders."""
        for trade_id in list(self.active_orders.keys()):
            self.cancel_order(trade_id)

    def log_non_trade(self, candidate: ScreenedCandidate):
        """Log a candidate we passed on."""
        log_id = generate_log_id(candidate.symbol)

        # Calculate straddle values
        straddle_bid = candidate.call_bid + candidate.put_bid if candidate.call_bid else None
        straddle_ask = candidate.call_ask + candidate.put_ask if candidate.call_ask else None

        # Premium per contract = mid price * 100 (options multiplier)
        straddle_premium = None
        if straddle_bid is not None and straddle_ask is not None:
            straddle_mid = (straddle_bid + straddle_ask) / 2
            straddle_premium = straddle_mid * 100

        non_trade = NonTradeLog(
            log_id=log_id,
            ticker=candidate.symbol,
            earnings_date=str(candidate.earnings_date),
            earnings_timing=candidate.timing,
            log_datetime=datetime.now().isoformat(),
            rejection_reason=candidate.rejection_reason or 'unknown',
            quoted_bid=straddle_bid,
            quoted_ask=straddle_ask,
            quoted_spread_pct=candidate.spread_pct if candidate.spread_pct < 100 else None,
            spot_price=candidate.spot_price if candidate.spot_price > 0 else None,
            implied_move=candidate.implied_move_pct / 100 if candidate.implied_move_pct > 0 else None,
            straddle_premium=straddle_premium,
        )

        self.logger.log_non_trade(non_trade)

    def get_active_count(self) -> int:
        """Get count of active (unfilled) orders."""
        return sum(1 for o in self.active_orders.values() if o.status in ('pending', 'partial'))

    def recover_orders(self) -> int:
        """Recover active orders from database after restart.

        Looks up IBKR orders by ID and reattaches them to active_orders dict.
        Works with both legacy OrderPair and new ComboOrder formats.
        Returns count of recovered orders.
        """
        pending_trades = self.logger.get_pending_trades_with_orders()

        if not pending_trades:
            logger.info("No pending orders to recover")
            return 0

        recovered = 0
        # Get all open orders from IBKR
        open_orders = {t.order.orderId: t for t in self.ib.openTrades()}

        for trade in pending_trades:
            # For combo orders, call_order_id stores the combo order ID
            order_id = trade.call_order_id

            if not order_id:
                continue

            # Try to find matching IBKR order
            ib_trade = open_orders.get(order_id)

            # Parse strike from DB
            try:
                import json
                strikes = json.loads(trade.strikes) if trade.strikes else []
                strike = strikes[0] if strikes else 0.0
                if strike <= 0:
                    logger.warning(f"{trade.ticker}: No valid strike in DB record: {trade.strikes}")
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"{trade.ticker}: Failed to parse strikes from DB: {trade.strikes} - {e}")
                strike = 0.0

            if not ib_trade:
                # Order not found in IBKR's open orders
                # If it was already filled, leave it alone (filled orders aren't "open")
                if trade.status == 'filled':
                    logger.info(f"{trade.ticker}: Already filled, skipping recovery")
                    continue

                # Check fills history before assuming the order is truly gone
                fills = self.ib.fills()
                order_filled = any(f.execution.orderId == order_id for f in fills)

                if order_filled:
                    logger.info(f"{trade.ticker}: Order {order_id} found in fills, marking as filled")
                    self.logger.update_trade(
                        trade.trade_id,
                        status='filled',
                        notes='Recovered from fills history'
                    )
                    continue

                # Only cancel truly pending orders that aren't in IBKR
                logger.warning(f"{trade.ticker}: Order not found in IBKR (ID: {order_id}) - marking as cancelled")
                self.logger.update_trade(
                    trade.trade_id,
                    status='cancelled',
                    notes='Auto-cancelled: order not found in IBKR on recovery'
                )
                continue

            # Check status
            status = ib_trade.orderStatus.status
            if status == 'Filled':
                order_status = 'filled'
            elif status in ('Cancelled', 'Inactive'):
                order_status = 'cancelled'
            else:
                order_status = 'pending'

            combo_order = ComboOrder(
                trade_id=trade.trade_id,
                symbol=trade.ticker,
                expiry=trade.expiration,
                strike=strike,
                order_id=order_id,
                trade=ib_trade,
                status=order_status,
            )

            # Get fill price if available
            if order_status == 'filled':
                combo_order.fill_price = ib_trade.orderStatus.avgFillPrice

            self.active_orders[trade.trade_id] = combo_order
            recovered += 1

            logger.info(
                f"{trade.ticker}: Recovered order {trade.trade_id} "
                f"(status={order_status}, order_id={order_id})"
            )

        logger.info(f"Recovered {recovered} orders from database")
        return recovered


def _create_exit_combo(
    call: Option,
    put: Option,
) -> Contract:
    """Create a combo contract for selling a straddle (call + put)."""
    return create_straddle_contract(call, put, 'SELL')


async def close_position(
    ib: IB,
    trade_logger: TradeLogger,
    trade_id: str,
    symbol: str,
    expiry: str,
    strike: float,
    contracts: int,
    limit_aggression: float = 0.3,
    entry_fill_price: Optional[float] = None,
    spot_at_entry: Optional[float] = None,
) -> Optional[ExitComboOrder]:
    """
    Close an existing straddle position using separate leg orders.

    Places individual sell orders for call and put legs to avoid
    IBKR's "riskless combination orders" rejection.
    Returns ExitComboOrder for tracking, or None on failure.
    """
    # Create option contracts
    call = Option(symbol, expiry, strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, expiry, strike, 'P', 'SMART', tradingClass=symbol)

    try:
        await ib.qualifyContractsAsync(call, put)
    except Exception as e:
        logger.error(f"Could not qualify options for close: {e}")
        return None

    # Get current quotes
    call_ticker = None
    put_ticker = None
    call_bid = call_ask = put_bid = put_ask = 0
    try:
        call_ticker = ib.reqMktData(call, '', False, False)
        put_ticker = ib.reqMktData(put, '', False, False)

        # Wait for data (up to 2s)
        for _ in range(20):
            if (call_ticker.bid > 0 and call_ticker.ask > 0 and
                put_ticker.bid > 0 and put_ticker.ask > 0):
                break
            await asyncio.sleep(0.1)

        # Handle NaN values from market data
        call_bid = call_ticker.bid if call_ticker.bid and not math.isnan(call_ticker.bid) else 0
        call_ask = call_ticker.ask if call_ticker.ask and not math.isnan(call_ticker.ask) else 0
        put_bid = put_ticker.bid if put_ticker.bid and not math.isnan(put_ticker.bid) else 0
        put_ask = put_ticker.ask if put_ticker.ask and not math.isnan(put_ticker.ask) else 0
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

    if call_bid <= 0 or put_bid <= 0:
        logger.error(f"No valid bids for close")
        return None

    # Calculate limit prices for each leg (slightly below mid)
    call_mid = (call_bid + call_ask) / 2
    call_spread = call_ask - call_bid
    call_limit = round(call_mid - limit_aggression * call_spread, 2)

    put_mid = (put_bid + put_ask) / 2
    put_spread = put_ask - put_bid
    put_limit = round(put_mid - limit_aggression * put_spread, 2)

    # Combined values for logging
    straddle_bid = call_bid + put_bid
    straddle_ask = call_ask + put_ask
    straddle_mid = call_mid + put_mid
    straddle_limit = call_limit + put_limit

    # Place separate orders for each leg (avoids "riskless combination" rejection)
    call_order = LimitOrder('SELL', contracts, call_limit)
    put_order = LimitOrder('SELL', contracts, put_limit)

    try:
        call_trade = ib.placeOrder(call, call_order)
        put_trade = ib.placeOrder(put, put_order)

        # Log initial placement
        trade_logger.log_order_event(
            trade_id=trade_id,
            ib_order_id=call_trade.order.orderId,
            event='placed',
            status='PendingSubmit',
            filled=0,
            remaining=contracts,
            avg_fill_price=0.0,
            limit_price=call_limit,
            details={'type': 'exit_call', 'symbol': symbol}
        )
        trade_logger.log_order_event(
            trade_id=trade_id,
            ib_order_id=put_trade.order.orderId,
            event='placed',
            status='PendingSubmit',
            filled=0,
            remaining=contracts,
            avg_fill_price=0.0,
            limit_price=put_limit,
            details={'type': 'exit_put', 'symbol': symbol}
        )

        # Save exit order IDs for recovery
        trade_logger.update_trade(
            trade_id,
            exit_call_order_id=call_trade.order.orderId,
            exit_put_order_id=put_trade.order.orderId,
        )

    except Exception as e:
        logger.error(f"Exit order placement error: {e}")
        return None

    # Update trade log with exit info
    trade_logger.update_trade(
        trade_id,
        exit_datetime=datetime.now().isoformat(),
        exit_quoted_bid=straddle_bid,
        exit_quoted_ask=straddle_ask,
        exit_quoted_mid=straddle_mid,
        exit_limit_price=straddle_limit,
        status='exiting',
    )

    logger.info(
        f"{symbol}: Placed exit orders - "
        f"{contracts}x call @ ${call_limit:.2f}, put @ ${put_limit:.2f} "
        f"(combined: ${straddle_limit:.2f}, mid: ${straddle_mid:.2f})"
    )

    # Create exit order for tracking (stores both trades)
    exit_order = ExitComboOrder(
        trade_id=trade_id,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        order_id=call_trade.order.orderId,  # Primary order ID (call)
        trade=call_trade,  # Primary trade (call) - put tracked via put_trade
        call_conId=call.conId,
        put_conId=put.conId,
        entry_fill_price=entry_fill_price,
        spot_at_entry=spot_at_entry,
        status='pending',
    )
    # Store put trade for monitoring
    exit_order.put_trade = put_trade

    return exit_order


async def _retry_failed_exit_leg(
    ib: IB,
    exit_order: ExitComboOrder,
    failed_side: str,  # 'call' or 'put'
    use_market_order: bool = True,
) -> Optional[Trade]:
    """Retry a failed exit leg with more aggressive pricing (market order)."""
    symbol = exit_order.symbol
    expiry = exit_order.expiry
    strike = exit_order.strike
    contracts = exit_order.contracts

    # Create option contract for failed leg
    right = 'C' if failed_side == 'call' else 'P'
    option = Option(symbol, expiry, strike, right, 'SMART', tradingClass=symbol)

    try:
        await ib.qualifyContractsAsync(option)
    except Exception as e:
        logger.error(f"{symbol}: Failed to qualify {failed_side} for retry: {e}")
        return None

    # Use market order for reliability
    if use_market_order:
        order = MarketOrder('SELL', contracts)
    else:
        # Get current quote and use bid price (most aggressive limit)
        ticker = ib.reqMktData(option, '', False, False)
        for _ in range(10):
            if ticker.bid and ticker.bid > 0:
                break
            await asyncio.sleep(0.1)
        limit_price = ticker.bid if ticker.bid and ticker.bid > 0 else 0.01
        ib.cancelMktData(option)
        order = LimitOrder('SELL', contracts, limit_price)

    try:
        trade = ib.placeOrder(option, order)
        logger.info(f"{symbol}: Retry {failed_side} exit order placed ({'MKT' if use_market_order else 'LMT'})")
        return trade
    except Exception as e:
        logger.error(f"{symbol}: Failed to place retry order: {e}")
        return None


async def check_exit_fills(
    exit_orders: dict[str, ExitComboOrder],
    trade_logger: TradeLogger,
    ib: IB = None,
) -> list[ExitComboOrder]:
    """Check status of exit orders (separate call/put legs) and update trade logs.

    Returns list of fully filled exit orders (both legs filled).

    If ib is provided, fetches spot price at exit to calculate realized move.
    """
    filled = []

    for trade_id, exit_order in list(exit_orders.items()):
        if not exit_order.trade:
            continue

        # Skip already completed exits
        if exit_order.status == 'filled':
            continue

        # Process pending, submitted, and retrying orders
        # Note: 'retrying' orders need to be monitored for their retry fill
        if exit_order.status not in ('pending', 'submitted', 'retrying', 'partial'):
            # Skip cancelled, orphan, or other terminal states
            if exit_order.status != 'exiting':  # 'exiting' is also valid to monitor
                continue

        # Check call leg status
        call_status = exit_order.trade.orderStatus.status
        call_filled = exit_order.trade.orderStatus.filled
        call_avg_fill = exit_order.trade.orderStatus.avgFillPrice

        # Check put leg status
        put_status = 'Unknown'
        put_filled = 0
        put_avg_fill = 0.0
        if exit_order.put_trade:
            put_status = exit_order.put_trade.orderStatus.status
            put_filled = exit_order.put_trade.orderStatus.filled
            put_avg_fill = exit_order.put_trade.orderStatus.avgFillPrice

        # Log call leg events
        prev_call_status = getattr(exit_order, '_last_call_status', None)
        prev_call_filled = getattr(exit_order, '_last_call_filled', 0)

        if call_status != prev_call_status or call_filled > prev_call_filled:
            trade_logger.log_order_event(
                trade_id=trade_id,
                ib_order_id=exit_order.trade.order.orderId,
                event='status_update' if call_filled == prev_call_filled else 'fill',
                status=call_status,
                filled=call_filled,
                remaining=exit_order.trade.orderStatus.remaining,
                avg_fill_price=call_avg_fill,
                limit_price=exit_order.trade.order.lmtPrice,
                details={'type': 'exit_call'}
            )
            exit_order._last_call_status = call_status
            exit_order._last_call_filled = call_filled

        # Log put leg events
        if exit_order.put_trade:
            prev_put_status = getattr(exit_order, '_last_put_status', None)
            prev_put_filled = getattr(exit_order, '_last_put_filled', 0)

            if put_status != prev_put_status or put_filled > prev_put_filled:
                trade_logger.log_order_event(
                    trade_id=trade_id,
                    ib_order_id=exit_order.put_trade.order.orderId,
                    event='status_update' if put_filled == prev_put_filled else 'fill',
                    status=put_status,
                    filled=put_filled,
                    remaining=exit_order.put_trade.orderStatus.remaining,
                    avg_fill_price=put_avg_fill,
                    limit_price=exit_order.put_trade.order.lmtPrice,
                    details={'type': 'exit_put'}
                )
                exit_order._last_put_status = put_status
                exit_order._last_put_filled = put_filled

        # Check if both legs are filled
        both_filled = (call_status == 'Filled' and put_status == 'Filled')

        if both_filled and exit_order.fill_price is None:
            # Store individual fill prices
            exit_order.call_fill_price = call_avg_fill
            exit_order.put_fill_price = put_avg_fill
            exit_order.fill_price = call_avg_fill + put_avg_fill
            exit_order.status = 'filled'
            filled.append(exit_order)

            call_limit = exit_order.trade.order.lmtPrice
            put_limit = exit_order.put_trade.order.lmtPrice if exit_order.put_trade else 0
            combined_limit = call_limit + put_limit

            # Calculate P&L
            premium_received = exit_order.fill_price * exit_order.contracts * 100
            exit_pnl = None
            exit_pnl_pct = None

            if exit_order.entry_fill_price:
                premium_paid = exit_order.entry_fill_price * exit_order.contracts * 100
                exit_pnl = premium_received - premium_paid
                exit_pnl_pct = (exit_order.fill_price / exit_order.entry_fill_price - 1)

            # Fetch spot price at exit and calculate realized move
            spot_at_exit = None
            realized_move = None
            realized_move_pct = None

            if ib and ib.isConnected():
                try:
                    stock = Stock(exit_order.symbol, 'SMART', 'USD')
                    await ib.qualifyContractsAsync(stock)
                    ticker = ib.reqMktData(stock, '', False, False)

                    # Wait for price (max 2s)
                    for _ in range(20):
                        if ticker.last and ticker.last > 0:
                            break
                        if ticker.close and ticker.close > 0:
                            break
                        await asyncio.sleep(0.1)

                    spot_at_exit = ticker.marketPrice()
                    if (spot_at_exit is None or math.isnan(spot_at_exit) or spot_at_exit <= 0):
                        spot_at_exit = ticker.last if ticker.last and ticker.last > 0 else ticker.close

                    ib.cancelMktData(stock)

                    if spot_at_exit and spot_at_exit > 0 and exit_order.spot_at_entry:
                        realized_move = abs(spot_at_exit - exit_order.spot_at_entry)
                        realized_move_pct = realized_move / exit_order.spot_at_entry

                except Exception as e:
                    logger.warning(f"{exit_order.symbol}: Failed to fetch spot at exit: {e}")

            trade_logger.update_trade(
                trade_id,
                status='exited',
                exit_fill_price=exit_order.fill_price,
                # Exit slippage: limit - fill (positive = sold below limit = bad)
                # Consistent with entry: positive slippage = worse execution
                exit_slippage=combined_limit - exit_order.fill_price if combined_limit else None,
                exit_pnl=exit_pnl,
                exit_pnl_pct=exit_pnl_pct,
                spot_at_exit=spot_at_exit,
                realized_move=realized_move,
                realized_move_pct=realized_move_pct,
            )

            pnl_str = f", P&L: ${exit_pnl:.2f}" if exit_pnl else ""
            move_str = f", move: {realized_move_pct*100:.1f}%" if realized_move_pct else ""
            logger.info(
                f"{exit_order.symbol}: EXIT COMPLETE - "
                f"Call: ${call_avg_fill:.2f}, Put: ${put_avg_fill:.2f}, "
                f"Combined: ${exit_order.fill_price:.2f}{pnl_str}{move_str}"
            )

        # Handle partial exits (orphan leg detection)
        call_filled = call_status == 'Filled'
        put_filled = put_status == 'Filled'
        call_failed = call_status in ('Cancelled', 'Inactive')
        put_failed = put_status in ('Cancelled', 'Inactive')

        if (call_filled and put_failed) or (put_filled and call_failed):
            # ORPHAN DETECTED - one leg closed, other failed
            failed_side = 'put' if put_failed else 'call'
            logger.warning(f"{exit_order.symbol}: ORPHAN LEG - {failed_side} failed while other filled, retrying...")

            if ib and ib.isConnected():
                retry_trade = await _retry_failed_exit_leg(ib, exit_order, failed_side)
                if retry_trade:
                    # Update tracking to monitor retry
                    if failed_side == 'put':
                        exit_order.put_trade = retry_trade
                        exit_order._last_put_status = None  # Reset status tracking
                        exit_order._last_put_filled = 0
                    else:
                        exit_order.trade = retry_trade
                        exit_order._last_call_status = None
                        exit_order._last_call_filled = 0
                    exit_order.status = 'retrying'
                else:
                    exit_order.status = 'orphan'
                    logger.error(f"{exit_order.symbol}: Failed to retry {failed_side} exit - MANUAL INTERVENTION REQUIRED")
            else:
                exit_order.status = 'orphan'
                logger.error(f"{exit_order.symbol}: Cannot retry - IB not connected - MANUAL INTERVENTION REQUIRED")

        # Handle both legs cancelled
        elif call_failed and put_failed:
            exit_order.status = 'cancelled'
            logger.warning(f"{exit_order.symbol}: Exit order cancelled/inactive (call={call_status}, put={put_status})")

    return filled


async def reprice_exit_to_bid(
    exit_order: ExitComboOrder,
    ib: IB,
    trade_logger: TradeLogger,
) -> Optional[ExitComboOrder]:
    """
    Cancel existing exit orders and resubmit at current bid price.

    Used when initial limit orders haven't filled and we need more aggressive pricing.
    Returns updated ExitComboOrder, or None on failure.
    """
    symbol = exit_order.symbol
    logger.info(f"{symbol}: Repricing exit orders to bid...")

    # Cancel existing orders
    cancelled = False
    if exit_order.trade and exit_order.trade.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive'):
        ib.cancelOrder(exit_order.trade.order)
        cancelled = True
    if exit_order.put_trade and exit_order.put_trade.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive'):
        ib.cancelOrder(exit_order.put_trade.order)
        cancelled = True

    if cancelled:
        await asyncio.sleep(0.5)  # Wait for cancellation to process

    # Get fresh quotes
    call = Option(symbol, exit_order.expiry, exit_order.strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, exit_order.expiry, exit_order.strike, 'P', 'SMART', tradingClass=symbol)

    try:
        await ib.qualifyContractsAsync(call, put)
    except Exception as e:
        logger.error(f"{symbol}: Failed to qualify contracts for reprice: {e}")
        return None

    call_ticker = None
    put_ticker = None
    call_bid = put_bid = 0
    try:
        call_ticker = ib.reqMktData(call, '', False, False)
        put_ticker = ib.reqMktData(put, '', False, False)

        for _ in range(20):
            if (call_ticker.bid > 0 and put_ticker.bid > 0):
                break
            await asyncio.sleep(0.1)

        # Handle NaN values from market data
        call_bid = call_ticker.bid if call_ticker.bid and not math.isnan(call_ticker.bid) else 0
        put_bid = put_ticker.bid if put_ticker.bid and not math.isnan(put_ticker.bid) else 0
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

    if call_bid <= 0 or put_bid <= 0:
        logger.error(f"{symbol}: No valid bids for reprice (call={call_bid}, put={put_bid})")
        return None

    # Place new orders at bid
    call_order = LimitOrder('SELL', exit_order.contracts, call_bid)
    put_order = LimitOrder('SELL', exit_order.contracts, put_bid)

    try:
        call_trade = ib.placeOrder(call, call_order)
        put_trade = ib.placeOrder(put, put_order)

        straddle_bid = call_bid + put_bid
        logger.info(
            f"{symbol}: Repriced exit to bid - "
            f"Call ${call_bid:.2f}, Put ${put_bid:.2f}, Combined ${straddle_bid:.2f}"
        )

        # Update trade_logger with new order IDs
        trade_logger.update_trade(
            exit_order.trade_id,
            exit_call_order_id=call_trade.order.orderId,
            exit_put_order_id=put_trade.order.orderId,
            exit_limit_price=straddle_bid,
        )

        # Return updated ExitComboOrder (preserve conIds from original)
        return ExitComboOrder(
            trade_id=exit_order.trade_id,
            symbol=symbol,
            expiry=exit_order.expiry,
            strike=exit_order.strike,
            contracts=exit_order.contracts,
            order_id=call_trade.order.orderId,
            trade=call_trade,
            put_trade=put_trade,
            call_conId=exit_order.call_conId,
            put_conId=exit_order.put_conId,
            entry_fill_price=exit_order.entry_fill_price,
            spot_at_entry=exit_order.spot_at_entry,
            status='pending',
        )

    except Exception as e:
        logger.error(f"{symbol}: Failed to place repriced exit orders: {e}")
        return None


async def convert_exit_to_market(
    exit_order: ExitComboOrder,
    ib: IB,
    trade_logger: TradeLogger,
) -> Optional[ExitComboOrder]:
    """
    Cancel limit exit orders and submit market orders.

    Used as last resort when limit orders haven't filled and market close is imminent.
    Returns updated ExitComboOrder, or None on failure.
    """
    symbol = exit_order.symbol
    logger.warning(f"{symbol}: Converting exit to MARKET ORDER (last resort)...")

    # Cancel existing orders
    cancelled = False
    if exit_order.trade and exit_order.trade.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive'):
        ib.cancelOrder(exit_order.trade.order)
        cancelled = True
    if exit_order.put_trade and exit_order.put_trade.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive'):
        ib.cancelOrder(exit_order.put_trade.order)
        cancelled = True

    if cancelled:
        await asyncio.sleep(0.5)  # Wait for cancellation to process

    # Create option contracts
    call = Option(symbol, exit_order.expiry, exit_order.strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, exit_order.expiry, exit_order.strike, 'P', 'SMART', tradingClass=symbol)

    try:
        await ib.qualifyContractsAsync(call, put)
    except Exception as e:
        logger.error(f"{symbol}: Failed to qualify contracts for market order: {e}")
        return None

    # Place market orders
    call_order = MarketOrder('SELL', exit_order.contracts)
    put_order = MarketOrder('SELL', exit_order.contracts)

    try:
        call_trade = ib.placeOrder(call, call_order)
        put_trade = ib.placeOrder(put, put_order)

        logger.warning(f"{symbol}: Market exit orders placed (call={call_trade.order.orderId}, put={put_trade.order.orderId})")

        # Update trade_logger with new order IDs
        trade_logger.update_trade(
            exit_order.trade_id,
            exit_call_order_id=call_trade.order.orderId,
            exit_put_order_id=put_trade.order.orderId,
            notes="Converted to market order at close",
        )

        # Return updated ExitComboOrder (preserve conIds from original)
        return ExitComboOrder(
            trade_id=exit_order.trade_id,
            symbol=symbol,
            expiry=exit_order.expiry,
            strike=exit_order.strike,
            contracts=exit_order.contracts,
            order_id=call_trade.order.orderId,
            trade=call_trade,
            put_trade=put_trade,
            call_conId=exit_order.call_conId,
            put_conId=exit_order.put_conId,
            entry_fill_price=exit_order.entry_fill_price,
            spot_at_entry=exit_order.spot_at_entry,
            status='pending',
        )

    except Exception as e:
        logger.error(f"{symbol}: Failed to place market exit orders: {e}")
        return None


async def close_position_market(
    ib: IB,
    trade_logger: TradeLogger,
    trade_id: str,
    symbol: str,
    expiry: str,
    strike: float,
    contracts: int,
    entry_fill_price: Optional[float] = None,
    spot_at_entry: Optional[float] = None,
) -> Optional[ExitComboOrder]:
    """
    Close position using market orders - no quotes required.

    Use this for force exits at market close or when quotes are unavailable.
    Returns ExitComboOrder for tracking, or None on failure.
    """
    # Create option contracts
    call = Option(symbol, expiry, strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, expiry, strike, 'P', 'SMART', tradingClass=symbol)

    try:
        await ib.qualifyContractsAsync(call, put)
    except Exception as e:
        logger.error(f"{symbol}: Could not qualify options for market close: {e}")
        return None

    # Place market orders (no quotes needed)
    call_order = MarketOrder('SELL', contracts)
    put_order = MarketOrder('SELL', contracts)

    try:
        call_trade = ib.placeOrder(call, call_order)
        put_trade = ib.placeOrder(put, put_order)

        logger.warning(
            f"{symbol}: Market exit orders placed "
            f"(call={call_trade.order.orderId}, put={put_trade.order.orderId})"
        )

        # Update trade logger with new order IDs
        trade_logger.update_trade(
            trade_id,
            exit_call_order_id=call_trade.order.orderId,
            exit_put_order_id=put_trade.order.orderId,
            notes="Force market exit",
        )

        # Log order events
        trade_logger.log_order_event(
            trade_id=trade_id,
            ib_order_id=call_trade.order.orderId,
            event='placed',
            status=call_trade.orderStatus.status,
            filled=0,
            remaining=call_trade.orderStatus.remaining,
            avg_fill_price=0.0,
            limit_price=0,  # Market order
        )
        trade_logger.log_order_event(
            trade_id=trade_id,
            ib_order_id=put_trade.order.orderId,
            event='placed',
            status=put_trade.orderStatus.status,
            filled=0,
            remaining=put_trade.orderStatus.remaining,
            avg_fill_price=0.0,
            limit_price=0,  # Market order
        )

        return ExitComboOrder(
            trade_id=trade_id,
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            contracts=contracts,
            order_id=call_trade.order.orderId,
            trade=call_trade,
            put_trade=put_trade,
            entry_fill_price=entry_fill_price,
            spot_at_entry=spot_at_entry,
            status='pending',
        )

    except Exception as e:
        logger.error(f"{symbol}: Failed to place market exit orders: {e}")
        return None
