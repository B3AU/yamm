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
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ib_insync import IB, Option, LimitOrder, Trade, Contract, ComboLeg

from trading.earnings.screener import ScreenedCandidate
from trading.earnings.logging import (
    TradeLog, NonTradeLog, TradeLogger,
    generate_trade_id, generate_log_id
)

logger = logging.getLogger(__name__)


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

    status: str = 'pending'  # pending, filled, cancelled


@dataclass
class ExitComboOrder:
    """Tracks exit combo order for a straddle position."""
    trade_id: str
    symbol: str
    expiry: str
    strike: float
    contracts: int

    order_id: Optional[int] = None
    trade: Optional[Trade] = None

    call_conId: Optional[int] = None
    put_conId: Optional[int] = None

    fill_price: Optional[float] = None
    entry_fill_price: Optional[float] = None  # For P&L calculation

    status: str = 'pending'  # pending, filled


class Phase0Executor:
    """Executes Phase 0 validation trades using combo orders."""

    def __init__(
        self,
        ib: IB,
        trade_logger: TradeLogger,
        max_contracts: int = 1,  # Start with 1 contract
        limit_aggression: float = 0.3,  # How much above mid to place limit
    ):
        self.ib = ib
        self.logger = trade_logger
        self.max_contracts = max_contracts
        self.limit_aggression = limit_aggression
        self.active_orders: dict[str, ComboOrder] = {}

    def _create_straddle_combo(
        self,
        call: Option,
        put: Option,
    ) -> Optional[Contract]:
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
                action='BUY',
                exchange='SMART',
            ),
            ComboLeg(
                conId=put.conId,
                ratio=1,
                action='BUY',
                exchange='SMART',
            ),
        ]

        return combo

    async def place_straddle(
        self,
        candidate: ScreenedCandidate,
        contracts: int = None,
    ) -> Optional[ComboOrder]:
        """
        Place a straddle combo order (buy call + put atomically).

        Uses a single combo/BAG order to ensure both legs fill together,
        eliminating orphan leg risk.
        """
        contracts = contracts or self.max_contracts

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

        # Log the trade entry
        trade_log = TradeLog(
            trade_id=trade_id,
            ticker=candidate.symbol,
            earnings_date=str(candidate.earnings_date),
            earnings_timing=candidate.timing,
            entry_datetime=datetime.now().isoformat(),
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
            predicted_q75=candidate.pred_q75,
            edge_q75=candidate.edge_q75,
            implied_move=candidate.implied_move_pct / 100,
            spot_at_entry=candidate.spot_price,
            status='pending',
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

    def check_fills(self) -> list[ComboOrder]:
        """Check status of all active combo orders and update logs."""
        filled = []

        for trade_id, combo_order in list(self.active_orders.items()):
            if not combo_order.trade:
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

                logger.info(
                    f"{combo_order.symbol}: Straddle filled @ ${combo_order.fill_price:.2f}"
                )

                # Update trade log
                self.logger.update_trade(
                    trade_id,
                    status='filled',
                    entry_fill_price=combo_order.fill_price,
                    entry_fill_time=datetime.now().isoformat(),
                    entry_slippage=combo_order.fill_price - combo_order.trade.order.lmtPrice,
                    premium_paid=combo_order.fill_price * filled_qty * 100,  # Use actual filled quantity
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
            except (json.JSONDecodeError, IndexError):
                strike = 0.0

            if not ib_trade:
                # Order not found in IBKR's open orders
                # If it was already filled, leave it alone (filled orders aren't "open")
                if trade.status == 'filled':
                    logger.info(f"{trade.ticker}: Already filled, skipping recovery")
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
    combo = Contract()
    combo.symbol = call.symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'

    combo.comboLegs = [
        ComboLeg(
            conId=call.conId,
            ratio=1,
            action='SELL',
            exchange='SMART',
        ),
        ComboLeg(
            conId=put.conId,
            ratio=1,
            action='SELL',
            exchange='SMART',
        ),
    ]

    return combo


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
) -> Optional[ExitComboOrder]:
    """
    Close an existing straddle position using a combo order.

    Places a single combo sell order for both legs atomically.
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
    call_ticker = ib.reqMktData(call, '', False, False)
    put_ticker = ib.reqMktData(put, '', False, False)

    # Wait for data (up to 2s)
    for _ in range(20):
        if (call_ticker.bid > 0 and call_ticker.ask > 0 and
            put_ticker.bid > 0 and put_ticker.ask > 0):
            break
        await asyncio.sleep(0.1)

    call_bid = call_ticker.bid if call_ticker.bid == call_ticker.bid else 0
    call_ask = call_ticker.ask if call_ticker.ask == call_ticker.ask else 0
    put_bid = put_ticker.bid if put_ticker.bid == put_ticker.bid else 0
    put_ask = put_ticker.ask if put_ticker.ask == put_ticker.ask else 0

    ib.cancelMktData(call)
    ib.cancelMktData(put)

    if call_bid <= 0 or put_bid <= 0:
        logger.error(f"No valid bids for close")
        return None

    # Calculate combined straddle limit price (for selling: below mid)
    straddle_bid = call_bid + put_bid
    straddle_ask = call_ask + put_ask
    straddle_mid = (straddle_bid + straddle_ask) / 2
    straddle_spread = straddle_ask - straddle_bid

    # For selling, go slightly below mid
    straddle_limit = round(straddle_mid - limit_aggression * straddle_spread, 2)

    # Create combo contract for exit
    combo = _create_exit_combo(call, put)

    # Place combo sell order
    order = LimitOrder('SELL', contracts, straddle_limit)

    try:
        trade = ib.placeOrder(combo, order)

        # Log initial placement
        trade_logger.log_order_event(
            trade_id=trade_id,
            ib_order_id=trade.order.orderId,
            event='placed',
            status='PendingSubmit',
            filled=0,
            remaining=contracts,
            avg_fill_price=0.0,
            limit_price=straddle_limit,
            details={'type': 'exit', 'symbol': symbol}
        )

        # Save exit order ID for recovery
        trade_logger.update_trade(
            trade_id,
            exit_call_order_id=trade.order.orderId
        )

    except Exception as e:
        logger.error(f"Exit combo order placement error: {e}")
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
        f"{symbol}: Placed exit COMBO order - "
        f"{contracts}x @ ${straddle_limit:.2f} (mid: ${straddle_mid:.2f})"
    )

    # Create exit combo order for tracking
    exit_order = ExitComboOrder(
        trade_id=trade_id,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        order_id=trade.order.orderId,
        trade=trade,
        call_conId=call.conId,
        put_conId=put.conId,
        entry_fill_price=entry_fill_price,
        status='pending',
    )

    return exit_order


def check_exit_fills(
    exit_orders: dict[str, ExitComboOrder],
    trade_logger: TradeLogger,
) -> list[ExitComboOrder]:
    """Check status of exit combo orders and update trade logs.

    Returns list of filled exit orders.
    """
    filled = []

    for trade_id, exit_order in list(exit_orders.items()):
        if not exit_order.trade:
            continue

        status = exit_order.trade.orderStatus.status
        filled_qty = exit_order.trade.orderStatus.filled
        remaining_qty = exit_order.trade.orderStatus.remaining
        avg_fill_price = exit_order.trade.orderStatus.avgFillPrice
        last_fill_price = exit_order.trade.orderStatus.lastFillPrice # Note: IB doesn't always populate this for combos

        # Log granular event
        prev_status = getattr(exit_order, '_last_log_status', None)
        prev_filled = getattr(exit_order, '_last_log_filled', 0)

        if status != prev_status or filled_qty > prev_filled:
            trade_logger.log_order_event(
                trade_id=trade_id,
                ib_order_id=exit_order.trade.order.orderId,
                event='status_update' if filled_qty == prev_filled else 'fill',
                status=status,
                filled=filled_qty,
                remaining=remaining_qty,
                avg_fill_price=avg_fill_price,
                limit_price=exit_order.trade.order.lmtPrice,
                last_fill_price=last_fill_price,
                last_fill_qty=filled_qty - prev_filled if filled_qty > prev_filled else 0,
                details={'type': 'exit'}
            )
            exit_order._last_log_status = status
            exit_order._last_log_filled = filled_qty

        if status == 'Filled' and exit_order.fill_price is None:
            exit_order.fill_price = avg_fill_price
            exit_order.status = 'filled'
            filled.append(exit_order)

            exit_limit = exit_order.trade.order.lmtPrice

            # Calculate P&L
            premium_received = exit_order.fill_price * exit_order.contracts * 100
            exit_pnl = None
            exit_pnl_pct = None

            if exit_order.entry_fill_price:
                premium_paid = exit_order.entry_fill_price * exit_order.contracts * 100
                exit_pnl = premium_received - premium_paid
                exit_pnl_pct = (exit_order.fill_price / exit_order.entry_fill_price - 1)

            trade_logger.update_trade(
                trade_id,
                status='exited',
                exit_fill_price=exit_order.fill_price,
                exit_slippage=exit_order.fill_price - exit_limit if exit_limit else None,
                exit_pnl=exit_pnl,
                exit_pnl_pct=exit_pnl_pct,
            )

            pnl_str = f", P&L: ${exit_pnl:.2f}" if exit_pnl else ""
            logger.info(
                f"{exit_order.symbol}: EXIT COMPLETE - "
                f"Fill: ${exit_order.fill_price:.2f}{pnl_str}"
            )

        elif status in ('Cancelled', 'Inactive'):
            exit_order.status = 'cancelled'
            logger.warning(f"{exit_order.symbol}: Exit order cancelled/inactive")

    return filled
