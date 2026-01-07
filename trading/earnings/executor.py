"""Phase 0 executor - places and manages straddle orders.

Handles:
- Order placement with limit pricing strategy
- Fill monitoring and logging
- Position tracking
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ib_insync import IB, Option, LimitOrder, Trade

from trading.earnings.screener import ScreenedCandidate
from trading.earnings.logging import (
    TradeLog, NonTradeLog, TradeLogger,
    generate_trade_id, generate_log_id
)

logger = logging.getLogger(__name__)


@dataclass
class OrderPair:
    """Tracks a straddle order (call + put)."""
    trade_id: str
    symbol: str
    expiry: str
    strike: float

    call_order_id: Optional[int] = None
    put_order_id: Optional[int] = None

    call_trade: Optional[Trade] = None
    put_trade: Optional[Trade] = None

    call_fill_price: Optional[float] = None
    put_fill_price: Optional[float] = None

    status: str = 'pending'  # pending, partial, filled, cancelled


@dataclass
class ExitOrderPair:
    """Tracks exit orders for a straddle position."""
    trade_id: str
    symbol: str
    expiry: str
    strike: float
    contracts: int

    call_order_id: Optional[int] = None
    put_order_id: Optional[int] = None

    call_trade: Optional[Trade] = None
    put_trade: Optional[Trade] = None

    call_fill_price: Optional[float] = None
    put_fill_price: Optional[float] = None

    entry_fill_price: Optional[float] = None  # For P&L calculation

    status: str = 'pending'  # pending, partial, filled


class Phase0Executor:
    """Executes Phase 0 validation trades."""

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
        self.active_orders: dict[str, OrderPair] = {}

    def place_straddle(
        self,
        candidate: ScreenedCandidate,
        contracts: int = None,
    ) -> Optional[OrderPair]:
        """
        Place a straddle order (buy call + put).

        Uses limit orders at mid + aggression * spread.
        """
        contracts = contracts or self.max_contracts

        # Calculate limit prices
        call_mid = (candidate.call_bid + candidate.call_ask) / 2
        put_mid = (candidate.put_bid + candidate.put_ask) / 2
        call_spread = candidate.call_ask - candidate.call_bid
        put_spread = candidate.put_ask - candidate.put_bid

        call_limit = round(call_mid + self.limit_aggression * call_spread, 2)
        put_limit = round(put_mid + self.limit_aggression * put_spread, 2)

        # Create option contracts
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
            qualified = self.ib.qualifyContracts(call, put)
            if len(qualified) < 2:
                logger.error(f"{candidate.symbol}: Could not qualify options for order")
                return None
        except Exception as e:
            logger.error(f"{candidate.symbol}: Qualification error: {e}")
            return None

        # Generate trade ID
        trade_id = generate_trade_id(candidate.symbol, str(candidate.earnings_date))

        # Log the trade entry
        trade_log = TradeLog(
            trade_id=trade_id,
            ticker=candidate.symbol,
            earnings_date=str(candidate.earnings_date),
            earnings_timing=candidate.timing,
            entry_datetime=datetime.now().isoformat(),
            entry_quoted_bid=candidate.call_bid + candidate.put_bid,
            entry_quoted_ask=candidate.call_ask + candidate.put_ask,
            entry_quoted_mid=candidate.straddle_mid,
            entry_limit_price=call_limit + put_limit,
            structure='straddle',
            strikes=str([candidate.atm_strike]),
            expiration=candidate.expiry,
            contracts=contracts,
            premium_paid=0,  # Updated on fill
            max_loss=(call_limit + put_limit) * contracts * 100,
            predicted_q75=candidate.pred_q75,
            edge_q75=candidate.edge_q75,
            implied_move=candidate.implied_move_pct / 100,
            spot_at_entry=candidate.spot_price,
            status='pending',
        )
        self.logger.log_trade(trade_log)

        # Place orders
        call_order = LimitOrder('BUY', contracts, call_limit)
        put_order = LimitOrder('BUY', contracts, put_limit)

        try:
            call_trade = self.ib.placeOrder(call, call_order)
            put_trade = self.ib.placeOrder(put, put_order)
        except Exception as e:
            logger.error(f"{candidate.symbol}: Order placement error: {e}")
            self.logger.update_trade(trade_id, status='error', notes=str(e))
            return None

        order_pair = OrderPair(
            trade_id=trade_id,
            symbol=candidate.symbol,
            expiry=candidate.expiry,
            strike=candidate.atm_strike,
            call_order_id=call_trade.order.orderId,
            put_order_id=put_trade.order.orderId,
            call_trade=call_trade,
            put_trade=put_trade,
            status='pending',
        )

        self.active_orders[trade_id] = order_pair

        # Save order IDs to DB for recovery after restart
        self.logger.update_trade(
            trade_id,
            call_order_id=call_trade.order.orderId,
            put_order_id=put_trade.order.orderId,
        )

        logger.info(
            f"{candidate.symbol}: Placed straddle order - "
            f"Call {contracts}x @ ${call_limit:.2f}, Put {contracts}x @ ${put_limit:.2f}"
        )

        return order_pair

    def check_fills(self) -> list[OrderPair]:
        """Check status of all active orders and update logs."""
        filled = []

        for trade_id, order_pair in list(self.active_orders.items()):
            call_status = order_pair.call_trade.orderStatus.status if order_pair.call_trade else 'Unknown'
            put_status = order_pair.put_trade.orderStatus.status if order_pair.put_trade else 'Unknown'

            # Get fill quantities
            call_filled_qty = order_pair.call_trade.orderStatus.filled if order_pair.call_trade else 0
            put_filled_qty = order_pair.put_trade.orderStatus.filled if order_pair.put_trade else 0
            call_total_qty = order_pair.call_trade.order.totalQuantity if order_pair.call_trade else 0
            put_total_qty = order_pair.put_trade.order.totalQuantity if order_pair.put_trade else 0

            # Check for fills
            call_filled = call_status == 'Filled'
            put_filled = put_status == 'Filled'

            if call_filled and order_pair.call_fill_price is None:
                order_pair.call_fill_price = order_pair.call_trade.orderStatus.avgFillPrice
                logger.info(f"{order_pair.symbol}: Call filled @ ${order_pair.call_fill_price:.2f}")

            if put_filled and order_pair.put_fill_price is None:
                order_pair.put_fill_price = order_pair.put_trade.orderStatus.avgFillPrice
                logger.info(f"{order_pair.symbol}: Put filled @ ${order_pair.put_fill_price:.2f}")

            # Update status
            if call_filled and put_filled:
                order_pair.status = 'filled'
                filled.append(order_pair)

                # Update trade log
                total_fill = order_pair.call_fill_price + order_pair.put_fill_price
                self.logger.update_trade(
                    trade_id,
                    status='filled',
                    entry_fill_price=total_fill,
                    entry_fill_time=datetime.now().isoformat(),
                    entry_slippage=total_fill - (
                        order_pair.call_trade.order.lmtPrice + order_pair.put_trade.order.lmtPrice
                    ) if order_pair.call_trade and order_pair.put_trade else None,
                    premium_paid=total_fill * self.max_contracts * 100,
                )

            elif call_filled or put_filled:
                order_pair.status = 'partial'
                # Log partial fill details
                logger.warning(
                    f"{order_pair.symbol}: PARTIAL FILL - "
                    f"Call: {int(call_filled_qty)}/{int(call_total_qty)} ({call_status}), "
                    f"Put: {int(put_filled_qty)}/{int(put_total_qty)} ({put_status})"
                )
            elif call_filled_qty > 0 or put_filled_qty > 0:
                # Partial quantity fill (e.g., 1 of 5 contracts)
                order_pair.status = 'partial'
                logger.warning(
                    f"{order_pair.symbol}: PARTIAL QUANTITY - "
                    f"Call: {int(call_filled_qty)}/{int(call_total_qty)}, "
                    f"Put: {int(put_filled_qty)}/{int(put_total_qty)}"
                )
            elif call_status in ('Cancelled', 'Inactive') and put_status in ('Cancelled', 'Inactive'):
                order_pair.status = 'cancelled'
                self.logger.update_trade(trade_id, status='cancelled')

        return filled

    def get_partial_fills(self) -> list[OrderPair]:
        """Get all orders with partial fills (orphan risk)."""
        partials = []
        for order_pair in self.active_orders.values():
            if order_pair.status == 'partial':
                partials.append(order_pair)
                continue

            # Also check for partial quantity fills
            if order_pair.call_trade and order_pair.put_trade:
                call_filled = order_pair.call_trade.orderStatus.filled
                put_filled = order_pair.put_trade.orderStatus.filled
                call_total = order_pair.call_trade.order.totalQuantity
                put_total = order_pair.put_trade.order.totalQuantity

                # Partial if any fills but not complete
                if (call_filled > 0 or put_filled > 0) and (call_filled < call_total or put_filled < put_total):
                    partials.append(order_pair)

        return partials

    def cancel_unfilled_orders(self, trade_id: str) -> dict:
        """Cancel only the unfilled portion of an order pair.

        Returns dict with what was cancelled.
        """
        if trade_id not in self.active_orders:
            return {'cancelled': False, 'reason': 'not found'}

        order_pair = self.active_orders[trade_id]
        cancelled = {'call': False, 'put': False}

        try:
            # Cancel call if not fully filled
            if order_pair.call_trade:
                call_status = order_pair.call_trade.orderStatus.status
                if call_status not in ('Filled', 'Cancelled', 'Inactive'):
                    self.ib.cancelOrder(order_pair.call_trade.order)
                    cancelled['call'] = True
                    logger.info(f"{order_pair.symbol}: Cancelled unfilled call order")

            # Cancel put if not fully filled
            if order_pair.put_trade:
                put_status = order_pair.put_trade.orderStatus.status
                if put_status not in ('Filled', 'Cancelled', 'Inactive'):
                    self.ib.cancelOrder(order_pair.put_trade.order)
                    cancelled['put'] = True
                    logger.info(f"{order_pair.symbol}: Cancelled unfilled put order")

        except Exception as e:
            logger.error(f"Cancel error for {trade_id}: {e}")
            return {'cancelled': False, 'reason': str(e)}

        # Update status based on what we have
        call_filled_qty = order_pair.call_trade.orderStatus.filled if order_pair.call_trade else 0
        put_filled_qty = order_pair.put_trade.orderStatus.filled if order_pair.put_trade else 0

        if call_filled_qty > 0 or put_filled_qty > 0:
            # We have some fills - mark as partial
            self.logger.update_trade(
                trade_id,
                status='partial',
                notes=f"Partial fill: {int(call_filled_qty)} calls, {int(put_filled_qty)} puts. Unfilled orders cancelled."
            )
        else:
            # No fills at all
            order_pair.status = 'cancelled'
            self.logger.update_trade(trade_id, status='cancelled', notes='No fills before close')

        return {'cancelled': True, **cancelled}

    def cancel_order(self, trade_id: str) -> bool:
        """Cancel an active order pair."""
        if trade_id not in self.active_orders:
            return False

        order_pair = self.active_orders[trade_id]

        try:
            if order_pair.call_trade:
                self.ib.cancelOrder(order_pair.call_trade.order)
            if order_pair.put_trade:
                self.ib.cancelOrder(order_pair.put_trade.order)
        except Exception as e:
            logger.error(f"Cancel error for {trade_id}: {e}")
            return False

        order_pair.status = 'cancelled'
        self.logger.update_trade(trade_id, status='cancelled')

        return True

    def cancel_all(self):
        """Cancel all active orders."""
        for trade_id in list(self.active_orders.keys()):
            self.cancel_order(trade_id)

    def log_non_trade(self, candidate: ScreenedCandidate):
        """Log a candidate we passed on."""
        log_id = generate_log_id(candidate.symbol)

        non_trade = NonTradeLog(
            log_id=log_id,
            ticker=candidate.symbol,
            earnings_date=str(candidate.earnings_date),
            earnings_timing=candidate.timing,
            log_datetime=datetime.now().isoformat(),
            rejection_reason=candidate.rejection_reason or 'unknown',
            quoted_bid=candidate.call_bid + candidate.put_bid if candidate.call_bid else None,
            quoted_ask=candidate.call_ask + candidate.put_ask if candidate.call_ask else None,
            quoted_spread_pct=candidate.spread_pct if candidate.spread_pct < 100 else None,
            spot_price=candidate.spot_price if candidate.spot_price > 0 else None,
            implied_move=candidate.implied_move_pct / 100 if candidate.implied_move_pct > 0 else None,
        )

        self.logger.log_non_trade(non_trade)

    def get_active_count(self) -> int:
        """Get count of active (unfilled) orders."""
        return sum(1 for o in self.active_orders.values() if o.status in ('pending', 'partial'))

    def recover_orders(self) -> int:
        """Recover active orders from database after restart.

        Looks up IBKR orders by ID and reattaches them to active_orders dict.
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
            call_order_id = trade.call_order_id
            put_order_id = trade.put_order_id

            if not call_order_id and not put_order_id:
                continue

            # Try to find matching IBKR orders
            call_trade = open_orders.get(call_order_id)
            put_trade = open_orders.get(put_order_id)

            # Parse strike from DB
            try:
                import json
                strikes = json.loads(trade.strikes) if trade.strikes else []
                strike = strikes[0] if strikes else 0.0
            except (json.JSONDecodeError, IndexError):
                strike = 0.0

            # Check if orders are still open or already filled
            call_filled = False
            put_filled = False

            if call_trade:
                call_filled = call_trade.orderStatus.status == 'Filled'
            if put_trade:
                put_filled = put_trade.orderStatus.status == 'Filled'

            # Determine status
            if call_filled and put_filled:
                status = 'filled'
            elif call_filled or put_filled:
                status = 'partial'
            elif call_trade or put_trade:
                status = 'pending'
            else:
                # Orders not found - may have been cancelled or filled and position closed
                logger.warning(f"{trade.ticker}: Orders not found in IBKR (IDs: {call_order_id}, {put_order_id})")
                continue

            order_pair = OrderPair(
                trade_id=trade.trade_id,
                symbol=trade.ticker,
                expiry=trade.expiration,
                strike=strike,
                call_order_id=call_order_id,
                put_order_id=put_order_id,
                call_trade=call_trade,
                put_trade=put_trade,
                status=status,
            )

            # Get fill prices if available
            if call_trade and call_filled:
                order_pair.call_fill_price = call_trade.orderStatus.avgFillPrice
            if put_trade and put_filled:
                order_pair.put_fill_price = put_trade.orderStatus.avgFillPrice

            self.active_orders[trade.trade_id] = order_pair
            recovered += 1

            logger.info(
                f"{trade.ticker}: Recovered order {trade.trade_id} "
                f"(status={status}, call_id={call_order_id}, put_id={put_order_id})"
            )

        logger.info(f"Recovered {recovered} orders from database")
        return recovered


def close_position(
    ib: IB,
    trade_logger: TradeLogger,
    trade_id: str,
    symbol: str,
    expiry: str,
    strike: float,
    contracts: int,
    limit_aggression: float = 0.3,
    entry_fill_price: Optional[float] = None,
) -> Optional[ExitOrderPair]:
    """
    Close an existing straddle position.

    Places sell orders for both legs.
    Returns ExitOrderPair for tracking, or None on failure.
    """
    # Create option contracts
    call = Option(symbol, expiry, strike, 'C', 'SMART', tradingClass=symbol)
    put = Option(symbol, expiry, strike, 'P', 'SMART', tradingClass=symbol)

    try:
        ib.qualifyContracts(call, put)
    except Exception as e:
        logger.error(f"Could not qualify options for close: {e}")
        return None

    # Get current quotes
    call_ticker = ib.reqMktData(call, '', False, False)
    put_ticker = ib.reqMktData(put, '', False, False)
    ib.sleep(2)

    call_bid = call_ticker.bid if call_ticker.bid == call_ticker.bid else 0
    call_ask = call_ticker.ask if call_ticker.ask == call_ticker.ask else 0
    put_bid = put_ticker.bid if put_ticker.bid == put_ticker.bid else 0
    put_ask = put_ticker.ask if put_ticker.ask == put_ticker.ask else 0

    ib.cancelMktData(call)
    ib.cancelMktData(put)

    if call_bid <= 0 or put_bid <= 0:
        logger.error(f"No valid bids for close")
        return None

    # Calculate limit prices (sell slightly below mid)
    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2
    call_spread = call_ask - call_bid
    put_spread = put_ask - put_bid

    call_limit = round(call_mid - limit_aggression * call_spread, 2)
    put_limit = round(put_mid - limit_aggression * put_spread, 2)

    # Place sell orders
    call_order = LimitOrder('SELL', contracts, call_limit)
    put_order = LimitOrder('SELL', contracts, put_limit)

    try:
        call_trade = ib.placeOrder(call, call_order)
        put_trade = ib.placeOrder(put, put_order)
    except Exception as e:
        logger.error(f"Exit order placement error: {e}")
        return None

    # Update trade log with exit info
    trade_logger.update_trade(
        trade_id,
        exit_datetime=datetime.now().isoformat(),
        exit_quoted_bid=call_bid + put_bid,
        exit_quoted_ask=call_ask + put_ask,
        exit_quoted_mid=call_mid + put_mid,
        exit_limit_price=call_limit + put_limit,
        status='exiting',
    )

    logger.info(
        f"{symbol}: Placed exit orders - "
        f"Call {contracts}x @ ${call_limit:.2f}, Put {contracts}x @ ${put_limit:.2f}"
    )

    # Create exit order pair for tracking
    exit_pair = ExitOrderPair(
        trade_id=trade_id,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        call_order_id=call_trade.order.orderId,
        put_order_id=put_trade.order.orderId,
        call_trade=call_trade,
        put_trade=put_trade,
        entry_fill_price=entry_fill_price,
        status='pending',
    )

    return exit_pair


def check_exit_fills(
    exit_orders: dict[str, ExitOrderPair],
    trade_logger: TradeLogger,
) -> list[ExitOrderPair]:
    """Check status of exit orders and update trade logs.

    Returns list of fully filled exit order pairs.
    """
    filled = []

    for trade_id, exit_pair in list(exit_orders.items()):
        call_status = exit_pair.call_trade.orderStatus.status if exit_pair.call_trade else 'Unknown'
        put_status = exit_pair.put_trade.orderStatus.status if exit_pair.put_trade else 'Unknown'

        call_filled = call_status == 'Filled'
        put_filled = put_status == 'Filled'

        # Get fill prices
        if call_filled and exit_pair.call_fill_price is None:
            exit_pair.call_fill_price = exit_pair.call_trade.orderStatus.avgFillPrice
            logger.info(f"{exit_pair.symbol}: Exit call filled @ ${exit_pair.call_fill_price:.2f}")

        if put_filled and exit_pair.put_fill_price is None:
            exit_pair.put_fill_price = exit_pair.put_trade.orderStatus.avgFillPrice
            logger.info(f"{exit_pair.symbol}: Exit put filled @ ${exit_pair.put_fill_price:.2f}")

        # Both legs filled - calculate P&L
        if call_filled and put_filled:
            exit_pair.status = 'filled'
            filled.append(exit_pair)

            exit_fill = exit_pair.call_fill_price + exit_pair.put_fill_price
            exit_limit = (
                exit_pair.call_trade.order.lmtPrice + exit_pair.put_trade.order.lmtPrice
                if exit_pair.call_trade and exit_pair.put_trade else None
            )

            # Calculate P&L
            premium_received = exit_fill * exit_pair.contracts * 100
            exit_pnl = None
            exit_pnl_pct = None

            if exit_pair.entry_fill_price:
                premium_paid = exit_pair.entry_fill_price * exit_pair.contracts * 100
                exit_pnl = premium_received - premium_paid
                exit_pnl_pct = (exit_fill / exit_pair.entry_fill_price - 1)

            trade_logger.update_trade(
                trade_id,
                status='exited',
                exit_fill_price=exit_fill,
                exit_slippage=exit_fill - exit_limit if exit_limit else None,
                exit_pnl=exit_pnl,
                exit_pnl_pct=exit_pnl_pct,
            )

            logger.info(
                f"{exit_pair.symbol}: EXIT COMPLETE - "
                f"Fill: ${exit_fill:.2f}, P&L: ${exit_pnl:.2f}" if exit_pnl else
                f"{exit_pair.symbol}: EXIT COMPLETE - Fill: ${exit_fill:.2f}"
            )

        elif call_filled or put_filled:
            exit_pair.status = 'partial'
            logger.warning(
                f"{exit_pair.symbol}: PARTIAL EXIT - "
                f"Call: {call_status}, Put: {put_status}"
            )

    return filled
