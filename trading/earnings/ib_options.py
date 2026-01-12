"""Interactive Brokers options client for earnings volatility strategy.

Extends base IB functionality with options-specific methods:
- Option chain fetching
- Straddle/strangle pricing
- Options order placement
- Greeks and IV data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
import asyncio

from ib_insync import (
    IB, Stock, Option, Contract, Trade, Ticker,
    LimitOrder, MarketOrder, ComboLeg, Bag
)

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """Quote data for a single option."""
    symbol: str
    expiry: date
    strike: float
    right: str  # 'C' or 'P'
    bid: float
    ask: float
    mid: float
    spread: float
    spread_pct: float
    last: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class StraddleQuote:
    """Quote for ATM straddle (call + put at same strike)."""
    symbol: str
    expiry: date
    strike: float
    call: OptionQuote
    put: OptionQuote
    total_bid: float
    total_ask: float
    total_mid: float
    total_spread: float
    total_spread_pct: float
    implied_move: float  # straddle_mid / spot
    spot_price: float


@dataclass
class OptionOrder:
    """Order details for options trade."""
    symbol: str
    expiry: date
    strike: float
    right: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'LMT' or 'MKT'
    limit_price: Optional[float] = None
    order_id: Optional[int] = None
    status: str = 'pending'
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    error: Optional[str] = None


class IBOptionsClient:
    """Interactive Brokers client for options trading."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # TWS live port (7496=TWS paper, 4001=Gateway live, 4002=Gateway paper)
        client_id: int = 1,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._connected = False

    def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        try:
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
            self._connected = True
            logger.info(f"Connected to IB on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price."""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)

        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)

        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        return price if price > 0 else None

    def get_option_chain(
        self,
        symbol: str,
        expiry: date,
        strikes: Optional[list[float]] = None,
        rights: list[str] = ['C', 'P'],
    ) -> list[OptionQuote]:
        """
        Fetch option chain for a symbol and expiry.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strikes: List of strikes to fetch (None = all available)
            rights: List of option types ('C' for call, 'P' for put)

        Returns:
            List of OptionQuote objects
        """
        # Qualify underlying
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)

        # Get available strikes/expiries
        chains = self.ib.reqSecDefOptParams(
            stock.symbol,
            '',
            stock.secType,
            stock.conId,
        )

        if not chains:
            logger.warning(f"No option chains found for {symbol}")
            return []

        # Find matching chain (SMART exchange preferred)
        chain = None
        for c in chains:
            if c.exchange == 'SMART':
                chain = c
                break
        if not chain:
            chain = chains[0]

        # Filter to requested expiry
        expiry_str = expiry.strftime('%Y%m%d')
        if expiry_str not in chain.expirations:
            logger.warning(f"Expiry {expiry} not available for {symbol}")
            available = sorted(chain.expirations)[:5]
            logger.info(f"Available expiries: {available}")
            return []

        # Determine strikes to fetch
        if strikes is None:
            strikes = sorted(chain.strikes)

        # Create option contracts
        contracts = []
        for strike in strikes:
            for right in rights:
                opt = Option(symbol, expiry_str, strike, right, 'SMART')
                contracts.append(opt)

        # Qualify contracts
        qualified = self.ib.qualifyContracts(*contracts)

        # Request market data
        tickers = []
        for contract in qualified:
            ticker = self.ib.reqMktData(contract, '', False, False)
            tickers.append((contract, ticker))

        # Wait for data
        self.ib.sleep(3)

        # Collect quotes
        quotes = []
        for contract, ticker in tickers:
            bid = ticker.bid if ticker.bid > 0 else 0
            ask = ticker.ask if ticker.ask > 0 else 0

            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pct = spread / mid * 100 if mid > 0 else 100
            else:
                mid = ticker.last if ticker.last and ticker.last > 0 else 0
                spread = 0
                spread_pct = 100

            # Get Greeks from model
            greeks = ticker.modelGreeks

            quote = OptionQuote(
                symbol=contract.symbol,
                expiry=datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d').date(),
                strike=contract.strike,
                right=contract.right,
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                spread_pct=spread_pct,
                last=ticker.last if ticker.last else None,
                volume=ticker.volume if ticker.volume else 0,
                open_interest=0,  # Requires separate request
                iv=greeks.impliedVol if greeks else None,
                delta=greeks.delta if greeks else None,
                gamma=greeks.gamma if greeks else None,
                theta=greeks.theta if greeks else None,
                vega=greeks.vega if greeks else None,
            )
            quotes.append(quote)

            # Cancel market data
            self.ib.cancelMktData(contract)

        return quotes

    def get_atm_straddle(
        self,
        symbol: str,
        expiry: date,
        spot_price: Optional[float] = None,
    ) -> Optional[StraddleQuote]:
        """
        Get ATM straddle quote for a symbol and expiry.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            spot_price: Current stock price (fetched if not provided)

        Returns:
            StraddleQuote or None if not available
        """
        if spot_price is None:
            spot_price = self.get_stock_price(symbol)
            if not spot_price:
                logger.warning(f"Could not get spot price for {symbol}")
                return None

        # Get chain info to find available strikes
        stock = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)

        chains = self.ib.reqSecDefOptParams(
            stock.symbol,
            '',
            stock.secType,
            stock.conId,
        )

        if not chains:
            return None

        chain = next((c for c in chains if c.exchange == 'SMART'), chains[0])

        # Find ATM strike (closest to spot)
        strikes = sorted(chain.strikes)
        atm_strike = min(strikes, key=lambda s: abs(s - spot_price))

        # Get call and put quotes
        quotes = self.get_option_chain(symbol, expiry, [atm_strike], ['C', 'P'])

        if len(quotes) < 2:
            logger.warning(f"Could not get both call and put for {symbol} {expiry} {atm_strike}")
            return None

        call = next((q for q in quotes if q.right == 'C'), None)
        put = next((q for q in quotes if q.right == 'P'), None)

        if not call or not put:
            return None

        total_bid = call.bid + put.bid
        total_ask = call.ask + put.ask
        total_mid = call.mid + put.mid
        total_spread = total_ask - total_bid
        total_spread_pct = (total_spread / total_mid * 100) if total_mid > 0 else 100
        implied_move = total_mid / spot_price if spot_price > 0 else 0

        return StraddleQuote(
            symbol=symbol,
            expiry=expiry,
            strike=atm_strike,
            call=call,
            put=put,
            total_bid=total_bid,
            total_ask=total_ask,
            total_mid=total_mid,
            total_spread=total_spread,
            total_spread_pct=total_spread_pct,
            implied_move=implied_move,
            spot_price=spot_price,
        )

    def place_option_order(
        self,
        symbol: str,
        expiry: date,
        strike: float,
        right: str,
        action: str,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrder:
        """
        Place a single-leg option order.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strike: Strike price
            right: 'C' for call, 'P' for put
            action: 'BUY' or 'SELL'
            quantity: Number of contracts
            limit_price: Limit price (None for market order)

        Returns:
            OptionOrder with status
        """
        result = OptionOrder(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            action=action,
            quantity=quantity,
            order_type='LMT' if limit_price else 'MKT',
            limit_price=limit_price,
        )

        try:
            # Create contract
            expiry_str = expiry.strftime('%Y%m%d')
            contract = Option(symbol, expiry_str, strike, right, 'SMART')
            qualified = self.ib.qualifyContracts(contract)

            if not qualified:
                result.error = f"Failed to qualify contract"
                result.status = 'rejected'
                return result

            # Create order
            if limit_price:
                order = LimitOrder(action, quantity, limit_price)
            else:
                order = MarketOrder(action, quantity)

            # Place order
            trade = self.ib.placeOrder(contract, order)
            result.order_id = trade.order.orderId
            result.status = trade.orderStatus.status

            return result

        except Exception as e:
            result.error = str(e)
            result.status = 'error'
            return result

    def place_straddle_order(
        self,
        symbol: str,
        expiry: date,
        strike: float,
        action: str,
        quantity: int,
        limit_price: Optional[float] = None,
    ) -> tuple[OptionOrder, OptionOrder]:
        """
        Place a straddle order (call + put at same strike).

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strike: Strike price
            action: 'BUY' or 'SELL'
            quantity: Number of contracts per leg
            limit_price: Combined limit price per straddle (None for market)

        Returns:
            Tuple of (call_order, put_order)
        """
        # For simplicity, place as two separate orders
        # Could also use combo/bag orders for better execution

        call_limit = put_limit = None
        if limit_price:
            # Split limit price between legs (rough approximation)
            call_limit = put_limit = limit_price / 2

        call_order = self.place_option_order(
            symbol, expiry, strike, 'C', action, quantity, call_limit
        )
        put_order = self.place_option_order(
            symbol, expiry, strike, 'P', action, quantity, put_limit
        )

        return call_order, put_order

    def get_open_orders(self) -> list[Trade]:
        """Get all open orders."""
        return self.ib.openTrades()

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order by ID."""
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                return True
        return False

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.ib.reqGlobalCancel()

    def get_positions(self) -> dict:
        """Get current option positions."""
        positions = {}
        for pos in self.ib.positions():
            contract = pos.contract
            if contract.secType == 'OPT':
                key = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
                positions[key] = {
                    'symbol': contract.symbol,
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': contract.right,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                }
        return positions

    def get_account_summary(self) -> dict:
        """Get account summary."""
        values = {}
        for av in self.ib.accountValues():
            if av.currency == 'USD' and av.tag in [
                'NetLiquidation',
                'TotalCashValue',
                'AvailableFunds',
                'BuyingPower',
                'GrossPositionValue',
            ]:
                values[av.tag] = float(av.value)
        return values

    def sleep(self, seconds: float):
        """Sleep while processing IB messages."""
        self.ib.sleep(seconds)
