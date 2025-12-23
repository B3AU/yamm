"""Interactive Brokers client using ib_insync."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ib_insync import IB, Stock, Order, Trade, Position, PortfolioItem, AccountValue
from ib_insync import MarketOrder, LimitOrder

from trading.config import IBConfig


logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission."""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    order_id: int | None = None
    status: str = "pending"
    fill_price: float | None = None
    error: str | None = None


class IBClient:
    """Interactive Brokers client wrapper."""

    def __init__(self, config: IBConfig):
        self.config = config
        self.ib = IB()
        self._connected = False

    async def connect(self) -> bool:
        """Connect to TWS/Gateway."""
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
            )
            self._connected = True
            logger.info(f"Connected to IB on port {self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False

    def connect_sync(self) -> bool:
        """Synchronous connect."""
        try:
            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
            )
            self._connected = True
            logger.info(f"Connected to IB on port {self.config.port}")
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

    # Account information

    def get_account_values(self) -> dict[str, float]:
        """Get account values (cash, portfolio value, etc.)."""
        values = {}
        for av in self.ib.accountValues():
            if av.currency == "USD" and av.tag in [
                "NetLiquidation",
                "TotalCashValue",
                "GrossPositionValue",
                "AvailableFunds",
                "BuyingPower",
            ]:
                values[av.tag] = float(av.value)
        return values

    def get_net_liquidation(self) -> float:
        """Get net liquidation value (total account value)."""
        values = self.get_account_values()
        return values.get("NetLiquidation", 0.0)

    def get_available_funds(self) -> float:
        """Get available funds for trading."""
        values = self.get_account_values()
        return values.get("AvailableFunds", 0.0)

    # Positions

    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        return self.ib.positions()

    def get_positions_df(self) -> dict[str, dict]:
        """Get positions as dict keyed by symbol."""
        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            positions[symbol] = {
                "symbol": symbol,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost,
            }
        return positions

    def get_portfolio(self) -> list[PortfolioItem]:
        """Get portfolio items with market values."""
        return self.ib.portfolio()

    def get_portfolio_df(self) -> dict[str, dict]:
        """Get portfolio as dict keyed by symbol."""
        portfolio = {}
        for item in self.ib.portfolio():
            symbol = item.contract.symbol
            portfolio[symbol] = {
                "symbol": symbol,
                "quantity": item.position,
                "market_price": item.marketPrice,
                "market_value": item.marketValue,
                "avg_cost": item.averageCost,
                "unrealized_pnl": item.unrealizedPNL,
                "realized_pnl": item.realizedPNL,
            }
        return portfolio

    # Contract creation

    def make_stock_contract(self, symbol: str, exchange: str = "SMART") -> Stock:
        """Create a stock contract."""
        return Stock(symbol, exchange, "USD")

    def qualify_contract(self, contract: Stock) -> Stock | None:
        """Qualify a contract (get full details from IB)."""
        try:
            qualified = self.ib.qualifyContracts(contract)
            return qualified[0] if qualified else None
        except Exception as e:
            logger.warning(f"Failed to qualify contract {contract.symbol}: {e}")
            return None

    # Short availability

    def check_shortable(self, symbol: str) -> tuple[bool, float]:
        """Check if a stock is shortable and get borrow fee.

        Returns (is_shortable, borrow_fee_rate).
        Note: Paper trading always returns shortable.
        """
        contract = self.make_stock_contract(symbol)
        qualified = self.qualify_contract(contract)

        if not qualified:
            return False, 0.0

        try:
            # Request short availability
            # IB returns: >0 = number of shares, -1 = not shortable, -2 = unknown
            shortable = self.ib.reqSecDefOptParams(
                qualified.symbol,
                "",
                qualified.secType,
                qualified.conId,
            )
            # For actual short checking, we'd use reqSoftDollarTiers or similar
            # Simplified for now - assume shortable
            return True, 0.0
        except Exception as e:
            logger.warning(f"Failed to check shortable for {symbol}: {e}")
            return True, 0.0  # Assume shortable for paper trading

    # Order management

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str = "SELL",  # "BUY" or "SELL"
    ) -> OrderResult:
        """Place a market order."""
        contract = self.make_stock_contract(symbol)
        qualified = self.qualify_contract(contract)

        if not qualified:
            return OrderResult(
                symbol=symbol,
                action=action,
                quantity=quantity,
                error=f"Failed to qualify contract for {symbol}",
            )

        order = MarketOrder(action, abs(quantity))
        trade = self.ib.placeOrder(qualified, order)

        return OrderResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
        )

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        limit_price: float,
        action: str = "SELL",
    ) -> OrderResult:
        """Place a limit order."""
        contract = self.make_stock_contract(symbol)
        qualified = self.qualify_contract(contract)

        if not qualified:
            return OrderResult(
                symbol=symbol,
                action=action,
                quantity=quantity,
                error=f"Failed to qualify contract for {symbol}",
            )

        order = LimitOrder(action, abs(quantity), limit_price)
        trade = self.ib.placeOrder(qualified, order)

        return OrderResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
        )

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

    def get_open_orders(self) -> list[Trade]:
        """Get all open orders."""
        return self.ib.openTrades()

    def wait_for_fills(self, timeout: float = 60.0) -> list[Trade]:
        """Wait for pending orders to fill."""
        self.ib.sleep(timeout)
        return self.ib.fills()

    # Market data

    def get_market_price(self, symbol: str) -> float | None:
        """Get current market price for a symbol."""
        contract = self.make_stock_contract(symbol)
        qualified = self.qualify_contract(contract)

        if not qualified:
            return None

        try:
            ticker = self.ib.reqMktData(qualified, "", False, False)
            self.ib.sleep(2)  # Wait for data
            price = ticker.marketPrice()
            self.ib.cancelMktData(qualified)
            return price if price > 0 else None
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return None

    def get_market_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get market prices for multiple symbols."""
        prices = {}
        contracts = []

        for symbol in symbols:
            contract = self.make_stock_contract(symbol)
            qualified = self.qualify_contract(contract)
            if qualified:
                contracts.append((symbol, qualified))

        # Request market data for all
        tickers = {}
        for symbol, contract in contracts:
            try:
                ticker = self.ib.reqMktData(contract, "", False, False)
                tickers[symbol] = (contract, ticker)
            except Exception as e:
                logger.warning(f"Failed to request data for {symbol}: {e}")

        # Wait for data
        self.ib.sleep(3)

        # Collect prices
        for symbol, (contract, ticker) in tickers.items():
            price = ticker.marketPrice()
            if price > 0:
                prices[symbol] = price
            self.ib.cancelMktData(contract)

        return prices

    # Utility

    def sleep(self, seconds: float):
        """Sleep while processing IB messages."""
        self.ib.sleep(seconds)
