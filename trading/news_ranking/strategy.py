"""Strategy and position management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

import pandas as pd
import numpy as np

from trading.config import StrategyConfig
from trading.ib_client import IBClient, OrderResult


logger = logging.getLogger(__name__)


@dataclass
class TargetPosition:
    """Target position for a symbol."""
    symbol: str
    target_quantity: int  # Negative for short
    current_quantity: int
    trade_quantity: int  # What we need to trade
    price: float
    score: float
    reason: str = ""


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    timestamp: datetime
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float
    pnl: float | None = None


class PositionManager:
    """Manages position sizing and rebalancing."""

    def __init__(self, config: StrategyConfig):
        self.config = config

    def calculate_position_sizes(
        self,
        candidates: pd.DataFrame,
        portfolio_value: float,
        current_positions: dict[str, dict],
        prices: dict[str, float],
    ) -> list[TargetPosition]:
        """Calculate target positions for bottom-K stocks.

        Args:
            candidates: DataFrame with 'symbol', 'score', and optionally 'volatility'
            portfolio_value: Total portfolio value
            current_positions: Current positions keyed by symbol
            prices: Current prices keyed by symbol

        Returns:
            List of TargetPosition objects
        """
        k = self.config.k_short
        max_pos_value = portfolio_value * self.config.max_position_pct
        total_short_value = portfolio_value * self.config.max_portfolio_short

        # Get bottom-K by score
        bottom_k = candidates.nsmallest(k, "score")

        # Calculate position sizes
        targets = []
        remaining_short_capacity = total_short_value

        for _, row in bottom_k.iterrows():
            symbol = row["symbol"]
            score = row["score"]
            price = prices.get(symbol, row.get("price", 0))

            if price <= 0:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            # Base position value (equal weight)
            base_value = total_short_value / k

            # Apply volatility adjustment if available
            if "volatility" in row and row["volatility"] > 0:
                # Inverse volatility weighting
                vol_adj = 1.0 / row["volatility"]
                base_value = base_value * vol_adj

            # Cap at max position size
            position_value = min(base_value, max_pos_value, remaining_short_capacity)
            remaining_short_capacity -= position_value

            # Calculate shares (negative for short)
            target_shares = -int(position_value / price)

            # Get current position
            current = current_positions.get(symbol, {})
            current_qty = int(current.get("quantity", 0))

            # Trade needed
            trade_qty = target_shares - current_qty

            targets.append(TargetPosition(
                symbol=symbol,
                target_quantity=target_shares,
                current_quantity=current_qty,
                trade_quantity=trade_qty,
                price=price,
                score=score,
                reason=f"Bottom-{k} short candidate",
            ))

        # Handle existing positions not in bottom-K (need to close)
        bottom_k_symbols = set(bottom_k["symbol"])
        for symbol, pos in current_positions.items():
            if symbol not in bottom_k_symbols and pos["quantity"] != 0:
                current_qty = int(pos["quantity"])
                price = prices.get(symbol, pos.get("avg_cost", 0))

                targets.append(TargetPosition(
                    symbol=symbol,
                    target_quantity=0,
                    current_quantity=current_qty,
                    trade_quantity=-current_qty,  # Close position
                    price=price,
                    score=float("inf"),
                    reason="Closing - not in bottom-K",
                ))

        return targets

    def calculate_fees(self, trade_value: float, price: float) -> float:
        """Calculate trading fees."""
        if price <= 0:
            return 0.0

        n_shares = abs(trade_value / price)
        per_share_fee = n_shares * self.config.fee_per_share
        max_fee = abs(trade_value) * self.config.max_fee_pct
        return min(per_share_fee, max_fee)


class ShortStrategy:
    """Short-only strategy based on model predictions."""

    def __init__(
        self,
        config: StrategyConfig,
        ib_client: IBClient,
    ):
        self.config = config
        self.ib = ib_client
        self.position_manager = PositionManager(config)
        self.trade_history: list[TradeRecord] = []
        self.last_rebalance_date: date | None = None

    def should_rebalance(self) -> bool:
        """Check if we should rebalance today."""
        today = date.today()

        # Never rebalanced - yes
        if self.last_rebalance_date is None:
            return True

        # Check hold period
        days_since = (today - self.last_rebalance_date).days
        return days_since >= self.config.hold_days

    def get_current_state(self) -> dict:
        """Get current portfolio state."""
        account = self.ib.get_account_values()
        positions = self.ib.get_portfolio_df()

        total_value = account.get("NetLiquidation", 0)
        available = account.get("AvailableFunds", 0)

        # Calculate short exposure
        short_value = sum(
            abs(p["market_value"])
            for p in positions.values()
            if p["quantity"] < 0
        )

        return {
            "total_value": total_value,
            "available_funds": available,
            "short_exposure": short_value,
            "short_pct": short_value / total_value if total_value > 0 else 0,
            "positions": positions,
            "n_positions": len([p for p in positions.values() if p["quantity"] != 0]),
        }

    def generate_orders(
        self,
        candidates: pd.DataFrame,
    ) -> list[TargetPosition]:
        """Generate orders based on model predictions.

        Args:
            candidates: DataFrame with columns: symbol, score, price

        Returns:
            List of TargetPosition objects representing trades to execute
        """
        state = self.get_current_state()
        portfolio_value = state["total_value"]
        positions = state["positions"]

        # Get current prices
        symbols = candidates["symbol"].tolist()
        prices = self.ib.get_market_prices(symbols)

        # Also get prices for existing positions
        for symbol in positions:
            if symbol not in prices:
                pos_price = positions[symbol].get("market_price")
                if pos_price:
                    prices[symbol] = pos_price

        # Calculate target positions
        targets = self.position_manager.calculate_position_sizes(
            candidates=candidates,
            portfolio_value=portfolio_value,
            current_positions=positions,
            prices=prices,
        )

        # Filter to only trades that need execution
        trades_needed = [t for t in targets if t.trade_quantity != 0]

        logger.info(f"Generated {len(trades_needed)} trades from {len(targets)} targets")

        return trades_needed

    def execute_orders(
        self,
        targets: list[TargetPosition],
        use_limit_orders: bool = False,
        limit_offset_pct: float = 0.001,
    ) -> list[OrderResult]:
        """Execute trades.

        Args:
            targets: List of TargetPosition objects
            use_limit_orders: Use limit orders instead of market
            limit_offset_pct: Offset for limit price (0.1% default)

        Returns:
            List of OrderResult objects
        """
        results = []

        for target in targets:
            if target.trade_quantity == 0:
                continue

            # Determine action
            if target.trade_quantity < 0:
                action = "SELL"  # Selling to go short or increase short
            else:
                action = "BUY"  # Buying to cover short

            quantity = abs(target.trade_quantity)

            logger.info(
                f"Executing {action} {quantity} {target.symbol} "
                f"(target: {target.target_quantity}, current: {target.current_quantity})"
            )

            try:
                if use_limit_orders and target.price > 0:
                    # Set limit price slightly worse than market
                    if action == "SELL":
                        limit_price = target.price * (1 - limit_offset_pct)
                    else:
                        limit_price = target.price * (1 + limit_offset_pct)

                    result = self.ib.place_limit_order(
                        symbol=target.symbol,
                        quantity=quantity,
                        limit_price=round(limit_price, 2),
                        action=action,
                    )
                else:
                    result = self.ib.place_market_order(
                        symbol=target.symbol,
                        quantity=quantity,
                        action=action,
                    )

                results.append(result)

                # Record trade
                self.trade_history.append(TradeRecord(
                    timestamp=datetime.now(),
                    symbol=target.symbol,
                    action=action,
                    quantity=quantity,
                    price=target.price,
                    commission=self.position_manager.calculate_fees(
                        quantity * target.price, target.price
                    ),
                ))

            except Exception as e:
                logger.error(f"Failed to execute order for {target.symbol}: {e}")
                results.append(OrderResult(
                    symbol=target.symbol,
                    action=action,
                    quantity=quantity,
                    error=str(e),
                ))

        return results

    def rebalance(
        self,
        candidates: pd.DataFrame,
        dry_run: bool = False,
    ) -> dict:
        """Full rebalance cycle.

        Args:
            candidates: DataFrame with model predictions
            dry_run: If True, don't execute orders

        Returns:
            Summary of rebalance operation
        """
        logger.info("Starting rebalance...")

        # Get current state
        state = self.get_current_state()
        logger.info(
            f"Current state: ${state['total_value']:.0f} total, "
            f"{state['n_positions']} positions, "
            f"{state['short_pct']*100:.1f}% short"
        )

        # Generate orders
        targets = self.generate_orders(candidates)

        if not targets:
            logger.info("No trades needed")
            return {"status": "no_trades", "state": state}

        # Log planned trades
        for t in targets:
            logger.info(
                f"  {t.symbol}: {t.current_quantity} -> {t.target_quantity} "
                f"(trade: {t.trade_quantity}, score: {t.score:.4f})"
            )

        if dry_run:
            logger.info("Dry run - not executing orders")
            return {
                "status": "dry_run",
                "targets": targets,
                "state": state,
            }

        # Execute orders
        results = self.execute_orders(targets)

        # Wait for fills
        self.ib.sleep(5)

        # Update state
        new_state = self.get_current_state()
        self.last_rebalance_date = date.today()

        # Summary
        successful = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)

        logger.info(
            f"Rebalance complete: {successful} orders submitted, {failed} failed"
        )

        return {
            "status": "executed",
            "orders_submitted": successful,
            "orders_failed": failed,
            "results": results,
            "old_state": state,
            "new_state": new_state,
        }

    def close_all_positions(self, dry_run: bool = False) -> dict:
        """Close all positions."""
        logger.info("Closing all positions...")

        state = self.get_current_state()
        positions = state["positions"]

        targets = []
        for symbol, pos in positions.items():
            qty = int(pos["quantity"])
            if qty != 0:
                targets.append(TargetPosition(
                    symbol=symbol,
                    target_quantity=0,
                    current_quantity=qty,
                    trade_quantity=-qty,
                    price=pos.get("market_price", 0),
                    score=0,
                    reason="Closing all positions",
                ))

        if not targets:
            logger.info("No positions to close")
            return {"status": "no_positions"}

        if dry_run:
            logger.info(f"Dry run - would close {len(targets)} positions")
            return {"status": "dry_run", "targets": targets}

        results = self.execute_orders(targets)

        return {
            "status": "executed",
            "results": results,
        }
