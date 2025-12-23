"""Risk management for live trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

import pandas as pd

from trading.config import StrategyConfig
from trading.ib_client import IBClient


logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk check."""
    name: str
    passed: bool
    value: float
    limit: float
    message: str


class RiskManager:
    """Manages risk limits and stop-losses."""

    def __init__(self, config: StrategyConfig, ib_client: IBClient):
        self.config = config
        self.ib = ib_client

        # Track position entry prices for stop-loss
        self.entry_prices: dict[str, float] = {}
        self.entry_dates: dict[str, date] = {}

    def check_position_limits(
        self,
        proposed_positions: dict[str, float],
        portfolio_value: float,
    ) -> list[RiskCheck]:
        """Check if proposed positions are within limits.

        Args:
            proposed_positions: Dict of symbol -> position value (negative for short)
            portfolio_value: Total portfolio value
        """
        checks = []

        # Check individual position sizes
        max_pos_value = portfolio_value * self.config.max_position_pct

        for symbol, value in proposed_positions.items():
            pos_pct = abs(value) / portfolio_value if portfolio_value > 0 else 0

            checks.append(RiskCheck(
                name=f"position_size_{symbol}",
                passed=abs(value) <= max_pos_value,
                value=pos_pct,
                limit=self.config.max_position_pct,
                message=f"{symbol}: {pos_pct*100:.1f}% of portfolio (limit: {self.config.max_position_pct*100:.0f}%)",
            ))

        # Check total short exposure
        total_short = sum(abs(v) for v in proposed_positions.values() if v < 0)
        short_pct = total_short / portfolio_value if portfolio_value > 0 else 0

        checks.append(RiskCheck(
            name="total_short_exposure",
            passed=short_pct <= self.config.max_portfolio_short,
            value=short_pct,
            limit=self.config.max_portfolio_short,
            message=f"Total short: {short_pct*100:.1f}% (limit: {self.config.max_portfolio_short*100:.0f}%)",
        ))

        return checks

    def check_stop_losses(self) -> list[tuple[str, float, str]]:
        """Check if any positions hit stop-loss.

        Returns:
            List of (symbol, current_loss_pct, action) tuples
        """
        triggers = []

        portfolio = self.ib.get_portfolio_df()

        for symbol, pos in portfolio.items():
            quantity = pos["quantity"]
            if quantity == 0:
                continue

            entry_price = self.entry_prices.get(symbol)
            current_price = pos["market_price"]

            if entry_price is None or entry_price <= 0 or current_price <= 0:
                continue

            # Calculate P&L percentage
            if quantity < 0:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
            else:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price

            # Check stop-loss
            if pnl_pct < -self.config.stop_loss_pct:
                triggers.append((
                    symbol,
                    pnl_pct,
                    "STOP_LOSS",
                ))
                logger.warning(
                    f"Stop-loss triggered for {symbol}: {pnl_pct*100:.1f}% loss"
                )

        return triggers

    def update_entry_prices(self, trades: list[dict]):
        """Update entry prices after trades."""
        for trade in trades:
            symbol = trade["symbol"]
            price = trade.get("price", 0)
            quantity = trade.get("quantity", 0)

            if price > 0 and quantity != 0:
                # For new positions or additions, update entry price
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = price
                    self.entry_dates[symbol] = date.today()

    def remove_closed_positions(self, symbols: list[str]):
        """Remove tracking for closed positions."""
        for symbol in symbols:
            self.entry_prices.pop(symbol, None)
            self.entry_dates.pop(symbol, None)

    def run_pre_trade_checks(
        self,
        candidates: pd.DataFrame,
        portfolio_value: float,
    ) -> tuple[bool, list[RiskCheck]]:
        """Run all pre-trade risk checks.

        Returns:
            (all_passed, list_of_checks)
        """
        checks = []

        # Check market cap filter
        if "marketCap" in candidates.columns:
            below_min = candidates[candidates["marketCap"] < self.config.min_market_cap]
            checks.append(RiskCheck(
                name="market_cap_filter",
                passed=len(below_min) == 0,
                value=len(below_min),
                limit=0,
                message=f"{len(below_min)} candidates below ${self.config.min_market_cap/1e6:.0f}M market cap",
            ))

        # Check price filter
        if "price" in candidates.columns:
            below_min = candidates[candidates["price"] < self.config.min_price]
            checks.append(RiskCheck(
                name="min_price_filter",
                passed=len(below_min) == 0,
                value=len(below_min),
                limit=0,
                message=f"{len(below_min)} candidates below ${self.config.min_price} price",
            ))

        # Check volume filter
        if "avgVolume" in candidates.columns:
            below_min = candidates[candidates["avgVolume"] < self.config.min_volume]
            checks.append(RiskCheck(
                name="volume_filter",
                passed=len(below_min) == 0,
                value=len(below_min),
                limit=0,
                message=f"{len(below_min)} candidates below {self.config.min_volume/1000:.0f}K avg volume",
            ))

        # Check portfolio value is reasonable
        checks.append(RiskCheck(
            name="portfolio_value",
            passed=portfolio_value > 1000,
            value=portfolio_value,
            limit=1000,
            message=f"Portfolio value: ${portfolio_value:,.0f}",
        ))

        all_passed = all(c.passed for c in checks)

        return all_passed, checks

    def apply_filters(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Apply risk filters to candidate stocks."""
        df = candidates.copy()
        n_before = len(df)

        # Market cap filter
        if "marketCap" in df.columns:
            df = df[df["marketCap"] >= self.config.min_market_cap]

        # Price filter
        if "price" in df.columns:
            df = df[df["price"] >= self.config.min_price]

        # Volume filter
        if "avgVolume" in df.columns:
            df = df[df["avgVolume"] >= self.config.min_volume]

        n_after = len(df)
        logger.info(f"Risk filters: {n_before} -> {n_after} candidates")

        return df


class CircuitBreaker:
    """Emergency circuit breaker for extreme market conditions."""

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.10,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        self.start_of_day_value: float | None = None
        self.high_water_mark: float = 0
        self.is_triggered: bool = False

    def update(self, current_value: float) -> bool:
        """Update circuit breaker state.

        Returns True if circuit breaker is triggered.
        """
        # Update high water mark
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value

        # Set start of day value
        if self.start_of_day_value is None:
            self.start_of_day_value = current_value

        # Check daily loss
        daily_pnl_pct = (current_value - self.start_of_day_value) / self.start_of_day_value
        if daily_pnl_pct < -self.max_daily_loss_pct:
            logger.critical(
                f"CIRCUIT BREAKER: Daily loss {daily_pnl_pct*100:.1f}% exceeds limit"
            )
            self.is_triggered = True
            return True

        # Check drawdown
        drawdown_pct = (self.high_water_mark - current_value) / self.high_water_mark
        if drawdown_pct > self.max_drawdown_pct:
            logger.critical(
                f"CIRCUIT BREAKER: Drawdown {drawdown_pct*100:.1f}% exceeds limit"
            )
            self.is_triggered = True
            return True

        return False

    def reset_daily(self, current_value: float):
        """Reset daily tracking (call at start of trading day)."""
        self.start_of_day_value = current_value
        self.is_triggered = False

    def reset_all(self, current_value: float):
        """Full reset including high water mark."""
        self.start_of_day_value = current_value
        self.high_water_mark = current_value
        self.is_triggered = False
