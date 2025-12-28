"""Backtrader strategy for cross-sectional short selling."""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any

import backtrader as bt
import pandas as pd
import numpy as np

from trading.config import StrategyConfig
from trading.model import ModelInference


logger = logging.getLogger(__name__)


class ShortRankerStrategy(bt.Strategy):
    """Short-only strategy based on cross-sectional model rankings.

    Features:
    - Bottom-K short selection from model scores
    - Drawdown-based position scaling (optional)
    - Confidence-weighted sizing (optional)
    - Daily rebalancing at specified time
    """

    params = (
        # Position sizing
        ("k_short", 5),                    # Number of stocks to short
        ("max_position_pct", 0.20),        # Max 20% per position
        ("max_portfolio_short", 1.0),      # Max 100% short exposure

        # Holding period
        ("hold_days", 1),                  # Days between rebalances

        # Drawdown scaling (DD 10-20% from backtest)
        ("use_dd_scaling", True),          # Enable drawdown-based scaling
        ("dd_threshold", 0.10),            # Start scaling at 10% DD
        ("dd_max", 0.20),                  # Max scaling at 20% DD
        ("dd_min_scale", 0.25),            # Minimum scale factor

        # Confidence weighting
        ("use_confidence", False),         # Enable confidence-weighted sizing
        ("confidence_col", "confidence"),  # Column name for confidence scores

        # Volatility/liquidity filters
        ("max_volatility", None),          # Max realized vol filter (e.g., 1.5 = 150% annualized)
        ("min_dollar_volume", None),       # Min avg dollar volume filter (e.g., 1e6 = $1M)
        ("use_inverse_vol", False),        # Enable inverse-volatility position sizing

        # Stop loss
        ("stop_loss_pct", None),           # Stop loss disabled (None = no stop loss)

        # Fees
        ("commission", 0.001),             # 0.1% commission

        # Model
        ("model", None),                   # ModelInference instance
        ("features_df", None),             # DataFrame with features for all symbols
    )

    def __init__(self):
        self.order_refs = {}  # Track pending orders
        self.last_rebalance_date = None
        self.peak_value = None
        self.entry_prices = {}  # Track entry prices for stop loss
        self.bar_count = 0  # For progress logging
        self.last_progress_date = None

        # Map data feeds by symbol
        self.data_map = {d._name: d for d in self.datas}
        logger.info(f"Strategy initialized with {len(self.data_map):,} symbols")

    def log(self, msg: str, dt: datetime | None = None):
        """Log message with timestamp."""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt} - {msg}")

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED: {order.data._name} @ {order.executed.price:.2f}")
            else:
                self.log(f"SELL EXECUTED: {order.data._name} @ {order.executed.price:.2f}")
                # Track entry price for stop loss
                self.entry_prices[order.data._name] = order.executed.price

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.data._name} failed: {order.status}")

        # Remove from pending
        if order.ref in self.order_refs:
            del self.order_refs[order.ref]

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            self.log(f"TRADE CLOSED: {trade.data._name} PnL={trade.pnl:.2f} ({trade.pnlcomm:.2f} net)")
            # Clear entry price
            if trade.data._name in self.entry_prices:
                del self.entry_prices[trade.data._name]

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        current_value = self.broker.getvalue()

        if self.peak_value is None:
            self.peak_value = current_value
            return 0.0

        if current_value > self.peak_value:
            self.peak_value = current_value

        return (self.peak_value - current_value) / self.peak_value

    def get_dd_scale(self) -> float:
        """Calculate position scale based on drawdown."""
        if not self.p.use_dd_scaling:
            return 1.0

        dd = self.get_drawdown()

        if dd <= self.p.dd_threshold:
            return 1.0
        elif dd >= self.p.dd_max:
            return self.p.dd_min_scale
        else:
            # Linear interpolation
            scale = 1.0 - (1.0 - self.p.dd_min_scale) * (dd - self.p.dd_threshold) / (self.p.dd_max - self.p.dd_threshold)
            return scale

    def should_rebalance(self) -> bool:
        """Check if we should rebalance today."""
        current_date = self.datas[0].datetime.date(0)

        if self.last_rebalance_date is None:
            return True

        days_since = (current_date - self.last_rebalance_date).days
        return days_since >= self.p.hold_days

    def check_stop_losses(self):
        """Check and execute stop losses."""
        # Skip if stop loss is disabled
        if self.p.stop_loss_pct is None:
            return

        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)

            if pos.size < 0 and symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                current = data.close[0]

                # For shorts, loss is when price goes up
                loss_pct = (current - entry) / entry

                if loss_pct > self.p.stop_loss_pct:
                    self.log(f"STOP LOSS: {symbol} entry={entry:.2f} current={current:.2f} loss={loss_pct*100:.1f}%")
                    self.close(data)

    def get_candidates_for_date(self, current_date: date) -> pd.DataFrame:
        """Get model scores for current date."""
        if self.p.features_df is None or self.p.model is None:
            return pd.DataFrame()

        # Filter features to current date
        df = self.p.features_df
        if "date" in df.columns:
            df = df[df["date"] == pd.Timestamp(current_date)]

        if df.empty:
            return pd.DataFrame()

        # Score with model
        try:
            scores = self.p.model.score(df)
            df = df.copy()
            df["score"] = scores

            # Compute confidence as |z-score| of scores
            if len(scores) > 1:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if std_score > 0:
                    df["confidence"] = np.abs((scores - mean_score) / std_score)
                else:
                    df["confidence"] = 1.0
            else:
                df["confidence"] = 1.0

        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            return pd.DataFrame()

        return df

    def calculate_positions(self, candidates: pd.DataFrame) -> dict[str, float]:
        """Calculate target position sizes for bottom-K candidates.

        Returns dict of symbol -> target_weight (negative for shorts).
        """
        k = self.p.k_short

        # Apply volatility filter if enabled
        if self.p.max_volatility is not None and "realized_vol" in candidates.columns:
            n_before = len(candidates)
            candidates = candidates[candidates["realized_vol"] <= self.p.max_volatility]
            if len(candidates) < n_before:
                logger.debug(f"Vol filter: {n_before} -> {len(candidates)} candidates")

        # Apply dollar volume filter if enabled
        if self.p.min_dollar_volume is not None and "avg_dollar_vol" in candidates.columns:
            n_before = len(candidates)
            candidates = candidates[candidates["avg_dollar_vol"] >= self.p.min_dollar_volume]
            if len(candidates) < n_before:
                logger.debug(f"Dollar vol filter: {n_before} -> {len(candidates)} candidates")

        if candidates.empty:
            logger.warning("No candidates after filters")
            return {}

        # Get bottom-K by score
        bottom_k = candidates.nsmallest(k, "score")

        # Base weight (equal weight among selected)
        n_selected = len(bottom_k)
        if n_selected == 0:
            return {}

        # Apply drawdown scaling
        dd_scale = self.get_dd_scale()

        targets = {}

        # Compute inverse-vol weights if enabled
        if self.p.use_inverse_vol and "realized_vol" in bottom_k.columns:
            # Inverse volatility weighting: w_i = (1/vol_i) / sum(1/vol_j)
            vols = bottom_k["realized_vol"].values
            # Clip to avoid division by zero and extreme weights
            vols = np.clip(vols, 0.1, 5.0)  # 10% to 500% vol range
            inv_vols = 1.0 / vols
            inv_vol_weights = inv_vols / inv_vols.sum()
        else:
            inv_vol_weights = np.ones(n_selected) / n_selected

        for i, (_, row) in enumerate(bottom_k.iterrows()):
            symbol = row["symbol"]

            # Skip if not in our data feeds
            if symbol not in self.data_map:
                continue

            # Base weight from inverse-vol or equal weight
            weight = self.p.max_portfolio_short * inv_vol_weights[i] * dd_scale

            # Apply confidence weighting if enabled (scale by confidence)
            if self.p.use_confidence and self.p.confidence_col in row:
                conf = row[self.p.confidence_col]
                # Normalize confidence for weighting (higher confidence = more weight)
                weight = weight * conf

            # Cap at max position size
            weight = min(weight, self.p.max_position_pct)

            # Negative for short
            targets[symbol] = -weight

        return targets

    def rebalance(self, targets: dict[str, float]):
        """Rebalance portfolio to target weights."""
        portfolio_value = self.broker.getvalue()

        # First, close positions not in targets
        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)

            if pos.size != 0 and symbol not in targets:
                self.log(f"CLOSING: {symbol} (not in bottom-K)")
                self.close(data)

        # Then, adjust positions to targets
        for symbol, target_weight in targets.items():
            if symbol not in self.data_map:
                continue

            data = self.data_map[symbol]
            pos = self.getposition(data)
            price = data.close[0]

            if price <= 0:
                continue

            # Target value and shares
            target_value = portfolio_value * abs(target_weight)
            target_shares = int(target_value / price)

            # For shorts, shares are negative
            if target_weight < 0:
                target_shares = -target_shares

            current_shares = pos.size
            trade_shares = target_shares - current_shares

            if abs(trade_shares) > 0:
                if trade_shares < 0:
                    # Sell to open/increase short
                    self.log(f"SHORT: {symbol} {abs(trade_shares)} shares @ ~{price:.2f}")
                    self.sell(data, size=abs(trade_shares))
                else:
                    # Buy to cover
                    self.log(f"COVER: {symbol} {trade_shares} shares @ ~{price:.2f}")
                    self.buy(data, size=trade_shares)

    def next(self):
        """Called on each bar."""
        self.bar_count += 1
        current_date = self.datas[0].datetime.date(0)

        # Log progress monthly
        current_month = (current_date.year, current_date.month)
        if self.last_progress_date is None or current_month != (self.last_progress_date.year, self.last_progress_date.month):
            value = self.broker.getvalue()
            pnl_pct = (value - 100_000) / 100_000 * 100  # Assumes 100k initial
            logger.info(f"Progress: {current_date} | Value: ${value:,.0f} ({pnl_pct:+.1f}%)")
            self.last_progress_date = current_date

        # Check stop losses first
        self.check_stop_losses()

        # Check if we should rebalance
        if not self.should_rebalance():
            return

        # Get candidates with model scores
        candidates = self.get_candidates_for_date(current_date)

        if candidates.empty:
            return

        # Calculate target positions
        targets = self.calculate_positions(candidates)

        if not targets:
            self.log("No valid targets found")
            return

        # Log state
        dd = self.get_drawdown()
        dd_scale = self.get_dd_scale()
        self.log(f"Rebalancing: DD={dd*100:.1f}%, scale={dd_scale:.2f}, targets={len(targets)}")

        # Execute rebalance
        self.rebalance(targets)

        self.last_rebalance_date = current_date


class ShortRankerStrategyLive(ShortRankerStrategy):
    """Live trading version with real-time data fetching.

    In live mode, we don't have pre-computed features_df.
    Instead, we fetch data and compute features on each rebalance.
    """

    # Extend parent params - backtrader uses tuple of tuples
    params = (
        # Inherited from parent
        ("k_short", 5),
        ("max_position_pct", 0.20),
        ("max_portfolio_short", 1.0),
        ("hold_days", 1),
        ("use_dd_scaling", True),
        ("dd_threshold", 0.10),
        ("dd_max", 0.20),
        ("dd_min_scale", 0.25),
        ("use_confidence", False),
        ("confidence_col", "confidence"),
        ("max_volatility", None),
        ("min_dollar_volume", None),
        ("use_inverse_vol", False),
        ("stop_loss_pct", None),
        ("commission", 0.001),
        ("model", None),
        ("features_df", None),
        # Live-specific params
        ("data_pipeline", None),           # DataPipeline instance for live data
        ("fundamentals_df", None),         # Cached fundamentals
        ("embeddings_df", None),           # Cached embeddings
    )

    def get_candidates_for_date(self, current_date: date) -> pd.DataFrame:
        """Fetch live data and compute model scores."""
        if self.p.data_pipeline is None or self.p.model is None:
            logger.error("data_pipeline and model required for live trading")
            return pd.DataFrame()

        # Get symbols from our data feeds
        symbols = list(self.data_map.keys())

        try:
            # Fetch and prepare features
            df = self.p.data_pipeline.prepare_features(
                symbols=symbols,
                fundamentals_df=self.p.fundamentals_df,
                embeddings_df=self.p.embeddings_df,
            )

            if df.empty:
                logger.warning("No features prepared")
                return pd.DataFrame()

            # Score with model
            scores = self.p.model.score(df)
            df["score"] = scores

            return df

        except Exception as e:
            logger.error(f"Live data fetch failed: {e}")
            return pd.DataFrame()
