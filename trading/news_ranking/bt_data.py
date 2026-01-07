"""Backtrader data feeds for the short strategy."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class PandasDataFeed(bt.feeds.PandasData):
    """Custom pandas data feed with OHLCV."""

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


def load_price_data(
    data_dir: Path,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data from parquet files.

    Expected structure: data_dir/prices.pqt with columns:
    - date, symbol, open, high, low, close, volume

    Returns dict of symbol -> DataFrame with DatetimeIndex.
    """
    prices_path = data_dir / "prices.pqt"

    if not prices_path.exists():
        logger.error(f"Price data not found: {prices_path}")
        return {}

    logger.info(f"Loading price data from {prices_path}...")
    df = pd.read_parquet(prices_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Ensure date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif df.index.name == "date":
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

    # Filter symbols
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]
        logger.info(f"  Filtered to {len(df):,} rows for {len(symbols)} symbols")

    # Filter date range
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # Required columns
    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try lowercase
        df.columns = df.columns.str.lower()
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return {}

    # Split by symbol using groupby (much faster than repeated filtering)
    logger.info(f"  Splitting into per-symbol DataFrames...")
    result = {}
    grouped = df.groupby("symbol")
    n_symbols = len(grouped)
    for i, (symbol, group) in enumerate(grouped):
        symbol_df = group.set_index("date").sort_index()
        symbol_df = symbol_df[["open", "high", "low", "close", "volume"]]
        result[symbol] = symbol_df

        if (i + 1) % 1000 == 0:
            logger.info(f"    Processed {i + 1:,}/{n_symbols:,} symbols...")

    logger.info(f"  Done: {len(result):,} symbols loaded")
    return result


def load_features_data(
    data_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load pre-computed features from ml_dataset.pqt.

    Returns DataFrame with columns: date, symbol, and all feature columns.
    The ml_dataset uses 'feature_date' which we rename to 'date' for consistency.
    """
    dataset_path = data_dir / "ml_dataset.pqt"

    if not dataset_path.exists():
        logger.error(f"Features dataset not found: {dataset_path}")
        return pd.DataFrame()

    logger.info(f"Loading features from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Handle date column - ml_dataset uses feature_date
    if "feature_date" in df.columns:
        df["date"] = pd.to_datetime(df["feature_date"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Filter date range
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    logger.info(f"  Filtered to {len(df):,} rows for {df['symbol'].nunique():,} symbols")
    return df


def add_data_feeds(
    cerebro: bt.Cerebro,
    data_dir: Path,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_symbols: int | None = None,
) -> list[str]:
    """Add price data feeds to cerebro.

    Note: We load ALL price data (not filtered by date) because backtrader
    handles date filtering internally via fromdate/todate params.
    This ensures we have sufficient warmup data.

    Returns list of symbols actually added.
    """
    # Load ALL price data (no date filtering - let backtrader handle it)
    price_data = load_price_data(
        data_dir=data_dir,
        symbols=symbols,
        start_date=None,  # Load all dates
        end_date=None,
    )

    if not price_data:
        logger.error("No price data loaded")
        return []

    # Limit symbols if requested
    symbol_list = list(price_data.keys())
    if max_symbols and len(symbol_list) > max_symbols:
        symbol_list = symbol_list[:max_symbols]

    # Convert dates for backtrader
    from_dt = pd.Timestamp(start_date).to_pydatetime() if start_date else None
    to_dt = pd.Timestamp(end_date).to_pydatetime() if end_date else None

    # Add each symbol as a data feed
    logger.info(f"Adding data feeds to cerebro...")
    added = []
    skipped = 0
    for i, symbol in enumerate(symbol_list):
        df = price_data[symbol]

        if df.empty or len(df) < 20:  # Need minimum history
            skipped += 1
            continue

        # Check if symbol has data in the requested range
        if start_date and df.index.max() < pd.Timestamp(start_date):
            skipped += 1
            continue
        if end_date and df.index.min() > pd.Timestamp(end_date):
            skipped += 1
            continue

        # CRITICAL: Check for data gaps - symbol must have data near the start date
        # Filter to test period and check first available date
        if start_date:
            test_period_data = df[df.index >= pd.Timestamp(start_date)]
            if test_period_data.empty:
                skipped += 1
                continue
            # Skip if first date in test period is too late (causes backtrader sync delay)
            first_test_date = test_period_data.index.min()
            if first_test_date > pd.Timestamp(start_date) + pd.Timedelta(days=7):
                skipped += 1
                continue

        data_feed = PandasDataFeed(
            dataname=df,
            name=symbol,
            fromdate=from_dt,
            todate=to_dt,
        )

        cerebro.adddata(data_feed, name=symbol)
        added.append(symbol)

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1:,}/{len(symbol_list):,} symbols ({len(added):,} added)...")

    logger.info(f"  Done: {len(added):,} data feeds added ({skipped:,} skipped)")
    return added


def compute_volatility_features(data_dir: Path) -> pd.DataFrame:
    """Compute realized volatility and dollar volume from price data.

    Returns DataFrame with columns: date, symbol, realized_vol, avg_dollar_vol
    """
    prices_path = data_dir / "prices.pqt"
    if not prices_path.exists():
        logger.warning("No prices.pqt for volatility computation")
        return pd.DataFrame()

    logger.info("Computing volatility features from prices...")
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])

    # Daily returns
    prices["ret"] = prices.groupby("symbol")["close"].pct_change()

    # 20-day realized volatility (annualized)
    prices["realized_vol"] = prices.groupby("symbol")["ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252)
    )

    # Dollar volume (proxy for liquidity/market cap)
    prices["dollar_volume"] = prices["close"] * prices["volume"]
    prices["avg_dollar_vol"] = prices.groupby("symbol")["dollar_volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    result = prices[["date", "symbol", "realized_vol", "avg_dollar_vol"]].copy()
    logger.info(f"  Computed volatility for {result['symbol'].nunique():,} symbols")
    return result


def prepare_backtest_features(
    data_dir: Path,
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    news_only: bool = True,
    add_vol_features: bool = True,
) -> pd.DataFrame:
    """Prepare features DataFrame for backtesting.

    Loads ml_dataset.pqt and filters to relevant symbols and dates.

    Args:
        data_dir: Path to data directory
        symbols: List of symbols to include
        start_date: Start date for filtering
        end_date: End date for filtering
        news_only: If True, filter to rows with news embeddings (default True)
        add_vol_features: If True, add realized_vol and avg_dollar_vol columns
    """
    df = load_features_data(data_dir, start_date, end_date)

    if df.empty:
        return df

    # Filter to symbols we have price data for
    df = df[df["symbol"].isin(symbols)]

    # Filter to news-only if requested
    if news_only:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if emb_cols:
            has_news = (df[emb_cols].abs().sum(axis=1) > 0)
            n_before = len(df)
            df = df[has_news]
            logger.info(f"  Filtered to news-only: {len(df):,} rows ({len(df)/n_before*100:.1f}% of {n_before:,})")

    # Add volatility features
    if add_vol_features:
        vol_df = compute_volatility_features(data_dir)
        if not vol_df.empty:
            df = df.merge(vol_df, on=["date", "symbol"], how="left")
            # Fill missing with high values (conservative - will be filtered/down-weighted)
            df["realized_vol"] = df["realized_vol"].fillna(df["realized_vol"].quantile(0.95))
            df["avg_dollar_vol"] = df["avg_dollar_vol"].fillna(df["avg_dollar_vol"].quantile(0.05))
            logger.info(f"  Added volatility features")

    logger.info(f"Prepared {len(df)} feature rows for backtest")
    return df


def select_tradeable_symbols(
    data_dir: Path,
    model,
    start_date: str | None = None,
    end_date: str | None = None,
    k_short: int = 5,
    buffer_multiplier: int = 5,
) -> list[str]:
    """Select symbols that are likely to be traded based on model scores.

    Pre-computes model scores across all dates and selects symbols that
    appear in the bottom-K at least once. Adds a buffer to account for
    day-to-day variation.

    Returns list of symbols to load price data for.
    """
    df = load_features_data(data_dir, start_date, end_date)

    if df.empty:
        logger.warning("No features to select symbols from")
        return []

    # Score all rows
    logger.info(f"Scoring {len(df)} rows to select tradeable symbols...")
    scores = model.score(df)
    df = df.copy()
    df["score"] = scores

    # For each date, get bottom-K symbols
    tradeable = set()
    for date, group in df.groupby("date"):
        bottom_k = group.nsmallest(k_short * buffer_multiplier, "score")
        tradeable.update(bottom_k["symbol"].tolist())

    logger.info(f"Selected {len(tradeable)} tradeable symbols (bottom-{k_short}*{buffer_multiplier} per day)")
    return list(tradeable)
