"""Data pipeline for live trading - fetches news, fundamentals, and prices."""
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from trading.config import DataConfig


logger = logging.getLogger(__name__)


class FMPClient:
    """Financial Modeling Prep API client."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not set")

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Make API request."""
        params = params or {}
        params["apikey"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_stock_news(
        self,
        symbols: list[str],
        from_date: str,
        to_date: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch news for symbols in date range."""
        # FMP limits to 5 tickers per request
        all_news = []
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            tickers = ",".join(batch)
            try:
                news = self._get("stock_news", {
                    "tickers": tickers,
                    "from": from_date,
                    "to": to_date,
                    "limit": limit,
                })
                if news:
                    all_news.extend(news)
            except Exception as e:
                logger.warning(f"Failed to fetch news for {batch}: {e}")

        return all_news

    def get_quote(self, symbols: list[str]) -> list[dict]:
        """Get current quotes for symbols."""
        # FMP allows bulk quotes
        tickers = ",".join(symbols)
        return self._get(f"quote/{tickers}")

    def get_historical_price(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """Get historical daily prices."""
        return self._get(f"historical-price-full/{symbol}", {
            "from": from_date,
            "to": to_date,
        }).get("historical", [])

    def get_key_metrics(self, symbol: str, period: str = "quarter") -> list[dict]:
        """Get key metrics (fundamentals)."""
        return self._get(f"key-metrics/{symbol}", {"period": period})

    def get_ratios(self, symbol: str, period: str = "quarter") -> list[dict]:
        """Get financial ratios."""
        return self._get(f"ratios/{symbol}", {"period": period})

    def get_income_growth(self, symbol: str, period: str = "quarter") -> list[dict]:
        """Get growth metrics."""
        return self._get(f"income-statement-growth/{symbol}", {"period": period})


class DataPipeline:
    """Pipeline to prepare features for model inference."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.fmp = FMPClient(config.fmp_api_key)

    def load_universe(self) -> list[str]:
        """Load trading universe (symbols to consider)."""
        if self.config.universe_file.exists():
            df = pd.read_csv(self.config.universe_file)
            return df["symbol"].tolist()

        # Default: use symbols from our dataset
        dataset_path = self.config.data_dir / "ml_dataset.pqt"
        if dataset_path.exists():
            df = pd.read_parquet(dataset_path, columns=["symbol"])
            return df["symbol"].unique().tolist()

        raise FileNotFoundError("No universe file or dataset found")

    def fetch_latest_news(
        self,
        symbols: list[str],
        lookback_days: int = 1,
    ) -> pd.DataFrame:
        """Fetch news from last N days."""
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info(f"Fetching news for {len(symbols)} symbols from {from_date} to {to_date}")
        news = self.fmp.get_stock_news(symbols, from_date, to_date)

        if not news:
            logger.warning("No news fetched")
            return pd.DataFrame()

        df = pd.DataFrame(news)
        df["publishedDate"] = pd.to_datetime(df["publishedDate"])
        logger.info(f"Fetched {len(df)} news articles")

        return df

    def fetch_quotes(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch current quotes for all symbols."""
        logger.info(f"Fetching quotes for {len(symbols)} symbols")

        # Batch into chunks of 100
        all_quotes = []
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i+100]
            try:
                quotes = self.fmp.get_quote(batch)
                all_quotes.extend(quotes)
            except Exception as e:
                logger.warning(f"Failed to fetch quotes for batch {i}: {e}")

        if not all_quotes:
            return pd.DataFrame()

        df = pd.DataFrame(all_quotes)
        logger.info(f"Fetched {len(df)} quotes")
        return df

    def compute_price_features(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Compute price features from quotes."""
        if quotes_df.empty:
            return pd.DataFrame()

        df = quotes_df.copy()

        # Available from quote: price, changesPercentage, dayLow, dayHigh,
        # yearLow, yearHigh, volume, avgVolume, previousClose, open

        # Simple features we can compute from quote data
        df["overnight_gap"] = (df["open"] - df["previousClose"]) / df["previousClose"]
        df["intraday_ret"] = (df["price"] - df["open"]) / df["open"]
        df["ret_1d"] = df["changesPercentage"] / 100

        # Approximate other features (would need historical data for proper calculation)
        # For now, use placeholders - should fetch historical prices for accurate features
        df["ret_2d"] = df["ret_1d"]  # Placeholder
        df["ret_3d"] = df["ret_1d"]  # Placeholder
        df["ret_5d"] = df["ret_1d"]  # Placeholder

        # Volatility proxy from day range
        df["vol_5d"] = (df["dayHigh"] - df["dayLow"]) / df["price"]

        # Distance from high/low
        df["dist_from_high_5d"] = (df["price"] - df["yearHigh"]) / df["yearHigh"]
        df["dist_from_low_5d"] = (df["price"] - df["yearLow"]) / df["yearLow"]

        return df

    def cross_sectional_zscore(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Z-score normalize columns cross-sectionally."""
        df = df.copy()
        for col in cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f"{col}_z"] = ((df[col] - mean) / std).clip(-3, 3)
                else:
                    df[f"{col}_z"] = 0.0
        return df

    def prepare_features(
        self,
        symbols: list[str],
        fundamentals_df: pd.DataFrame | None = None,
        embeddings_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Prepare full feature matrix for model inference."""
        # Fetch current quotes
        quotes = self.fetch_quotes(symbols)
        if quotes.empty:
            logger.error("No quotes available")
            return pd.DataFrame()

        # Compute price features
        df = self.compute_price_features(quotes)

        # Z-score normalize price features
        price_raw_cols = [
            "overnight_gap", "intraday_ret", "ret_1d", "ret_2d",
            "ret_3d", "ret_5d", "vol_5d", "dist_from_high_5d", "dist_from_low_5d"
        ]
        df = self.cross_sectional_zscore(df, price_raw_cols)

        # Join fundamentals if provided
        if fundamentals_df is not None and not fundamentals_df.empty:
            df = df.merge(fundamentals_df, on="symbol", how="left")

        # Join embeddings if provided
        if embeddings_df is not None and not embeddings_df.empty:
            df = df.merge(embeddings_df, on="symbol", how="left")

        # Fill missing embeddings with zeros
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if emb_cols:
            df[emb_cols] = df[emb_cols].fillna(0)

        logger.info(f"Prepared features for {len(df)} symbols")
        return df


def load_cached_fundamentals(data_dir: Path) -> pd.DataFrame:
    """Load most recent fundamentals from cached data."""
    # Load from our pre-processed data
    metrics_path = data_dir / "key_metrics.pqt"
    ratios_path = data_dir / "ratios.pqt"
    growth_path = data_dir / "growth.pqt"

    if not all(p.exists() for p in [metrics_path, ratios_path, growth_path]):
        logger.warning("Cached fundamentals not found")
        return pd.DataFrame()

    metrics = pd.read_parquet(metrics_path)
    ratios = pd.read_parquet(ratios_path)
    growth = pd.read_parquet(growth_path)

    # Get most recent per symbol
    metrics["date"] = pd.to_datetime(metrics["date"])
    ratios["date"] = pd.to_datetime(ratios["date"])
    growth["date"] = pd.to_datetime(growth["date"])

    latest_metrics = metrics.sort_values("date").groupby("symbol").last().reset_index()
    latest_ratios = ratios.sort_values("date").groupby("symbol").last().reset_index()
    latest_growth = growth.sort_values("date").groupby("symbol").last().reset_index()

    # Merge
    fund = latest_metrics.merge(latest_ratios, on="symbol", how="outer", suffixes=("", "_r"))
    fund = fund.merge(latest_growth, on="symbol", how="outer", suffixes=("", "_g"))

    # Select and rename columns to match training features
    fund_cols = [
        "evToEBITDA", "freeCashFlowYield", "earningsYield",
        "returnOnEquity", "returnOnAssets", "returnOnInvestedCapital", "currentRatio",
        "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
        "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
        "debtToEquityRatio", "debtToAssetsRatio",
        "revenueGrowth", "netIncomeGrowth", "epsgrowth", "operatingIncomeGrowth",
    ]

    available_cols = ["symbol"] + [c for c in fund_cols if c in fund.columns]
    fund = fund[available_cols]

    return fund
