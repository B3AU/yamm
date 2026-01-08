"""ML predictor for earnings move quantiles.

Loads trained LightGBM models and computes edge for candidates.

Data sourcing priority (live-first):
1. FMP API for live data (earnings, prices, fundamentals, news)
2. Parquet files as fallback (for training data or if API fails)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# FMP API key for live data
FMP_API_KEY = os.getenv('FMP_API_KEY', '')
LIVE_DATA_ENABLED = bool(FMP_API_KEY)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'


@dataclass
class EdgePrediction:
    """ML prediction for a candidate."""
    symbol: str
    pred_q50: float
    pred_q75: float
    pred_q90: float
    pred_q95: float
    hist_move_mean: float
    edge_q75: float  # pred_q75 - hist_move_mean
    edge_q90: float  # pred_q90 - hist_move_mean


class EarningsPredictor:
    """Loads trained models and predicts earnings move quantiles.

    Uses live FMP data first, falls back to parquet files if needed.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.feature_cols = []
        self.quantiles = []

        # Fallback data (parquet files) - only loaded if needed
        self._historical_moves_cache: Optional[pd.DataFrame] = None
        self._prices_cache: Optional[pd.DataFrame] = None
        self._news_embeddings_cache: Optional[pd.DataFrame] = None
        self.news_pca = None  # PCA model for embeddings

        self._load_models()

    def _load_models(self):
        """Load trained LightGBM models."""
        config_path = self.model_dir / 'feature_config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Feature config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        self.feature_cols = config['feature_cols']
        self.quantiles = config['quantiles']

        for q in self.quantiles:
            model_path = self.model_dir / f'earnings_q{int(q*100)}.txt'
            if model_path.exists():
                self.models[q] = lgb.Booster(model_file=str(model_path))
                logger.info(f"Loaded model: q{int(q*100)}")
            else:
                logger.warning(f"Model not found: {model_path}")

        if not self.models:
            raise FileNotFoundError("No models found")

        logger.info(f"Loaded {len(self.models)} models with {len(self.feature_cols)} features")

        # Load PCA model for news embeddings
        pca_path = self.model_dir / 'news_pca.joblib'
        if pca_path.exists():
            self.news_pca = joblib.load(pca_path)
            logger.info(f"Loaded PCA model ({self.news_pca.n_components_} components)")

    # =========================================================================
    # Fallback data loading (parquet files)
    # =========================================================================

    def _get_historical_moves_fallback(self) -> Optional[pd.DataFrame]:
        """Load historical moves from parquet (lazy, cached)."""
        if self._historical_moves_cache is not None:
            return self._historical_moves_cache

        moves_path = DATA_DIR / 'earnings' / 'historical_earnings_moves.parquet'
        if moves_path.exists():
            self._historical_moves_cache = pd.read_parquet(moves_path)
            self._historical_moves_cache['earnings_date'] = pd.to_datetime(
                self._historical_moves_cache['earnings_date']
            )
            logger.info(f"Loaded {len(self._historical_moves_cache)} historical moves from parquet (fallback)")
        return self._historical_moves_cache

    def _get_prices_fallback(self) -> Optional[pd.DataFrame]:
        """Load prices from parquet (lazy, cached)."""
        if self._prices_cache is not None:
            return self._prices_cache

        prices_path = DATA_DIR / 'prices.pqt'
        if prices_path.exists():
            self._prices_cache = pd.read_parquet(prices_path)
            self._prices_cache['date'] = pd.to_datetime(self._prices_cache['date'])
            logger.info(f"Loaded prices for {self._prices_cache['symbol'].nunique()} symbols from parquet (fallback)")
        return self._prices_cache

    def _get_news_embeddings_fallback(self) -> Optional[pd.DataFrame]:
        """Load news embeddings from parquet (lazy, cached)."""
        if self._news_embeddings_cache is not None:
            return self._news_embeddings_cache

        news_emb_path = DATA_DIR / 'news_ranking' / 'news_embeddings.pqt'
        news_dates_path = DATA_DIR / 'news_ranking' / 'all_the_news_anon.pqt'
        if news_emb_path.exists() and news_dates_path.exists() and self.news_pca is not None:
            try:
                emb = pd.read_parquet(news_emb_path)
                dates = pd.read_parquet(news_dates_path, columns=['url', 'symbol', 'trading_date'])
                # Both have 'symbol', merge on both to avoid _x/_y suffixes
                self._news_embeddings_cache = emb.merge(dates, on=['url', 'symbol'], how='inner')
                self._news_embeddings_cache['trading_date'] = pd.to_datetime(
                    self._news_embeddings_cache['trading_date']
                )
                logger.info(f"Loaded {len(self._news_embeddings_cache)} news embeddings from parquet (fallback)")
            except Exception as e:
                logger.warning(f"Could not load news embeddings: {e}")
        return self._news_embeddings_cache

    # =========================================================================
    # Live FMP data fetching
    # =========================================================================

    def _make_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make HTTP request with rate limit retry (429)."""
        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 429:
                    wait = 2 ** attempt
                    logger.debug(f"Rate limit hit (429), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                return r
            except requests.RequestException:
                if attempt == 3:
                    return None  # Or let caller handle
                time.sleep(1)
        return None

    def _fetch_historical_earnings_moves(self, symbol: str, earnings_date: date) -> Optional[dict]:
        """Fetch historical earnings moves from FMP.

        Computes overnight_move_abs from historical earnings dates and prices.
        Returns dict with move statistics or None if insufficient data.
        """
        if not LIVE_DATA_ENABLED:
            return None

        try:
            # Get historical earnings dates
            url = f"https://financialmodelingprep.com/stable/earnings?symbol={symbol}&apikey={FMP_API_KEY}"
            r = self._make_request(url)
            if not r or r.status_code != 200:
                return None

            earnings_data = r.json()
            if not earnings_data:
                return None

            df = pd.DataFrame(earnings_data)
            if 'date' not in df.columns:
                return None

            df['date'] = pd.to_datetime(df['date'])
            past_earnings = df[df['date'] < pd.Timestamp(earnings_date)].sort_values('date')

            if len(past_earnings) < 1:
                return None

            # Get historical prices to compute moves
            # Need prices around each earnings date
            price_url = f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol={symbol}&apikey={FMP_API_KEY}"
            r = self._make_request(price_url)
            if not r or r.status_code != 200:
                return None

            price_data = r.json()
            if not price_data:
                return None

            prices_df = pd.DataFrame(price_data)
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            prices_df = prices_df.sort_values('date')

            # Compute moves for each past earnings
            moves = []
            gap_moves = []

            for _, row in past_earnings.iterrows():
                earn_date = row['date']

                # Get close before earnings and close after
                before = prices_df[prices_df['date'] < earn_date].tail(1)
                after = prices_df[prices_df['date'] >= earn_date].head(1)

                if len(before) == 0 or len(after) == 0:
                    continue

                close_before = before['adjClose'].values[0]
                close_after = after['adjClose'].values[0]
                open_after = after['open'].values[0] if 'open' in after.columns else close_after

                # Overnight move (close to close)
                overnight_move = abs(close_after / close_before - 1)
                moves.append(overnight_move)

                # Gap move (close to open)
                gap_move = abs(open_after / close_before - 1)
                gap_moves.append(gap_move)

            if len(moves) < 1:
                return None

            moves = np.array(moves)
            gap_moves = np.array(gap_moves)

            result = {
                'hist_move_mean': float(np.mean(moves)),
                'hist_move_median': float(np.median(moves)),
                'hist_move_std': float(np.std(moves)) if len(moves) > 1 else 0.0,
                'hist_move_max': float(np.max(moves)),
                'hist_move_min': float(np.min(moves)),
                'hist_move_cv': float(np.std(moves) / np.mean(moves)) if np.mean(moves) > 0 else 0.0,
                'recent_move_mean': float(np.mean(moves[-2:])) if len(moves) >= 1 else float(np.mean(moves)),
                'move_trend': float(moves[-1] - moves[0]) if len(moves) >= 2 else 0.0,
                'gap_continuation_ratio': float(np.mean(moves) / np.mean(gap_moves)) if np.mean(gap_moves) > 0 else 1.0,
                'n_past_earnings': len(moves),
            }

            logger.debug(f"{symbol}: Fetched {len(moves)} historical earnings moves from FMP")
            return result

        except Exception as e:
            logger.debug(f"{symbol}: Error fetching historical earnings moves: {e}")
            return None

    def _fetch_prices(self, symbol: str, earnings_date: date, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch recent price history from FMP.

        Returns DataFrame with date, open, high, low, close, volume columns.
        """
        if not LIVE_DATA_ENABLED:
            return None

        try:
            # Calculate date range
            end_date = earnings_date - timedelta(days=1)
            start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

            url = (
                f"https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
                f"?symbol={symbol}&from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
            )
            r = self._make_request(url)
            if not r or r.status_code != 200:
                return None

            data = r.json()
            if not data:
                return None

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Rename adj* columns to standard names for consistency
            rename_map = {
                'adjClose': 'close',
                'adjOpen': 'open',
                'adjHigh': 'high',
                'adjLow': 'low',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            logger.debug(f"{symbol}: Fetched {len(df)} price bars from FMP")
            return df

        except Exception as e:
            logger.debug(f"{symbol}: Error fetching prices: {e}")
            return None

    def _fetch_fundamentals(self, symbol: str, earnings_date: date) -> dict:
        """Fetch fundamental features from FMP API.

        Returns dict with key metrics, ratios, and growth rates.
        Uses point-in-time logic: only returns data filed before earnings_date.

        Fetches from 3 FMP endpoints:
        - /stable/key-metrics: evToEBITDA, freeCashFlowYield, earningsYield, etc.
        - /stable/ratios: P/E, P/B, P/S, margins, debtToEquity
        - /stable/financial-growth: revenueGrowth, netIncomeGrowth, epsgrowth
        """
        metrics_cols = ['evToEBITDA', 'freeCashFlowYield', 'earningsYield',
                        'returnOnEquity', 'returnOnAssets', 'currentRatio']
        ratios_cols = ['priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
                       'grossProfitMargin', 'operatingProfitMargin', 'netProfitMargin',
                       'debtToEquityRatio']
        growth_cols = ['revenueGrowth', 'netIncomeGrowth', 'epsgrowth']

        all_cols = metrics_cols + ratios_cols + growth_cols
        defaults = {col: 0.0 for col in all_cols}

        if not LIVE_DATA_ENABLED:
            return defaults

        result = defaults.copy()

        def fetch_and_extract(endpoint: str, columns: list) -> dict:
            """Fetch from endpoint and extract point-in-time values."""
            try:
                url = f"https://financialmodelingprep.com/stable/{endpoint}?symbol={symbol}&apikey={FMP_API_KEY}"
                r = self._make_request(url)

                if not r or r.status_code != 200:
                    return {}

                data = r.json()
                if not data:
                    return {}

                df = pd.DataFrame(data)

                # Filter to filings available before earnings_date
                if 'fillingDate' in df.columns:
                    df['filing_date'] = pd.to_datetime(df['fillingDate'])
                elif 'date' in df.columns:
                    df['filing_date'] = pd.to_datetime(df['date']) + pd.Timedelta(days=45)
                else:
                    return {}

                past = df[df['filing_date'] < pd.Timestamp(earnings_date)].sort_values('filing_date', ascending=False)

                if len(past) == 0:
                    return {}

                latest = past.iloc[0]

                extracted = {}
                for col in columns:
                    if col in latest.index and pd.notna(latest[col]):
                        extracted[col] = float(latest[col])

                return extracted

            except Exception as e:
                logger.debug(f"{symbol}: Error fetching {endpoint}: {e}")
                return {}

        try:
            metrics_data = fetch_and_extract('key-metrics', metrics_cols)
            ratios_data = fetch_and_extract('ratios', ratios_cols)
            growth_data = fetch_and_extract('financial-growth', growth_cols)

            result.update(metrics_data)
            result.update(ratios_data)
            result.update(growth_data)

            filled_count = sum(1 for v in result.values() if v != 0.0)
            logger.debug(f"{symbol}: Fetched {filled_count}/16 fundamentals")
            return result

        except Exception as e:
            logger.warning(f"{symbol}: Error fetching fundamentals: {e}")
            return defaults

    def _fetch_earnings_surprises(self, symbol: str, earnings_date: date) -> dict:
        """Fetch earnings surprise features from FMP API.

        Returns dict with surprise_pct_mean, beat_rate, surprise_streak.
        """
        defaults = {
            'surprise_pct_mean': 0.0,
            'beat_rate': 0.5,
            'surprise_streak': 0,
        }

        if not LIVE_DATA_ENABLED:
            return defaults

        try:
            url = f"https://financialmodelingprep.com/stable/earnings?symbol={symbol}&apikey={FMP_API_KEY}"
            r = self._make_request(url)

            if not r or r.status_code != 200:
                return defaults

            data = r.json()
            if not data:
                return defaults

            df = pd.DataFrame(data)

            if 'epsActual' not in df.columns or 'epsEstimated' not in df.columns:
                return defaults

            df['date'] = pd.to_datetime(df['date'])
            past = df[
                (df['date'] < pd.Timestamp(earnings_date)) &
                (df['epsActual'].notna()) &
                (df['epsEstimated'].notna())
            ].sort_values('date')

            if len(past) == 0:
                return defaults

            past = past.copy()
            past['surprise_pct'] = (past['epsActual'] - past['epsEstimated']) / past['epsEstimated'].abs().clip(lower=0.01)

            result = {
                'surprise_pct_mean': float(past['surprise_pct'].mean()),
                'beat_rate': float((past['epsActual'] > past['epsEstimated']).mean()),
            }

            # Compute streak
            recent = past.tail(4)
            beats = (recent['epsActual'] > recent['epsEstimated']).values
            streak = 0
            if len(beats) > 0:
                last_val = beats[-1]
                for b in reversed(beats):
                    if b == last_val:
                        streak += 1
                    else:
                        break
                if not last_val:
                    streak = -streak
            result['surprise_streak'] = streak

            logger.debug(f"{symbol}: Fetched surprise features - beat_rate={result['beat_rate']:.2f}, streak={streak}")
            return result

        except Exception as e:
            logger.warning(f"{symbol}: Error fetching earnings surprises: {e}")
            return defaults

    def _fetch_news_features(self, symbol: str, earnings_date: date, lookback_days: int = 7) -> dict:
        """Fetch news and compute PCA features from FMP.

        Returns dict with pre_earnings_news_count and news_pca_0 through news_pca_9.
        """
        defaults = {'pre_earnings_news_count': 0}
        for i in range(10):
            defaults[f'news_pca_{i}'] = 0.0

        if not LIVE_DATA_ENABLED or self.news_pca is None:
            return defaults

        try:
            from .live_news import get_live_news_pca_features
            news_count, pca_features = get_live_news_pca_features(
                symbol=symbol,
                earnings_date=earnings_date,
                pca_model=self.news_pca,
                lookback_days=lookback_days,
            )

            result = {'pre_earnings_news_count': news_count}
            for i, val in enumerate(pca_features):
                result[f'news_pca_{i}'] = float(val)

            if news_count > 0:
                logger.debug(f"{symbol}: Fetched {news_count} news articles from FMP")
            return result

        except Exception as e:
            logger.debug(f"{symbol}: Error fetching news: {e}")
            return defaults

    # =========================================================================
    # Feature computation (live-first, parquet fallback)
    # =========================================================================

    def _compute_historical_features(self, symbol: str, earnings_date: date) -> Optional[dict]:
        """Compute historical earnings features. Live first, parquet fallback."""
        # Try live FMP first
        live_result = self._fetch_historical_earnings_moves(symbol, earnings_date)
        if live_result is not None:
            return live_result

        # Fallback to parquet
        logger.warning(f"{symbol}: FMP historical earnings fetch failed, using parquet fallback")
        historical_moves = self._get_historical_moves_fallback()
        if historical_moves is None:
            logger.warning(f"{symbol}: No historical data available (live or parquet)")
            return None

        symbol_moves = historical_moves[
            (historical_moves['symbol'] == symbol) &
            (historical_moves['earnings_date'] < pd.Timestamp(earnings_date))
        ].sort_values('earnings_date')

        if len(symbol_moves) < 1:
            logger.debug(f"{symbol}: No historical earnings in parquet fallback")
            return None

        moves = symbol_moves['overnight_move_abs'].values

        result = {
            'hist_move_mean': float(np.mean(moves)),
            'hist_move_median': float(np.median(moves)),
            'hist_move_std': float(np.std(moves)) if len(moves) > 1 else 0.0,
            'hist_move_max': float(np.max(moves)),
            'hist_move_min': float(np.min(moves)),
            'hist_move_cv': float(np.std(moves) / np.mean(moves)) if np.mean(moves) > 0 else 0.0,
            'recent_move_mean': float(np.mean(moves[-2:])) if len(moves) >= 1 else float(np.mean(moves)),
            'move_trend': float(moves[-1] - moves[0]) if len(moves) >= 2 else 0.0,
            'n_past_earnings': len(moves),
        }

        # Gap continuation
        if 'gap_move_abs' in symbol_moves.columns:
            gap_mean = symbol_moves['gap_move_abs'].mean()
            result['gap_continuation_ratio'] = float(np.mean(moves) / gap_mean) if gap_mean > 0 else 1.0
        else:
            result['gap_continuation_ratio'] = 1.0

        return result

    def _compute_price_features(self, symbol: str, earnings_date: date) -> dict:
        """Compute price-based features. Live first, parquet fallback."""
        defaults = {
            'rvol_5d': 0.3, 'rvol_10d': 0.3, 'rvol_20d': 0.3,
            'ret_5d': 0.0, 'ret_10d': 0.0, 'ret_20d': 0.0,
            'dist_from_high_20d': 0.0, 'dist_from_low_20d': 0.0,
            'gap_frequency': 0.0, 'volume_ratio': 1.0,
        }

        # Try live FMP first
        prices_df = self._fetch_prices(symbol, earnings_date, lookback_days=30)

        # Fallback to parquet
        if prices_df is None or len(prices_df) < 5:
            logger.warning(f"{symbol}: FMP price fetch failed, using parquet fallback")
            prices_cache = self._get_prices_fallback()
            if prices_cache is not None:
                prices_df = prices_cache[
                    (prices_cache['symbol'] == symbol) &
                    (prices_cache['date'] < pd.Timestamp(earnings_date))
                ].sort_values('date').tail(30)

        if prices_df is None or len(prices_df) < 5:
            return defaults

        # Compute features
        returns = prices_df['close'].pct_change().dropna()

        result = {}
        result['rvol_5d'] = float(returns.tail(5).std() * np.sqrt(252)) if len(returns) >= 5 else 0.3
        result['rvol_10d'] = float(returns.tail(10).std() * np.sqrt(252)) if len(returns) >= 10 else 0.3
        result['rvol_20d'] = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.3

        closes = prices_df['close'].values
        result['ret_5d'] = float(closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0.0
        result['ret_10d'] = float(closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0.0
        result['ret_20d'] = float(closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0.0

        if len(prices_df) >= 20:
            result['dist_from_high_20d'] = float(closes[-1] / prices_df['high'].tail(20).max() - 1)
            result['dist_from_low_20d'] = float(closes[-1] / prices_df['low'].tail(20).min() - 1)
        else:
            result['dist_from_high_20d'] = 0.0
            result['dist_from_low_20d'] = 0.0

        # Gap frequency
        if 'open' in prices_df.columns and len(prices_df) > 1:
            gaps = np.abs(prices_df['open'].values[1:] / prices_df['close'].values[:-1] - 1)
            result['gap_frequency'] = float((gaps > 0.02).mean())
        else:
            result['gap_frequency'] = 0.0

        # Volume ratio
        if 'volume' in prices_df.columns and len(prices_df) >= 20:
            recent_vol = prices_df['volume'].tail(5).mean()
            avg_vol = prices_df['volume'].mean()
            result['volume_ratio'] = float(recent_vol / avg_vol) if avg_vol > 0 else 1.0
        else:
            result['volume_ratio'] = 1.0

        return result

    def _compute_news_features(self, symbol: str, earnings_date: date) -> dict:
        """Compute news PCA features. Live first, parquet fallback."""
        defaults = {'pre_earnings_news_count': 0}
        for i in range(10):
            defaults[f'news_pca_{i}'] = 0.0

        if self.news_pca is None:
            return defaults

        # Try live FMP first
        live_result = self._fetch_news_features(symbol, earnings_date, lookback_days=7)
        if live_result['pre_earnings_news_count'] > 0:
            return live_result

        # Fallback to parquet
        logger.warning(f"{symbol}: FMP news fetch returned no articles, using parquet fallback")
        news_embeddings = self._get_news_embeddings_fallback()
        if news_embeddings is None:
            return defaults

        lookback_days = 7
        earn_dt = pd.Timestamp(earnings_date)
        start_dt = earn_dt - timedelta(days=lookback_days)
        end_dt = earn_dt - timedelta(days=1)

        symbol_news = news_embeddings[
            (news_embeddings['symbol'] == symbol) &
            (news_embeddings['trading_date'] >= start_dt) &
            (news_embeddings['trading_date'] <= end_dt)
        ]

        if len(symbol_news) == 0:
            return defaults

        result = {'pre_earnings_news_count': len(symbol_news)}

        # Get embedding columns and compute mean
        emb_cols = [c for c in symbol_news.columns if c.startswith('emb_')]
        if not emb_cols:
            return defaults

        mean_emb = symbol_news[emb_cols].mean().values.reshape(1, -1)

        # Apply PCA
        pca_features = self.news_pca.transform(mean_emb)[0]
        for i, val in enumerate(pca_features):
            result[f'news_pca_{i}'] = float(val)

        return result

    def compute_features(
        self,
        symbol: str,
        earnings_date: date,
        timing: str = 'unknown',
    ) -> Optional[dict]:
        """
        Compute features for a single candidate.

        Uses live FMP data first, falls back to parquet files.
        Returns dict of features or None if insufficient data.
        """
        features = {}

        # Historical earnings features (required - return None if missing)
        hist_features = self._compute_historical_features(symbol, earnings_date)
        if hist_features is None:
            logger.debug(f"{symbol}: No historical earnings data available")
            return None
        features.update(hist_features)

        # Price features
        price_features = self._compute_price_features(symbol, earnings_date)
        features.update(price_features)

        # Surprise features
        surprise_data = self._fetch_earnings_surprises(symbol, earnings_date)
        features['surprise_pct_mean'] = surprise_data['surprise_pct_mean']
        features['beat_rate'] = surprise_data['beat_rate']
        features['surprise_streak'] = surprise_data['surprise_streak']

        # Timing features
        earn_dt = datetime.combine(earnings_date, datetime.min.time())
        features['day_of_week'] = earn_dt.weekday()
        features['month'] = earn_dt.month
        features['quarter'] = (earn_dt.month - 1) // 3 + 1
        features['is_earnings_season'] = 1 if earn_dt.month in [1, 2, 4, 5, 7, 8, 10, 11] else 0

        # Fundamentals
        fund_data = self._fetch_fundamentals(symbol, earnings_date)
        features.update(fund_data)

        # News features
        news_features = self._compute_news_features(symbol, earnings_date)
        features.update(news_features)

        # Timing encoded
        timing_map = {'BMO': 0, 'AMC': 1, 'unknown': 2}
        features['timing_encoded'] = timing_map.get(timing, 2)

        return features

    def predict(
        self,
        symbol: str,
        earnings_date: date,
        timing: str = 'unknown',
    ) -> Optional[EdgePrediction]:
        """
        Predict earnings move quantiles and compute edge.

        Returns EdgePrediction or None if prediction fails.
        """
        features = self.compute_features(symbol, earnings_date, timing)

        if features is None:
            return None

        # Build feature vector in correct order
        try:
            feature_vector = np.array([[features.get(col, 0.0) for col in self.feature_cols]])
        except Exception as e:
            logger.error(f"{symbol}: Feature vector error: {e}")
            return None

        # Get predictions from all models
        predictions = {}
        for q, model in self.models.items():
            try:
                pred = model.predict(feature_vector)[0]
                predictions[q] = float(pred)
            except Exception as e:
                logger.error(f"{symbol}: Prediction error for q{int(q*100)}: {e}")
                return None

        # Compute edge
        hist_move_mean = features.get('hist_move_mean', 0.0)

        return EdgePrediction(
            symbol=symbol,
            pred_q50=predictions.get(0.50, 0.0),
            pred_q75=predictions.get(0.75, 0.0),
            pred_q90=predictions.get(0.90, 0.0),
            pred_q95=predictions.get(0.95, 0.0),
            hist_move_mean=hist_move_mean,
            edge_q75=predictions.get(0.75, 0.0) - hist_move_mean,
            edge_q90=predictions.get(0.90, 0.0) - hist_move_mean,
        )

    def get_prediction_status(
        self,
        symbol: str,
        earnings_date: date,
    ) -> str:
        """
        Check why a prediction might fail for a symbol.
        Returns a human-readable status string.
        """
        hist_features = self._compute_historical_features(symbol, earnings_date)
        if hist_features is None:
            return f"No historical earnings data for {symbol} (checked FMP + parquet)"

        return f"OK - {hist_features['n_past_earnings']} historical earnings events"

    def filter_by_edge(
        self,
        candidates: list,
        edge_threshold: float = 0.05,
    ) -> list:
        """
        Filter candidates by ML edge.

        Args:
            candidates: List of ScreenedCandidate objects
            edge_threshold: Minimum edge_q75 to pass (default 5%)

        Returns:
            List of (candidate, prediction) tuples that pass the threshold
        """
        passed = []

        for candidate in candidates:
            prediction = self.predict(
                symbol=candidate.symbol,
                earnings_date=candidate.earnings_date,
                timing=candidate.timing,
            )

            if prediction is None:
                logger.info(f"{candidate.symbol}: No ML prediction (missing data)")
                continue

            if prediction.edge_q75 >= edge_threshold:
                logger.info(
                    f"{candidate.symbol}: PASS - edge={prediction.edge_q75:.1%} "
                    f"(pred_q75={prediction.pred_q75:.1%}, hist={prediction.hist_move_mean:.1%})"
                )
                passed.append((candidate, prediction))
            else:
                logger.info(
                    f"{candidate.symbol}: FAIL - edge={prediction.edge_q75:.1%} < {edge_threshold:.0%}"
                )

        return passed


import threading

# Singleton instance and lock
_predictor: Optional[EarningsPredictor] = None
_predictor_lock = threading.Lock()


def get_predictor() -> EarningsPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            # Double-checked locking
            if _predictor is None:
                _predictor = EarningsPredictor()
    return _predictor
