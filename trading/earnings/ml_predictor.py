"""ML predictor for earnings move quantiles.

Loads trained LightGBM models and computes edge for candidates.
Uses PCA-reduced news embeddings for improved predictions.
Supports live news fetching when historical embeddings unavailable.
"""
from __future__ import annotations

import json
import logging
import os
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

# Check if live news is enabled (FMP_API_KEY set)
LIVE_NEWS_ENABLED = bool(os.getenv('FMP_API_KEY', ''))

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
    """Loads trained models and predicts earnings move quantiles."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.feature_cols = []
        self.quantiles = []

        # Data for feature computation
        self.historical_moves: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.fundamentals: Optional[pd.DataFrame] = None
        self.news_embeddings: Optional[pd.DataFrame] = None
        self.news_pca = None  # PCA model for embeddings

        self._load_models()
        self._load_data()

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

    def _load_data(self):
        """Load historical data for feature computation."""
        # Load historical earnings moves
        moves_path = DATA_DIR / 'earnings' / 'historical_earnings_moves.parquet'
        if moves_path.exists():
            self.historical_moves = pd.read_parquet(moves_path)
            self.historical_moves['earnings_date'] = pd.to_datetime(
                self.historical_moves['earnings_date']
            )
            logger.info(f"Loaded {len(self.historical_moves)} historical moves")

        # Load prices
        prices_path = DATA_DIR / 'prices.pqt'
        if prices_path.exists():
            self.prices = pd.read_parquet(prices_path)
            self.prices['date'] = pd.to_datetime(self.prices['date'])
            logger.info(f"Loaded prices for {self.prices['symbol'].nunique()} symbols")

        # Load fundamentals
        metrics_path = DATA_DIR / 'key_metrics.pqt'
        if metrics_path.exists():
            self.fundamentals = pd.read_parquet(metrics_path)
            logger.info(f"Loaded fundamentals")

        # Load news embeddings (full 768-dim for PCA transform)
        news_emb_path = DATA_DIR / 'news_ranking' / 'news_embeddings.pqt'
        news_dates_path = DATA_DIR / 'news_ranking' / 'all_the_news_anon.pqt'
        if news_emb_path.exists() and news_dates_path.exists() and self.news_pca is not None:
            try:
                # Load embeddings with dates
                emb = pd.read_parquet(news_emb_path)
                dates = pd.read_parquet(news_dates_path, columns=['url', 'trading_date'])
                self.news_embeddings = emb.merge(dates, on='url', how='inner')
                self.news_embeddings['trading_date'] = pd.to_datetime(self.news_embeddings['trading_date'])
                logger.info(f"Loaded {len(self.news_embeddings)} news embeddings")
            except Exception as e:
                logger.warning(f"Could not load news embeddings: {e}")

    def _fetch_earnings_surprises(self, symbol: str, earnings_date: date) -> dict:
        """Fetch earnings surprise features from FMP API.

        Returns dict with surprise_pct_mean, beat_rate, surprise_streak.
        """
        defaults = {
            'surprise_pct_mean': 0.0,
            'surprise_pct_std': 0.0,
            'beat_rate': 0.5,
            'surprise_streak': 0,
        }

        if not FMP_API_KEY:
            return defaults

        try:
            url = f"https://financialmodelingprep.com/stable/earnings?symbol={symbol}&apikey={FMP_API_KEY}"
            r = requests.get(url, timeout=10)

            if r.status_code != 200:
                logger.debug(f"{symbol}: FMP earnings API returned {r.status_code}")
                return defaults

            data = r.json()
            if not data:
                return defaults

            df = pd.DataFrame(data)

            # Filter to rows with actual EPS data before earnings_date
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

            # Compute surprise percentage
            past = past.copy()
            past['surprise_pct'] = (past['epsActual'] - past['epsEstimated']) / past['epsEstimated'].abs().clip(lower=0.01)

            result = {
                'surprise_pct_mean': float(past['surprise_pct'].mean()),
                'surprise_pct_std': float(past['surprise_pct'].std()) if len(past) > 1 else 0.0,
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

    def compute_features(
        self,
        symbol: str,
        earnings_date: date,
        timing: str = 'unknown',
    ) -> Optional[dict]:
        """
        Compute features for a single candidate.

        Returns dict of features or None if insufficient data.
        """
        features = {}

        # Historical earnings features
        if self.historical_moves is not None:
            symbol_moves = self.historical_moves[
                (self.historical_moves['symbol'] == symbol) &
                (self.historical_moves['earnings_date'] < pd.Timestamp(earnings_date))
            ].sort_values('earnings_date')

            if len(symbol_moves) >= 1:
                moves = symbol_moves['overnight_move_abs'].values
                features['hist_move_mean'] = float(np.mean(moves))
                features['hist_move_median'] = float(np.median(moves))
                features['hist_move_std'] = float(np.std(moves)) if len(moves) > 1 else 0.0
                features['hist_move_max'] = float(np.max(moves))
                features['hist_move_min'] = float(np.min(moves))
                features['hist_move_cv'] = features['hist_move_std'] / features['hist_move_mean'] if features['hist_move_mean'] > 0 else 0.0
                features['recent_move_mean'] = float(np.mean(moves[-2:])) if len(moves) >= 1 else features['hist_move_mean']
                features['move_trend'] = float(moves[-1] - moves[0]) if len(moves) >= 2 else 0.0

                # Gap continuation
                if 'gap_move_abs' in symbol_moves.columns:
                    gap_mean = symbol_moves['gap_move_abs'].mean()
                    features['gap_continuation_ratio'] = features['hist_move_mean'] / gap_mean if gap_mean > 0 else 1.0
                else:
                    features['gap_continuation_ratio'] = 1.0

                features['n_past_earnings'] = len(symbol_moves)
            else:
                # No history - can't predict
                logger.debug(f"{symbol}: No historical earnings data")
                return None
        else:
            logger.debug(f"{symbol}: Historical moves dataset not loaded")
            return None

        # Price features
        if self.prices is not None:
            symbol_prices = self.prices[
                (self.prices['symbol'] == symbol) &
                (self.prices['date'] < pd.Timestamp(earnings_date))
            ].sort_values('date').tail(25)

            if len(symbol_prices) >= 5:
                returns = symbol_prices['close'].pct_change().dropna()

                features['rvol_5d'] = float(returns.tail(5).std() * np.sqrt(252)) if len(returns) >= 5 else 0.3
                features['rvol_10d'] = float(returns.tail(10).std() * np.sqrt(252)) if len(returns) >= 10 else 0.3
                features['rvol_20d'] = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.3

                closes = symbol_prices['close'].values
                features['ret_5d'] = float(closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0.0
                features['ret_10d'] = float(closes[-1] / closes[-10] - 1) if len(closes) >= 10 else 0.0
                features['ret_20d'] = float(closes[-1] / closes[-20] - 1) if len(closes) >= 20 else 0.0

                if len(symbol_prices) >= 20:
                    features['dist_from_high_20d'] = float(closes[-1] / symbol_prices['high'].tail(20).max() - 1)
                    features['dist_from_low_20d'] = float(closes[-1] / symbol_prices['low'].tail(20).min() - 1)
                else:
                    features['dist_from_high_20d'] = 0.0
                    features['dist_from_low_20d'] = 0.0

                # Gap frequency
                if 'open' in symbol_prices.columns and len(symbol_prices) > 1:
                    gaps = np.abs(symbol_prices['open'].values[1:] / symbol_prices['close'].values[:-1] - 1)
                    features['gap_frequency'] = float((gaps > 0.02).mean())
                else:
                    features['gap_frequency'] = 0.0

                # Volume ratio
                if 'volume' in symbol_prices.columns and len(symbol_prices) >= 20:
                    recent_vol = symbol_prices['volume'].tail(5).mean()
                    avg_vol = symbol_prices['volume'].mean()
                    features['volume_ratio'] = float(recent_vol / avg_vol) if avg_vol > 0 else 1.0
                else:
                    features['volume_ratio'] = 1.0
            else:
                # Not enough price data - use defaults
                for col in ['rvol_5d', 'rvol_10d', 'rvol_20d']:
                    features[col] = 0.3
                for col in ['ret_5d', 'ret_10d', 'ret_20d', 'dist_from_high_20d',
                           'dist_from_low_20d', 'gap_frequency']:
                    features[col] = 0.0
                features['volume_ratio'] = 1.0
        else:
            # No price data - use defaults
            for col in ['rvol_5d', 'rvol_10d', 'rvol_20d']:
                features[col] = 0.3
            for col in ['ret_5d', 'ret_10d', 'ret_20d', 'dist_from_high_20d',
                       'dist_from_low_20d', 'gap_frequency']:
                features[col] = 0.0
            features['volume_ratio'] = 1.0

        # Surprise features - fetch from FMP API
        surprise_data = self._fetch_earnings_surprises(symbol, earnings_date)
        features['surprise_pct_mean'] = surprise_data['surprise_pct_mean']
        features['surprise_pct_std'] = surprise_data['surprise_pct_std']
        features['beat_rate'] = surprise_data['beat_rate']
        features['surprise_streak'] = surprise_data['surprise_streak']

        # Timing features
        earn_dt = datetime.combine(earnings_date, datetime.min.time())
        features['day_of_week'] = earn_dt.weekday()
        features['month'] = earn_dt.month
        features['quarter'] = (earn_dt.month - 1) // 3 + 1
        features['is_earnings_season'] = 1 if earn_dt.month in [1, 2, 4, 5, 7, 8, 10, 11] else 0

        # Fundamentals - use defaults (would need data lookup for real values)
        fund_cols = ['evToEBITDA', 'freeCashFlowYield', 'earningsYield', 'returnOnEquity',
                     'returnOnAssets', 'currentRatio', 'priceToEarningsRatio', 'priceToBookRatio',
                     'priceToSalesRatio', 'grossProfitMargin', 'operatingProfitMargin',
                     'netProfitMargin', 'debtToEquityRatio', 'revenueGrowth', 'netIncomeGrowth',
                     'epsgrowth']
        for col in fund_cols:
            features[col] = 0.0  # Default - median imputation happens in model

        # News features with PCA embeddings
        features['pre_earnings_news_count'] = 0
        # Initialize PCA features to 0
        for i in range(10):
            features[f'news_pca_{i}'] = 0.0

        news_features_computed = False

        # Try historical embeddings first
        if self.news_embeddings is not None and self.news_pca is not None:
            # Get news for this symbol in 7-day window before earnings
            lookback_days = 7
            earn_dt = pd.Timestamp(earnings_date)
            start_dt = earn_dt - timedelta(days=lookback_days)
            end_dt = earn_dt - timedelta(days=1)

            symbol_news = self.news_embeddings[
                (self.news_embeddings['symbol'] == symbol) &
                (self.news_embeddings['trading_date'] >= start_dt) &
                (self.news_embeddings['trading_date'] <= end_dt)
            ]

            features['pre_earnings_news_count'] = len(symbol_news)

            if len(symbol_news) > 0:
                # Get embedding columns and compute mean
                emb_cols = [c for c in symbol_news.columns if c.startswith('emb_')]
                mean_emb = symbol_news[emb_cols].mean().values.reshape(1, -1)

                # Apply PCA
                pca_features = self.news_pca.transform(mean_emb)[0]
                for i, val in enumerate(pca_features):
                    features[f'news_pca_{i}'] = float(val)
                news_features_computed = True

        # Fall back to live news fetching if no historical data and FMP enabled
        if not news_features_computed and LIVE_NEWS_ENABLED and self.news_pca is not None:
            try:
                from .live_news import get_live_news_pca_features
                news_count, pca_features = get_live_news_pca_features(
                    symbol=symbol,
                    earnings_date=earnings_date,
                    pca_model=self.news_pca,
                    lookback_days=7,
                )
                features['pre_earnings_news_count'] = news_count
                for i, val in enumerate(pca_features):
                    features[f'news_pca_{i}'] = float(val)
                if news_count > 0:
                    logger.debug(f"{symbol}: Used live news ({news_count} articles)")
            except Exception as e:
                logger.warning(f"{symbol}: Live news fetch failed: {e}")

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
        # Compute features
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
        # Check historical moves
        if self.historical_moves is None:
            return "Historical moves dataset not loaded"

        symbol_moves = self.historical_moves[
            (self.historical_moves['symbol'] == symbol) &
            (self.historical_moves['earnings_date'] < pd.Timestamp(earnings_date))
        ]

        if len(symbol_moves) == 0:
            # Check if symbol exists at all in dataset
            all_symbol_moves = self.historical_moves[
                self.historical_moves['symbol'] == symbol
            ]
            if len(all_symbol_moves) == 0:
                return f"No historical earnings data for {symbol} in dataset"
            else:
                return f"No earnings history before {earnings_date} ({len(all_symbol_moves)} future events exist)"

        return f"OK - {len(symbol_moves)} historical earnings events"

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


# Singleton instance
_predictor: Optional[EarningsPredictor] = None


def get_predictor() -> EarningsPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = EarningsPredictor()
    return _predictor
