"""Live news fetching and embedding for earnings predictions.

Fetches recent news from FMP API, anonymizes text, generates embeddings,
and applies PCA for model features. Uses same FMP format as training data.
"""
from __future__ import annotations

import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import requests

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# FMP API (same as training data source)
FMP_API_KEY = os.getenv('FMP_API_KEY', '')
FMP_NEWS_URL = 'https://financialmodelingprep.com/stable/news/stock'

# Lazy-loaded components
_sentence_model = None
_anonymizer = None
_universe_df = None


def _get_sentence_model():
    """Lazy load sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            logger.info("Loaded sentence transformer: BAAI/bge-base-en-v1.5")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _sentence_model


def _get_universe():
    """Load universe for anonymization."""
    global _universe_df
    if _universe_df is None:
        from pathlib import Path
        import pandas as pd
        universe_path = Path(__file__).parent.parent.parent / 'data' / 'universe.pqt'
        if universe_path.exists():
            _universe_df = pd.read_parquet(universe_path)
            logger.info(f"Loaded universe: {len(_universe_df)} symbols")
        else:
            # Fallback to empty dataframe
            _universe_df = pd.DataFrame(columns=['symbol', 'Security Name'])
            logger.warning("Universe file not found, anonymization will be limited")
    return _universe_df


def _get_anonymizer():
    """Lazy load anonymizer with name map."""
    global _anonymizer
    if _anonymizer is None:
        try:
            import sys
            from pathlib import Path
            # Add project root to path for imports
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from anonymize_news import build_name_map, Anonymizer
            universe = _get_universe()
            name_map = build_name_map(universe)
            _anonymizer = Anonymizer(name_map)
            logger.info(f"Initialized anonymizer with {len(name_map)} symbols")
        except ImportError as e:
            logger.error(f"Could not import anonymization module: {e}")
            raise
    return _anonymizer


def fetch_fmp_news(
    symbol: str,
    from_date: date,
    to_date: date,
    limit: int = 100,
) -> list[dict]:
    """Fetch company news from FMP API.

    Args:
        symbol: Stock ticker
        from_date: Start date (inclusive)
        to_date: End date (inclusive)
        limit: Max articles to fetch

    Returns:
        List of news articles with keys: title, text, publishedDate, url, site
    """
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not set, skipping news fetch")
        return []

    params = {
        'symbols': symbol,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'limit': limit,
        'apikey': FMP_API_KEY,
    }

    try:
        r = requests.get(FMP_NEWS_URL, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json()

        if not isinstance(articles, list):
            logger.warning(f"Unexpected FMP response for {symbol}: {type(articles)}")
            return []

        logger.debug(f"Fetched {len(articles)} articles for {symbol}")
        return articles

    except requests.exceptions.RequestException as e:
        logger.error(f"FMP API error for {symbol}: {e}")
        return []


def anonymize_and_embed(
    articles: list[dict],
    symbol: str,
) -> Optional[np.ndarray]:
    """Anonymize article text and generate mean embedding.

    Uses same text format as training: title + text combined.

    Args:
        articles: List of news articles from FMP
        symbol: Target symbol for anonymization

    Returns:
        Mean embedding (768-dim) or None if no articles
    """
    if not articles:
        return None

    try:
        anon = _get_anonymizer()
    except Exception as e:
        logger.error(f"{symbol}: Failed to load anonymizer: {e}")
        return None

    try:
        model = _get_sentence_model()
    except Exception as e:
        logger.error(f"{symbol}: Failed to load sentence transformer model: {e}")
        return None

    # Combine title + text for each article (same as training)
    texts = []
    for article in articles:
        # Handle None values from FMP API (keys can exist with None value)
        title = article.get('title') or ''
        text = article.get('text') or ''
        combined = f"{title}. {text}" if text else title
        if combined and combined.strip():
            # Anonymize to replace company names with tokens
            anon_text = anon(combined, symbol)
            texts.append(anon_text)

    if not texts:
        return None

    # Generate embeddings
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Return mean embedding
        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb
    except Exception as e:
        logger.error(f"Embedding error for {symbol}: {e}")
        return None


def get_live_news_pca_features(
    symbol: str,
    earnings_date: date,
    pca_model,
    lookback_days: int = 7,
    return_headlines: bool = False,
) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, list[dict]]:
    """Fetch live news and compute PCA features for prediction.

    Args:
        symbol: Stock ticker
        earnings_date: Upcoming earnings date
        pca_model: Fitted PCA model from training
        lookback_days: Days before earnings to fetch news
        return_headlines: If True, also return raw headline data for LLM sanity check

    Returns:
        If return_headlines=False: Tuple of (news_count, pca_features_array)
        If return_headlines=True: Tuple of (news_count, pca_features_array, headlines_list)
        pca_features_array is zeros if no news found
    """
    n_components = pca_model.n_components_ if pca_model else 10
    default_features = np.zeros(n_components)
    empty_headlines: list[dict] = []

    if not FMP_API_KEY:
        logger.debug(f"FMP_API_KEY not set, using default news features for {symbol}")
        if return_headlines:
            return 0, default_features, empty_headlines
        return 0, default_features

    # Fetch news for lookback window
    to_date = earnings_date - timedelta(days=1)  # Day before earnings
    from_date = earnings_date - timedelta(days=lookback_days)

    articles = fetch_fmp_news(symbol, from_date, to_date)

    # Extract headlines for LLM sanity check (limit to 10 most recent)
    headlines = [
        {
            "ts": a.get("publishedDate"),
            "source": a.get("site"),
            "title": a.get("title"),
        }
        for a in articles[:10]
    ]

    if not articles:
        logger.debug(f"No news found for {symbol} in {lookback_days}-day window")
        if return_headlines:
            return 0, default_features, empty_headlines
        return 0, default_features

    # Get mean embedding
    mean_emb = anonymize_and_embed(articles, symbol)

    if mean_emb is None:
        if return_headlines:
            return len(articles), default_features, headlines
        return len(articles), default_features

    # Apply PCA
    try:
        pca_features = pca_model.transform(mean_emb.reshape(1, -1))[0]
        logger.debug(f"Generated PCA features for {symbol} from {len(articles)} articles")
        if return_headlines:
            return len(articles), pca_features, headlines
        return len(articles), pca_features
    except Exception as e:
        logger.error(f"PCA transform error for {symbol}: {e}")
        if return_headlines:
            return len(articles), default_features, headlines
        return len(articles), default_features
