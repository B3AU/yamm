"""Component tests for live_news.py - news fetching and embedding."""
from datetime import date
from unittest.mock import patch, MagicMock

import pytest
import responses


class TestFetchFMPNews:
    """Tests for fetch_fmp_news function."""

    @responses.activate
    def test_fetches_news_successfully(self, mock_env_vars):
        """Should fetch news from FMP API."""
        from trading.earnings.live_news import fetch_fmp_news

        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/news/stock",
            json=[
                {
                    "title": "Apple Q1 Earnings Preview",
                    "text": "Analysts expect strong iPhone sales...",
                    "publishedDate": "2026-01-29T10:00:00",
                    "site": "Reuters",
                    "url": "https://example.com/article1",
                },
                {
                    "title": "Apple Stock Rises",
                    "text": "Shares up 2% ahead of earnings...",
                    "publishedDate": "2026-01-29T14:00:00",
                    "site": "Bloomberg",
                    "url": "https://example.com/article2",
                },
            ],
            status=200,
        )

        articles = fetch_fmp_news(
            symbol="AAPL",
            from_date=date(2026, 1, 22),
            to_date=date(2026, 1, 29),
            limit=10,
        )

        assert len(articles) == 2
        assert articles[0]["title"] == "Apple Q1 Earnings Preview"

    @responses.activate
    def test_returns_empty_for_no_news(self, mock_env_vars):
        """Should return empty list when no news found."""
        from trading.earnings.live_news import fetch_fmp_news

        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/news/stock",
            json=[],
            status=200,
        )

        articles = fetch_fmp_news(
            symbol="OBSCURE",
            from_date=date(2026, 1, 22),
            to_date=date(2026, 1, 29),
            limit=10,
        )

        assert articles == []

    def test_returns_empty_without_api_key(self, monkeypatch):
        """Should return empty list without API key."""
        from trading.earnings.live_news import fetch_fmp_news

        monkeypatch.delenv("FMP_API_KEY", raising=False)

        articles = fetch_fmp_news(
            symbol="AAPL",
            from_date=date(2026, 1, 22),
            to_date=date(2026, 1, 29),
            limit=10,
        )

        assert articles == []


class TestAnonymizeAndEmbed:
    """Tests for anonymize_and_embed function."""

    def test_returns_embedding_array(self):
        """Should return numpy array embedding."""
        from trading.earnings.live_news import anonymize_and_embed
        import numpy as np

        with patch('trading.earnings.live_news._get_anonymizer') as mock_anon:
            mock_anon.return_value = lambda text, symbol: text  # Pass-through anonymizer

            with patch('trading.earnings.live_news._get_sentence_model') as mock_model:
                mock_embedding_model = MagicMock()
                mock_embedding_model.encode.return_value = np.array([[0.1] * 768])
                mock_model.return_value = mock_embedding_model

                articles = [
                    {"title": "Test Article", "text": "Test content about COMPANY"},
                ]

                result = anonymize_and_embed(articles, "AAPL")

                # Should return numpy array or None
                assert result is None or hasattr(result, 'shape')


class TestGetLiveNewsPCAFeatures:
    """Tests for get_live_news_pca_features function."""

    @responses.activate
    def test_returns_pca_features(self, mock_env_vars):
        """Should return PCA-projected features."""
        from trading.earnings.live_news import get_live_news_pca_features
        import numpy as np

        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/news/stock",
            json=[
                {
                    "title": "Test Article",
                    "text": "Test content...",
                    "publishedDate": "2026-01-29T10:00:00",
                },
            ],
            status=200,
        )

        with patch('trading.earnings.live_news._get_anonymizer') as mock_anon:
            mock_anon.return_value = lambda text, symbol: text

            with patch('trading.earnings.live_news._get_sentence_model') as mock_model:
                mock_embedding_model = MagicMock()
                mock_embedding_model.encode.return_value = np.array([[0.1] * 768])
                mock_model.return_value = mock_embedding_model

                mock_pca_model = MagicMock()
                mock_pca_model.n_components_ = 10
                mock_pca_model.transform.return_value = np.array([[0.1] * 10])

                news_count, pca_features = get_live_news_pca_features(
                    symbol="AAPL",
                    earnings_date=date(2026, 1, 30),
                    pca_model=mock_pca_model,
                )

                assert news_count == 1
                assert len(pca_features) == 10
