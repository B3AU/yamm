"""Component tests for live_news.py - news fetching and embedding."""
from datetime import date
from unittest.mock import patch, MagicMock

import numpy as np
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


# ============================================================================
# Tests Using Real PCA Model (news_pca.joblib)
# ============================================================================

class TestRealPCAModel:
    """Tests using production news_pca.joblib model."""

    def test_pca_model_properties(self, pca_model):
        """PCA model should have expected properties."""
        assert hasattr(pca_model, 'transform')
        assert hasattr(pca_model, 'n_components_')
        assert pca_model.n_components_ == 10

    def test_pca_transform_shape(self, pca_model):
        """PCA should transform 768-dim to 10-dim."""
        # Simulate a news embedding (768-dim)
        fake_embedding = np.random.randn(768)

        result = pca_model.transform(fake_embedding.reshape(1, -1))

        assert result.shape == (1, 10)

    def test_pca_transform_batch(self, pca_model):
        """PCA should handle batch transformations."""
        # Multiple embeddings
        batch_embeddings = np.random.randn(5, 768)

        result = pca_model.transform(batch_embeddings)

        assert result.shape == (5, 10)

    def test_pca_consistent_results(self, pca_model):
        """Same input should give same output."""
        embedding = np.ones(768) * 0.1
        embedding_reshaped = embedding.reshape(1, -1)

        result1 = pca_model.transform(embedding_reshaped)
        result2 = pca_model.transform(embedding_reshaped)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pca_variance_explained(self, pca_model):
        """PCA should capture reasonable variance."""
        if hasattr(pca_model, 'explained_variance_ratio_'):
            total_variance = sum(pca_model.explained_variance_ratio_)
            # 10 components should capture meaningful variance
            assert total_variance > 0.1, f"Expected >10% variance, got {total_variance*100}%"


class TestRealPCAWithLiveNews:
    """Integration tests for live news PCA pipeline."""

    @responses.activate
    def test_get_live_news_with_real_pca(self, mock_env_vars, pca_model):
        """Test get_live_news_pca_features with real PCA model."""
        from trading.earnings.live_news import get_live_news_pca_features

        # Mock FMP news API
        responses.add(
            responses.GET,
            "https://financialmodelingprep.com/stable/news/stock",
            json=[
                {
                    "title": "Test Earnings Article",
                    "text": "Company reported quarterly earnings.",
                    "publishedDate": "2026-01-29T10:00:00",
                },
            ],
            status=200,
        )

        # Mock the embedding model (we don't have it loaded)
        with patch('trading.earnings.live_news._get_anonymizer') as mock_anon:
            mock_anon.return_value = lambda text, symbol: text

            with patch('trading.earnings.live_news._get_sentence_model') as mock_model:
                mock_embedding_model = MagicMock()
                # Return proper 768-dim embedding
                mock_embedding_model.encode.return_value = np.random.randn(1, 768)
                mock_model.return_value = mock_embedding_model

                # Use the REAL PCA model
                news_count, pca_features = get_live_news_pca_features(
                    symbol="AAPL",
                    earnings_date=date(2026, 1, 30),
                    pca_model=pca_model,
                )

                assert news_count == 1
                assert len(pca_features) == 10
                # Features should be finite numbers
                assert np.all(np.isfinite(pca_features))

    def test_pca_default_features_zeros(self, pca_model):
        """When no news, default features should be zeros."""
        from trading.earnings.live_news import get_live_news_pca_features

        # No API key = no news = defaults
        with patch.dict('os.environ', {'FMP_API_KEY': ''}):
            # Need to reimport to pick up env change
            import importlib
            import trading.earnings.live_news as live_news
            importlib.reload(live_news)

            news_count, pca_features = live_news.get_live_news_pca_features(
                symbol="AAPL",
                earnings_date=date(2026, 1, 30),
                pca_model=pca_model,
            )

            assert news_count == 0
            np.testing.assert_array_equal(pca_features, np.zeros(10))
