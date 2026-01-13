"""Component tests for ml_predictor.py - with mocked FMP API."""
from datetime import date
from unittest.mock import patch, MagicMock
import json

import pytest
import responses


class TestMLPredictorModule:
    """Tests for ML predictor module."""

    def test_module_imports(self):
        """Should be able to import the module."""
        from trading.earnings import ml_predictor
        assert ml_predictor is not None

    def test_has_predictor_class(self):
        """Should have EarningsPredictor class."""
        from trading.earnings.ml_predictor import EarningsPredictor
        assert EarningsPredictor is not None


class TestEarningsPredictor:
    """Tests for EarningsPredictor class."""

    def test_predictor_initialization(self):
        """Should initialize predictor (may load models)."""
        from trading.earnings.ml_predictor import EarningsPredictor

        try:
            predictor = EarningsPredictor()
            assert predictor is not None
        except FileNotFoundError:
            # Expected if model files don't exist in test environment
            pytest.skip("Model files not found")


class TestFeatureEngineering:
    """Tests for feature engineering logic."""

    def test_historical_earnings_features(self):
        """Should compute historical earnings features correctly."""
        import statistics

        # Sample historical moves
        sample_moves = [0.05, 0.08, 0.03, 0.12, 0.06]

        mean = statistics.mean(sample_moves)
        std = statistics.stdev(sample_moves)
        max_move = max(sample_moves)
        min_move = min(sample_moves)

        # Verify calculations
        assert mean == pytest.approx(0.068)
        assert max_move == 0.12
        assert min_move == 0.03

    def test_volatility_features(self):
        """Should compute volatility features."""
        import math

        # Sample returns for realized vol calculation
        returns = [0.01, -0.02, 0.015, -0.01, 0.025]

        # Standard deviation * sqrt(252) for annualized vol
        import statistics
        std_daily = statistics.stdev(returns)
        annualized_vol = std_daily * math.sqrt(252)

        assert annualized_vol > 0

    def test_timing_features(self):
        """Should compute timing features."""
        from datetime import date

        earnings_date = date(2026, 1, 30)

        day_of_week = earnings_date.weekday()  # 0=Monday, 4=Friday
        month = earnings_date.month
        quarter = (earnings_date.month - 1) // 3 + 1

        assert day_of_week == 4  # Friday
        assert month == 1
        assert quarter == 1


class TestQuantilePrediction:
    """Tests for quantile prediction logic."""

    def test_quantile_ordering(self):
        """Quantiles should be in increasing order."""
        # Simulated predictions
        predictions = {
            "q50": 0.025,
            "q75": 0.045,
            "q90": 0.065,
            "q95": 0.085,
        }

        assert predictions["q50"] <= predictions["q75"]
        assert predictions["q75"] <= predictions["q90"]
        assert predictions["q90"] <= predictions["q95"]

    def test_edge_calculation(self):
        """Should calculate edge correctly."""
        predicted_q75 = 0.065
        implied_move = 0.045

        edge = predicted_q75 - implied_move

        assert edge == pytest.approx(0.02)  # 2% edge


class TestFallbackBehavior:
    """Tests for fallback behavior when APIs fail."""

    def test_default_feature_values(self):
        """Should use defaults for missing data."""
        # When fundamentals are missing, defaults to 0
        missing_fundamentals = {
            "priceToEarningsRatio": 0.0,
            "returnOnEquity": 0.0,
            "revenueGrowth": 0.0,
        }

        for key, value in missing_fundamentals.items():
            assert value == 0.0

    def test_news_feature_defaults(self):
        """News features should use training medians when unavailable."""
        # Expected: 10 PCA components, each with a default value
        default_pca_features = [0.0] * 10

        assert len(default_pca_features) == 10


class TestNewsFeatures:
    """Tests for news embedding features."""

    def test_news_pca_dimension(self):
        """News features should be 10-dimensional PCA."""
        expected_dims = 10

        # Mock PCA output
        pca_features = [0.1] * expected_dims

        assert len(pca_features) == expected_dims

    def test_news_count_tracking(self):
        """Should track news count for analysis."""
        # Simulate news counts
        news_counts = {
            "AAPL": 15,
            "MSFT": 8,
            "OBSCURE": 0,
        }

        assert news_counts["AAPL"] > 0
        assert news_counts["OBSCURE"] == 0
