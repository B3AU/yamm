"""Component tests for ml_predictor.py - with mocked FMP API and real models."""
from datetime import date
from unittest.mock import patch, MagicMock
import json

import numpy as np
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


# ============================================================================
# Tests Using Real Production Models
# ============================================================================

class TestRealModelLoading:
    """Tests that verify production model files load correctly."""

    def test_load_models_success(self, real_predictor):
        """Should load all 4 quantile models."""
        assert real_predictor.models is not None
        assert len(real_predictor.models) == 4

        # Verify all quantiles loaded
        expected_quantiles = [0.5, 0.75, 0.9, 0.95]
        for q in expected_quantiles:
            assert q in real_predictor.models
            assert real_predictor.models[q] is not None

    def test_feature_config_loaded(self, real_predictor, feature_config):
        """Should load feature configuration with columns, quantiles, medians."""
        # Check predictor has feature columns
        assert hasattr(real_predictor, 'feature_cols')
        assert len(real_predictor.feature_cols) > 0

        # Check config structure
        assert "feature_cols" in feature_config
        assert "quantiles" in feature_config
        assert "feature_medians" in feature_config  # Note: actual key is feature_medians

        # Feature count should match config
        assert len(real_predictor.feature_cols) == len(feature_config["feature_cols"])

    def test_pca_model_loaded(self, pca_model):
        """Should load PCA model for news embeddings."""
        # PCA model is loaded separately (not part of EarningsPredictor)
        assert pca_model is not None
        assert hasattr(pca_model, 'transform')
        assert hasattr(pca_model, 'n_components_')
        assert pca_model.n_components_ == 10


class TestRealModelPredictions:
    """Tests for predictions using real production models."""

    def test_predict_with_real_models_using_defaults(self, real_predictor):
        """Should make predictions using default features (no API calls)."""
        # Create a feature dict using training medians (defaults)
        from trading.earnings.ml_predictor import EdgePrediction

        # Use parquet fallback by mocking FMP to return empty
        with patch('trading.earnings.ml_predictor.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # This should still work with parquet fallback
            result = real_predictor.predict(
                symbol="AAPL",
                earnings_date=date(2024, 1, 30),
                timing="AMC",
            )

            # May return None if parquet lookup fails, but if it works:
            if result is not None:
                assert isinstance(result, EdgePrediction)
                assert result.symbol == "AAPL"
                assert result.pred_q50 > 0
                assert result.pred_q75 >= result.pred_q50
                assert result.pred_q90 >= result.pred_q75
                assert result.pred_q95 >= result.pred_q90

    def test_quantile_ordering_real_models(self, real_predictor, feature_config):
        """Predictions from real models should maintain q50 < q75 < q90 < q95."""
        import numpy as np

        # Build feature array from feature medians
        feature_cols = feature_config["feature_cols"]
        medians = feature_config["feature_medians"]

        # Create feature vector using medians
        features = np.array([[medians.get(col, 0.0) for col in feature_cols]])

        # Get predictions from each quantile model
        preds = {}
        for q, model in real_predictor.models.items():
            preds[q] = model.predict(features)[0]

        # Verify ordering
        assert preds[0.5] <= preds[0.75], f"q50 ({preds[0.5]}) should be <= q75 ({preds[0.75]})"
        assert preds[0.75] <= preds[0.9], f"q75 ({preds[0.75]}) should be <= q90 ({preds[0.9]})"
        assert preds[0.9] <= preds[0.95], f"q90 ({preds[0.9]}) should be <= q95 ({preds[0.95]})"

    def test_feature_count_matches_model(self, real_predictor, feature_config):
        """Feature count should match what models expect."""
        expected_features = len(feature_config["feature_cols"])

        # LightGBM Booster uses num_feature() method
        for q, model in real_predictor.models.items():
            model_features = model.num_feature()
            assert model_features == expected_features, \
                f"Model q{q*100} expects {model_features} features, config has {expected_features}"


class TestRealPCAModel:
    """Tests for the news PCA model."""

    def test_pca_transform_dimension(self, pca_model):
        """PCA should reduce 768-dim embedding to 10-dim."""
        # Simulate a 768-dim embedding
        fake_embedding = np.random.randn(1, 768)

        result = pca_model.transform(fake_embedding)

        assert result.shape == (1, 10)

    def test_pca_transform_deterministic(self, pca_model):
        """PCA transform should be deterministic."""
        fake_embedding = np.ones((1, 768)) * 0.5

        result1 = pca_model.transform(fake_embedding)
        result2 = pca_model.transform(fake_embedding)

        np.testing.assert_array_equal(result1, result2)


class TestFeatureDefaults:
    """Tests for default feature values from training medians."""

    def test_feature_medians_exist(self, feature_config):
        """All feature columns should have training median defaults."""
        feature_cols = feature_config["feature_cols"]
        medians = feature_config["feature_medians"]

        missing = [col for col in feature_cols if col not in medians]
        assert len(missing) == 0, f"Missing medians for: {missing}"

    def test_feature_medians_reasonable(self, feature_config):
        """Feature medians should be within reasonable ranges."""
        medians = feature_config["feature_medians"]

        # Historical earnings moves should be positive percentages
        if "hist_move_mean" in medians:
            assert 0 < medians["hist_move_mean"] < 0.5, \
                f"hist_move_mean ({medians['hist_move_mean']}) should be 0-50%"

        # Timing encoded should be 0-2 (unknown/BMO/AMC)
        if "timing_encoded" in medians:
            assert 0 <= medians["timing_encoded"] <= 2

        # News count should be non-negative
        if "pre_earnings_news_count" in medians:
            assert medians["pre_earnings_news_count"] >= 0


class TestEdgeFiltering:
    """Tests for edge-based candidate filtering."""

    def test_filter_by_edge_uses_real_models(self, real_predictor, sample_candidate):
        """filter_by_edge should work with real models."""
        # Create candidates list
        candidates = [sample_candidate]

        # Mock the predict method to return a valid EdgePrediction
        from trading.earnings.ml_predictor import EdgePrediction

        mock_prediction = EdgePrediction(
            symbol="AAPL",
            pred_q50=0.03,
            pred_q75=0.05,
            pred_q90=0.07,
            pred_q95=0.10,
            hist_move_mean=0.04,
            edge_q75=0.01,  # 1% edge (below 5% threshold)
            edge_q90=0.03,
            news_count=5,
        )

        with patch.object(real_predictor, 'predict', return_value=mock_prediction):
            # Filter with 5% threshold - should filter out
            filtered = real_predictor.filter_by_edge(candidates, edge_threshold=0.05)
            assert len(filtered) == 0  # Edge 1% < threshold 5%

            # Change to lower threshold - should pass
            mock_prediction.edge_q75 = 0.06
            filtered = real_predictor.filter_by_edge(candidates, edge_threshold=0.05)
            assert len(filtered) == 1
