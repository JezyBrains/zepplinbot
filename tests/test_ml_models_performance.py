import numpy as np
import pytest
from src.models.ml_models import XGBoostPredictor, LightGBMPredictor

# Reference implementation of the original create_features for XGBoost
def original_xgboost_create_features(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]

        features = [
            np.mean(sequence),
            np.std(sequence),
            np.min(sequence),
            np.max(sequence),
            sequence[-1],
            sequence[-1] - sequence[-2] if len(sequence) > 1 else 0,
            np.mean(sequence[-5:]) if len(sequence) >= 5 else np.mean(sequence),
            np.mean(sequence[-10:]) if len(sequence) >= 10 else np.mean(sequence),
        ]
        features.extend(sequence[-10:].tolist() if len(sequence) >= 10 else sequence.tolist())

        X.append(features)
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Reference implementation of the original create_features for LightGBM
def original_lightgbm_create_features(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]

        features = [
            np.mean(sequence),
            np.std(sequence),
            np.min(sequence),
            np.max(sequence),
            sequence[-1],
            sequence[-1] - sequence[-2] if len(sequence) > 1 else 0,
            np.mean(sequence[-5:]) if len(sequence) >= 5 else np.mean(sequence),
        ]
        features.extend(sequence[-10:].tolist() if len(sequence) >= 10 else sequence.tolist())

        X.append(features)
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

class TestMLModelsPerformance:
    def test_xgboost_create_features_correctness(self):
        """Verify that the optimized XGBoost feature creation matches the original."""
        np.random.seed(42)
        # Test with sufficient data
        data = np.random.rand(100)
        sequence_length = 50

        predictor = XGBoostPredictor(sequence_length=sequence_length)

        # Get expected output from original implementation
        X_expected, y_expected = original_xgboost_create_features(data, sequence_length)

        # Get actual output from (to be) optimized implementation
        # Note: At this point, if not yet optimized, this test passes trivially if logic is same.
        # But after optimization, it ensures correctness.
        X_actual, y_actual = predictor.create_features(data)

        assert np.allclose(y_expected, y_actual), "y values mismatch"
        assert np.allclose(X_expected, X_actual), "X features mismatch"

        # Test with data length exactly sequence_length + 1 (minimum for 1 sample)
        data_min = np.random.rand(sequence_length + 1)
        X_exp_min, y_exp_min = original_xgboost_create_features(data_min, sequence_length)
        X_act_min, y_act_min = predictor.create_features(data_min)

        assert np.allclose(y_exp_min, y_act_min), "y values mismatch (min data)"
        assert np.allclose(X_exp_min, X_act_min), "X features mismatch (min data)"

    def test_lightgbm_create_features_correctness(self):
        """Verify that the optimized LightGBM feature creation matches the original."""
        np.random.seed(43)
        data = np.random.rand(100)
        sequence_length = 50

        predictor = LightGBMPredictor(sequence_length=sequence_length)

        X_expected, y_expected = original_lightgbm_create_features(data, sequence_length)
        X_actual, y_actual = predictor.create_features(data)

        assert np.allclose(y_expected, y_actual), "y values mismatch"
        assert np.allclose(X_expected, X_actual), "X features mismatch"

    def test_xgboost_create_features_short_data(self):
        """Test with short data (less than 10 but more than seq len if seq len is small)."""
        # If sequence_length is small, say 5.
        sequence_length = 5
        data = np.random.rand(20)
        predictor = XGBoostPredictor(sequence_length=sequence_length)

        X_expected, y_expected = original_xgboost_create_features(data, sequence_length)
        X_actual, y_actual = predictor.create_features(data)

        assert np.allclose(y_expected, y_actual)
        assert np.allclose(X_expected, X_actual)

    def test_create_features_insufficient_data(self):
        """Test error handling for insufficient data."""
        predictor = XGBoostPredictor(sequence_length=50)
        data = np.random.rand(10) # Less than sequence_length

        # The original code returns empty arrays if loop doesn't run,
        # but the train method raises ValueError if len(X) == 0.
        # create_features returns empty arrays.
        X, y = predictor.create_features(data)
        assert len(X) == 0
        assert len(y) == 0
