
import pytest
import numpy as np
import sys
import os
import yaml
from unittest.mock import MagicMock

# Mock dependencies before import
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['prophet'] = MagicMock()
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.tsa.arima.model'] = MagicMock()
sys.modules['statsmodels.tsa.statespace.sarimax'] = MagicMock()
sys.modules['statsmodels.tsa.holtwinters'] = MagicMock()
sys.modules['pmdarima'] = MagicMock()

# We also need to mock the modules that import these, if they do top-level imports that fail using the mocks above?
# Usually mocking the root package is enough if the submodules are accessed via attributes.
# But if they do `from tensorflow import keras`, `sys.modules['tensorflow']` mock needs to have `keras` attribute.

sys.modules['tensorflow'].keras = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now import
from prediction_engine import PredictionEngine

class TestPredictionEngine:
    @pytest.fixture
    def config_file(self):
        config = {
            'prediction': {
                'models': ['xgboost'],
                'sequence_length': 10,
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 2
                }
            }
        }
        path = 'test_config_engine.yaml'
        with open(path, 'w') as f:
            yaml.dump(config, f)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_evaluate_models_flow(self, config_file):
        engine = PredictionEngine(config_path=config_file)

        # Dummy data
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)

        engine.initialize_models()
        assert 'xgboost' in engine.models

        engine.train_models(data)

        # Evaluate
        test_size = 0.2
        results = engine.evaluate_models(data, test_size=test_size)

        assert 'xgboost' in results
        metrics = results['xgboost']
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
