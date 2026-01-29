import os
import sys
import pytest
import pandas as pd
import time
from unittest.mock import patch, MagicMock

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import feature_engineering
from feature_engineering import FeatureEngine

# We use a temporary file for testing to avoid messing with real data
TEST_DATA_FILE = 'data/test_round_timing.csv'

class TestFeatureEngineCache:

    @pytest.fixture
    def feature_engine(self):
        return FeatureEngine()

    @pytest.fixture
    def setup_data(self):
        # Create directory if needed
        os.makedirs(os.path.dirname(TEST_DATA_FILE), exist_ok=True)

        # Create initial data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='h'),
            'stake': range(10),
            'multiplier': [1.0] * 10
        })
        df.to_csv(TEST_DATA_FILE, index=False)
        yield
        # Cleanup
        if os.path.exists(TEST_DATA_FILE):
            os.remove(TEST_DATA_FILE)

    def test_load_data_caching(self, feature_engine, setup_data):
        with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
            # First load - should hit disk
            df1 = feature_engine.load_round_data()
            assert len(df1) == 10
            assert feature_engine._cached_df is not None

            # Modify file
            # Ensure mtime changes (wait at least 10ms for filesystems with low res,
            # though usually 1s is safer for ext3/4 but in containers mostly fine)
            time.sleep(0.1)

            # Force update mtime by writing new content
            df_new = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='h'),
                'stake': range(20),
                'multiplier': [1.0] * 20
            })
            df_new.to_csv(TEST_DATA_FILE, index=False)

            # Ensure file system registered the change
            new_stats = os.stat(TEST_DATA_FILE)
            if new_stats.st_mtime == feature_engine._last_mtime:
                 # If mtime didn't change (fast execution), force manual update or sleep longer
                 # But size changed (10 rows vs 20 rows), so it should trigger anyway.
                 pass

            # Second load - should detect change and reload
            df2 = feature_engine.load_round_data()
            assert len(df2) == 20
            assert len(feature_engine._cached_df) == 20

            # Third load - no change, should be fast
            # We mock pd.read_csv to ensure it's NOT called
            with patch('pandas.read_csv', side_effect=pd.read_csv) as mock_read:
                feature_engine.load_round_data()
                mock_read.assert_not_called()

    def test_cache_immutability(self, feature_engine, setup_data):
        """Test that modifying returned dataframe doesn't corrupt cache"""
        with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
            df1 = feature_engine.load_round_data()
            df1['new_col'] = 100

            df2 = feature_engine.load_round_data()
            assert 'new_col' not in df2.columns

    def test_missing_file(self, feature_engine):
        with patch('feature_engineering.ROUND_DATA_FILE', 'non_existent_file.csv'):
            df = feature_engine.load_round_data()
            assert df.empty
            assert feature_engine._cached_df is None
