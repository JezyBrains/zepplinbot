import pytest
import pandas as pd
import numpy as np
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.append('src')
from feature_engineering import FeatureEngine

# Constants
TEST_DATA_FILE = 'data/test_round_timing.csv'
# Mock the global ROUND_DATA_FILE in feature_engineering for testing
# We'll do this by patching or just setting it if possible, but patching is safer
# Since ROUND_DATA_FILE is a global variable in the module, we need to patch it.

@pytest.fixture
def test_data_file():
    # Setup
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='min'),
        'stake': np.random.uniform(10, 100, 10),
        'multiplier': np.random.uniform(1.0, 5.0, 10)
    })
    df.to_csv(TEST_DATA_FILE, index=False)
    yield TEST_DATA_FILE
    # Teardown
    if os.path.exists(TEST_DATA_FILE):
        os.remove(TEST_DATA_FILE)

@pytest.fixture
def feature_engine():
    return FeatureEngine()

def test_cache_hit(test_data_file, feature_engine):
    """Test that subsequent calls use the cache."""
    with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
        # First call - should load from file
        with patch('pandas.read_csv', side_effect=pd.read_csv) as mock_read:
            df1 = feature_engine.load_round_data()
            assert len(df1) == 10
            assert mock_read.call_count == 1

            # Second call - should use cache
            df2 = feature_engine.load_round_data()
            assert len(df2) == 10
            # call_count should still be 1
            assert mock_read.call_count == 1

            # Verify data is same
            pd.testing.assert_frame_equal(df1, df2)

def test_cache_invalidation_mtime(test_data_file, feature_engine):
    """Test that modifying the file (mtime change) invalidates cache."""
    with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
        # Load initial data
        df1 = feature_engine.load_round_data()
        assert len(df1) == 10

        # Modify file (touch it to update mtime, maybe wait a bit to ensure mtime change)
        time.sleep(0.1)
        # We need to actually change content or just touch.
        # If we just touch, size is same, but mtime changes.
        os.utime(TEST_DATA_FILE, None)

        # We need to verify that read_csv is called again.
        with patch('pandas.read_csv', side_effect=pd.read_csv) as mock_read:
            df2 = feature_engine.load_round_data()
            assert mock_read.call_count == 1 # Should read again because mtime changed

        pd.testing.assert_frame_equal(df1, df2)

def test_cache_invalidation_size(test_data_file, feature_engine):
    """Test that changing file size invalidates cache."""
    with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
        # Load initial data
        df1 = feature_engine.load_round_data()
        assert len(df1) == 10

        # Append data to file
        df_new = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01 00:10', periods=5, freq='min'),
            'stake': np.random.uniform(10, 100, 5),
            'multiplier': np.random.uniform(1.0, 5.0, 5)
        })
        df_new.to_csv(TEST_DATA_FILE, mode='a', header=False, index=False)

        # Load again
        df2 = feature_engine.load_round_data()
        assert len(df2) == 15

        # Should be different
        assert len(df2) != len(df1)

def test_cache_integrity(test_data_file, feature_engine):
    """Test that modifying the returned DataFrame does not corrupt the cache."""
    with patch('feature_engineering.ROUND_DATA_FILE', TEST_DATA_FILE):
        df1 = feature_engine.load_round_data()
        original_val = df1.iloc[0, 1]

        # Modify returned dataframe
        df1.iloc[0, 1] = 9999.99

        # Load again
        df2 = feature_engine.load_round_data()

        # Check that cache was not modified
        assert df2.iloc[0, 1] == original_val
        assert df2.iloc[0, 1] != 9999.99
