import os
import time
import pandas as pd
import unittest
from unittest.mock import patch
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from feature_engineering import FeatureEngine, ROUND_DATA_FILE

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        self.engine = FeatureEngine()
        self.test_file = ROUND_DATA_FILE

        # Create a dummy CSV file
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='min'),
            'stake': [100] * 10
        })
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)
        self.df.to_csv(self.test_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_caching_behavior(self):
        """Test that subsequent calls use the cache."""
        # We'll rely on the side effect that read_csv is slow/observable,
        # or use mocking. Mocking is better.

        with patch('pandas.read_csv') as mock_read:
            # Configure mock to return a DataFrame similar to what we expect
            # We need to make sure the timestamp conversion works, so we return string timestamps
            df_ret = self.df.copy()
            df_ret['timestamp'] = df_ret['timestamp'].astype(str)
            mock_read.return_value = df_ret

            # Reset engine to ensure clean state for this test block
            engine = FeatureEngine()

            # First call
            engine.load_round_data()
            self.assertEqual(mock_read.call_count, 1, "Should call read_csv on first load")

            # Second call
            engine.load_round_data()
            self.assertEqual(mock_read.call_count, 1, "Should NOT call read_csv on second load")

    def test_cache_invalidation_on_file_change(self):
        """Test that modifying the file invalidates the cache."""
        # Load first
        df1 = self.engine.load_round_data()
        self.assertEqual(len(df1), 10)

        # Modify file
        # We wait a bit to ensure mtime differs significantly if filesystem has low resolution
        time.sleep(1.1)

        new_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='min'),
            'stake': [100] * 20
        })
        new_df.to_csv(self.test_file, index=False)

        # Second load should detect change and reload
        df2 = self.engine.load_round_data()
        self.assertEqual(len(df2), 20)

    def test_copy_is_returned(self):
        """Test that the cache is safe from external mutation."""
        df1 = self.engine.load_round_data()
        df1['new_col'] = 1

        df2 = self.engine.load_round_data()
        self.assertNotIn('new_col', df2.columns)

if __name__ == '__main__':
    unittest.main()
