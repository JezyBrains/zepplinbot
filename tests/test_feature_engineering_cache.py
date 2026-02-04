
import unittest
import os
import pandas as pd
import time
from unittest.mock import patch
from src.feature_engineering import FeatureEngine

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngine()
        self.test_file = 'tests/test_round_timing.csv'
        # Create a dummy file
        with open(self.test_file, 'w') as f:
            f.write("timestamp,stake\n")
            f.write("2023-01-01 10:00:00,100\n")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_caching_logic(self):
        """Test that data is cached and only reloaded when file changes."""
        with patch('src.feature_engineering.ROUND_DATA_FILE', self.test_file):

            # 1. Initial load
            with patch('pandas.read_csv', side_effect=pd.read_csv) as mock_read:
                df1 = self.fe.load_round_data()
                self.assertEqual(len(df1), 1)
                # Should be called once initially
                self.assertEqual(mock_read.call_count, 1)

            # 2. Second load - Should use cache
            with patch('pandas.read_csv', side_effect=pd.read_csv) as mock_read:
                df2 = self.fe.load_round_data()
                self.assertEqual(len(df2), 1)
                # Should NOT be called if cached
                self.assertEqual(mock_read.call_count, 0)

    def test_cache_update_on_change(self):
        """Test that cache is invalidated when file changes."""
        with patch('src.feature_engineering.ROUND_DATA_FILE', self.test_file):
            # Load initial
            self.fe.load_round_data()

            # Modify file
            time.sleep(1.1) # Ensure mtime changes (some filesystems have 1s resolution)
            with open(self.test_file, 'a') as f:
                f.write("2023-01-01 10:01:00,200\n")

            # Reload
            df = self.fe.load_round_data()
            self.assertEqual(len(df), 2)

if __name__ == '__main__':
    unittest.main()
