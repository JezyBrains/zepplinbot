
import unittest
import pandas as pd
import os
import shutil
import tempfile
import sys
from unittest.mock import patch

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from feature_engineering import FeatureEngine

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and file
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_round_timing.csv')

        # Patch the global ROUND_DATA_FILE in feature_engineering module
        self.patcher = patch('feature_engineering.ROUND_DATA_FILE', self.test_file)
        self.patcher.start()

        # Create initial data
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='min'),
            'stake': [100.0] * 10
        })
        self.df.to_csv(self.test_file, index=False)

        self.engine = FeatureEngine()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_load_data_caching(self):
        # First load - should hit disk
        with patch('pandas.read_csv', wraps=pd.read_csv) as mock_read_csv:
            df1 = self.engine.load_round_data()
            self.assertEqual(len(df1), 10)
            mock_read_csv.assert_called_once()

        # Second load - should use cache
        with patch('pandas.read_csv', wraps=pd.read_csv) as mock_read_csv:
            df2 = self.engine.load_round_data()
            self.assertEqual(len(df2), 10)
            mock_read_csv.assert_not_called()

        # Verify returned objects are different (copies)
        self.assertIsNot(df1, df2)

    def test_cache_invalidation_on_change(self):
        # Load initial data
        self.engine.load_round_data()

        # Modify file (append data)
        # Ensure timestamp is different enough or just size change
        new_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='min'),
            'stake': [100.0] * 20
        })

        # Wait a bit to ensure mtime change if OS resolution is low (though usually unnecessary with write)
        # But we'll rely on size change too.
        new_df.to_csv(self.test_file, index=False)

        # Should reload
        with patch('pandas.read_csv', wraps=pd.read_csv) as mock_read_csv:
            df3 = self.engine.load_round_data()
            self.assertEqual(len(df3), 20)
            mock_read_csv.assert_called_once()

    def test_limit_slicing(self):
        # Initial load
        self.engine.load_round_data()

        # Load with limit - should use cache but return slice
        with patch('pandas.read_csv', wraps=pd.read_csv) as mock_read_csv:
            df_limit = self.engine.load_round_data(limit=5)
            self.assertEqual(len(df_limit), 5)
            mock_read_csv.assert_not_called()

if __name__ == '__main__':
    unittest.main()
