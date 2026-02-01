import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from feature_engineering import FeatureEngine, ROUND_DATA_FILE

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        self.feature_engine = FeatureEngine()
        self.mock_df = pd.DataFrame({
            'timestamp': ['2023-01-01 10:00:00'],
            'stake': [100.0]
        })
        self.mock_df['timestamp'] = pd.to_datetime(self.mock_df['timestamp'])

    @patch('feature_engineering.pd.read_csv')
    @patch('feature_engineering.os.stat')
    @patch('feature_engineering.os.path.exists')
    def test_caching_behavior(self, mock_exists, mock_stat, mock_read_csv):
        # Setup mocks
        mock_exists.return_value = True

        # Initial file stats
        mock_stat_1 = MagicMock()
        mock_stat_1.st_mtime = 1000
        mock_stat_1.st_size = 500
        mock_stat.return_value = mock_stat_1

        mock_read_csv.return_value = self.mock_df.copy()

        # 1. First call - should load from file
        df1 = self.feature_engine.load_round_data()
        self.assertEqual(len(df1), 1)
        mock_read_csv.assert_called_once()

        # 2. Second call - should use cache (no new read_csv call)
        df2 = self.feature_engine.load_round_data()
        self.assertEqual(len(df2), 1)
        mock_read_csv.assert_called_once()  # Call count should still be 1

        # 3. Modify file stats - should reload
        mock_stat_2 = MagicMock()
        mock_stat_2.st_mtime = 1001 # Changed mtime
        mock_stat_2.st_size = 500
        mock_stat.return_value = mock_stat_2

        df3 = self.feature_engine.load_round_data()
        self.assertEqual(len(df3), 1)
        self.assertEqual(mock_read_csv.call_count, 2) # Should have called read_csv again

        # 4. Modify returned DF - should not affect cache
        df3['new_col'] = 1

        # Reset stats to avoid reload
        mock_stat.return_value = mock_stat_2

        df4 = self.feature_engine.load_round_data()
        self.assertNotIn('new_col', df4.columns)
        self.assertEqual(mock_read_csv.call_count, 2) # Still 2 calls

if __name__ == '__main__':
    unittest.main()
