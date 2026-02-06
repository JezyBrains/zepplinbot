import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngine

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        self.engine = FeatureEngine()

    @patch('feature_engineering.pd.read_csv')
    @patch('feature_engineering.os.stat')
    @patch('feature_engineering.os.path.exists')
    def test_caching_behavior(self, mock_exists, mock_stat, mock_read_csv):
        # Setup mocks
        mock_exists.return_value = True

        # Mock stat object
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = 1000
        mock_stat_obj.st_size = 500
        mock_stat.return_value = mock_stat_obj

        # Mock dataframe
        # Note: pd.to_datetime will be called on 'timestamp' col
        mock_df = pd.DataFrame({
            'a': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        mock_read_csv.return_value = mock_df

        # 1. First call - should load from CSV
        df1 = self.engine.load_round_data()
        self.assertEqual(len(df1), 3)
        mock_read_csv.assert_called_once()

        # 2. Second call - same mtime/size - should use cache
        mock_read_csv.reset_mock()
        df2 = self.engine.load_round_data()
        self.assertEqual(len(df2), 3)
        mock_read_csv.assert_not_called()

        # 3. Third call - changed mtime - should reload
        mock_stat_obj.st_mtime = 1001
        mock_read_csv.reset_mock()
        df3 = self.engine.load_round_data()
        self.assertEqual(len(df3), 3)
        mock_read_csv.assert_called_once()

        # 4. Fourth call - changed size - should reload
        mock_stat_obj.st_size = 501
        mock_read_csv.reset_mock()
        df4 = self.engine.load_round_data()
        self.assertEqual(len(df4), 3)
        mock_read_csv.assert_called_once()

    @patch('feature_engineering.pd.read_csv')
    @patch('feature_engineering.os.stat')
    @patch('feature_engineering.os.path.exists')
    def test_cache_immutability(self, mock_exists, mock_stat, mock_read_csv):
        # Ensure that modifying the returned dataframe doesn't affect the cache
        mock_exists.return_value = True
        mock_stat.return_value.st_mtime = 1000
        mock_stat.return_value.st_size = 500

        mock_df = pd.DataFrame({
            'val': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-01', '2023-01-01']
        })
        mock_read_csv.return_value = mock_df

        # Load data
        df1 = self.engine.load_round_data()

        # Modify returned df
        df1.loc[0, 'val'] = 999

        # Load again (from cache)
        df2 = self.engine.load_round_data()

        # Check if cache was affected (it shouldn't be, because we return copy)
        self.assertEqual(df2.loc[0, 'val'], 1)
        self.assertNotEqual(df2.loc[0, 'val'], df1.loc[0, 'val'])

if __name__ == '__main__':
    unittest.main()
