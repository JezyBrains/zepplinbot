
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import time
import sys
import shutil

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from feature_engineering import FeatureEngine

class TestFeatureEngineCache(unittest.TestCase):
    def setUp(self):
        self.engine = FeatureEngine()
        self.test_dir = 'tests/temp_data'
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, 'test_round_data.csv')

        # Create a dummy CSV
        self.df_initial = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10),
            'stake': range(10)
        })
        self.df_initial.to_csv(self.test_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_caching_behavior(self):
        # We need to patch the variable in the module where it is defined
        with patch('feature_engineering.ROUND_DATA_FILE', self.test_file):
            # 1. Initial load
            # We spy on pd.read_csv. Note: feature_engineering imports pandas as pd
            with patch('feature_engineering.pd.read_csv', side_effect=pd.read_csv) as mock_read_csv:
                df1 = self.engine.load_round_data()
                self.assertEqual(len(df1), 10)
                mock_read_csv.assert_called()
                initial_call_count = mock_read_csv.call_count

                # 2. Cached load
                df2 = self.engine.load_round_data()
                self.assertEqual(len(df2), 10)
                # Should not call read_csv again
                self.assertEqual(mock_read_csv.call_count, initial_call_count)

                # 3. Modify file
                # Ensure mtime changes. Some filesystems have 1s resolution, so we might need to wait or force mtime update.
                # Just writing a larger file works if size checks are in place.
                df_new = pd.DataFrame({
                    'timestamp': pd.date_range(start='2023-01-01', periods=20),
                    'stake': range(20)
                })
                # Ensure mtime is at least 1 second different if possible, or rely on size change
                time.sleep(1.1)
                df_new.to_csv(self.test_file, index=False)

                # 4. Reload (should trigger read_csv)
                df3 = self.engine.load_round_data()
                self.assertEqual(len(df3), 20)
                self.assertGreater(mock_read_csv.call_count, initial_call_count)

    def test_limit_slicing_with_cache(self):
        with patch('feature_engineering.ROUND_DATA_FILE', self.test_file):
            # Load fully first
            df1 = self.engine.load_round_data()
            self.assertEqual(len(df1), 10)

            # Load with limit
            df2 = self.engine.load_round_data(limit=5)
            self.assertEqual(len(df2), 5)

            # Verify it is the tail
            # We need to ensure types match for assertion
            pd.testing.assert_frame_equal(
                df2.reset_index(drop=True),
                df1.tail(5).reset_index(drop=True)
            )

            # Verify cache was not modified (df1 should still be full size if we retrieved it again)
            # Actually df1 is a copy returned by first call.
            # Let's verify internal cache is full size if we can inspect it,
            # or just call load_round_data() again without limit
            df3 = self.engine.load_round_data()
            self.assertEqual(len(df3), 10)

if __name__ == '__main__':
    unittest.main()
