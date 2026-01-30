import pytest
import pandas as pd
import numpy as np
import os
import time
import shutil
from datetime import datetime, timedelta
import sys

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngine, ROUND_DATA_FILE

# Mock data configuration
NUM_ROWS = 100000
DATA_DIR = os.path.dirname(ROUND_DATA_FILE)

@pytest.fixture(scope="module", autouse=True)
def setup_data():
    """Create a dummy CSV file for testing."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Backup existing file if any
    backup_file = None
    if os.path.exists(ROUND_DATA_FILE):
        backup_file = ROUND_DATA_FILE + ".bak"
        shutil.copy(ROUND_DATA_FILE, backup_file)

    # Generate data
    dates = [datetime.now() - timedelta(minutes=i) for i in range(NUM_ROWS)]
    df = pd.DataFrame({
        'timestamp': dates,
        'stake': np.random.uniform(1, 100, NUM_ROWS),
        'multiplier': np.random.uniform(1, 10, NUM_ROWS)
    })
    df.to_csv(ROUND_DATA_FILE, index=False)

    yield

    # Cleanup
    if os.path.exists(ROUND_DATA_FILE):
        os.remove(ROUND_DATA_FILE)
    if backup_file:
        shutil.move(backup_file, ROUND_DATA_FILE)

def test_load_round_data_performance():
    """Measure performance of load_round_data."""
    engine = FeatureEngine()

    # First load
    start_time = time.time()
    df1 = engine.load_round_data()
    duration1 = time.time() - start_time

    # Second load
    start_time = time.time()
    df2 = engine.load_round_data()
    duration2 = time.time() - start_time

    print(f"\nLoad 1 time: {duration1:.4f}s")
    print(f"Load 2 time: {duration2:.4f}s")

    assert len(df1) == NUM_ROWS
    assert len(df2) == NUM_ROWS

def test_cache_invalidation():
    """Verify that data updates when file changes."""
    engine = FeatureEngine()

    # Initial load
    df1 = engine.load_round_data()
    initial_len = len(df1)

    # Append a row to the file
    with open(ROUND_DATA_FILE, 'a') as f:
        f.write(f"\n{datetime.now()},50.0,2.0")

    # Force mtime update (filesystem resolution might be low)
    os.utime(ROUND_DATA_FILE, None)

    # Load again
    df2 = engine.load_round_data()

    assert len(df2) == initial_len + 1, "Data should reflect file update"
