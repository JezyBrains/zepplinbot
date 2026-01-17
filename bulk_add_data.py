#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime, timedelta
from src.data_scraper import DataManager

# Parse the data
data_text = """2.25x
4.04x
1.19x
1.63x
2.67x
1.63x
1.16x
4.06x
1.25x
2.19x
10.88x
1.00x
1.65x
1.44x
3.79x
3.52x
1.00x
14.13x
3.51x
1.59x
1.08x
1.65x
1.65x
1.55x
3.97x
2.49x
6.21x
5.70x
1.00x
1.50x
1.93x
1.02x
2.36x
1.64x
1.44x
3.50x
4.11x
3.49x
1.00x
1.19x
3.02x
1.89x
2.48x
1.60x
1.48x
2.51x
4.47x
1.26x
1.20x
1.10x
1.00x
43.42x
1.33x
1.00x
1.07x
1.05x
1.06x
1.27x
1.84x
3.97x
2.54x
3.04x
3.60x
1.00x
3.20x
1.10x
37.60x
1.08x
13.86x
2.68x
1.70x
1.81x
2.28x
1.47x
1.28x
1.33x
4.62x
7.82x
5.35x
1.13x
1.33x
1.86x
1.09x
1.50x
2.09x
1.00x
49.82x
1.65x
1.00x
4.21x
33.59x
7.42x
1.17x
1.00x
4.34x
2.34x
1.44x
3.40x
1.17x"""

# Extract numbers
lines = data_text.strip().split('\n')
values = [float(line.replace('x', '')) for line in lines if line.strip()]

print(f"Extracted {len(values)} values")

# Create timestamps (going backwards from now)
base_time = datetime.now()
data = []

for i, value in enumerate(values):
    timestamp = base_time - timedelta(minutes=len(values) - i)
    data.append({
        'timestamp': timestamp,
        'value': value
    })

df = pd.DataFrame(data)

# Save to file
data_manager = DataManager(data_file='data/zeppelin_data.csv')
data_manager.save_data(df, append=True)

print(f"\n✅ Added {len(df)} data points to data/zeppelin_data.csv")
print(f"\n📊 Statistics:")
print(f"   Min: {df['value'].min()}")
print(f"   Max: {df['value'].max()}")
print(f"   Mean: {df['value'].mean():.2f}")
print(f"   Median: {df['value'].median():.2f}")
print(f"\n📈 Latest 10 values: {df['value'].tail(10).tolist()}")
