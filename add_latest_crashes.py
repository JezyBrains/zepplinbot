#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timedelta

# New crash data (chronological order, 2.61 is most recent)
new_crashes_text = """2.61x
2.10x
1.17x
2.90x
1.15x
4.93x
2.60x
2.08x
1.10x
2.55x
1.78x
1.32x
1.00x
1.28x
2.09x
2.23x
2.97x
1.52x
1.03x
3.29x
4.07x
1.14x
2.03x
1.12x
1.65x
1.00x
3.80x
1.70x
1.42x
3.03x
4.99x
3.55x
1.74x
1.48x
1.59x
1.16x
3.08x
1.93x
1.15x
1.31x
5.36x
2.05x
1.05x
2.12x
6.22x
1.17x
1.77x
2.40x
1.19x
25.17x
2.38x
1.58x
4.50x
2.52x
3.96x
1.56x
1.43x
1.21x
8.58x
1.51x
3.28x
2.35x
5.94x
1.58x
1.28x
2.16x
23.84x
25.95x
1.06x
3.80x
1.13x
11.28x
1.56x
1.32x
1.12x
3.21x
2.25x
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
1.65x"""

# Parse crashes
crashes = [float(line.replace('x', '')) for line in new_crashes_text.strip().split('\n') if line.strip()]

print(f"📊 Parsed {len(crashes)} new crash points")
print(f"   Oldest: {crashes[0]}x")
print(f"   Most recent: {crashes[-1]}x")

# Load existing data
df_existing = pd.read_csv('data/zeppelin_data.csv')
df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])

print(f"\n📁 Current dataset: {len(df_existing)} crashes")

# Create new entries with timestamps
last_timestamp = df_existing['timestamp'].iloc[-1]
new_data = []

for i, crash in enumerate(crashes):
    timestamp = last_timestamp + timedelta(minutes=i + 1)
    new_data.append({
        'timestamp': timestamp,
        'value': crash
    })

new_df = pd.DataFrame(new_data)

# Combine
combined_df = pd.concat([df_existing, new_df], ignore_index=True)
combined_df.to_csv('data/zeppelin_data.csv', index=False)

print(f"\n✅ Updated dataset: {len(combined_df)} total crashes")
print(f"\n📈 Statistics:")
print(f"   Min: {combined_df['value'].min():.2f}x")
print(f"   Max: {combined_df['value'].max():.2f}x")
print(f"   Mean: {combined_df['value'].mean():.2f}x")
print(f"   Median: {combined_df['value'].median():.2f}x")
print(f"\n🎯 Last 10 crashes: {combined_df['value'].tail(10).tolist()}")
