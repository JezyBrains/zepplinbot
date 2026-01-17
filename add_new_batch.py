#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timedelta

# New crashes (4.63 is most recent)
new_crashes_text = """4.63x
2.99x
1.54x
1.01x
16.83x
1.47x
1.34x
1.45x
1.51x
47.49x
1.24x
3.40x
3.46x
1.30x
30.75x
2.04x
1.59x
1.19x
2.80x
1.07x
6.20x
2.13x
1.48x
1.76x
1.32x
1.36x
1.04x
1.22x
2.35x
8.05x
1.00x
8.69x
2.61x
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
23.84x"""

crashes = [float(line.replace('x', '')) for line in new_crashes_text.strip().split('\n') if line.strip()]

# Reverse to get chronological order (23.84 oldest, 4.63 newest)
crashes = list(reversed(crashes))

print(f"📊 Parsed {len(crashes)} new crashes")
print(f"   Oldest: {crashes[0]}x")
print(f"   Most recent: {crashes[-1]}x")

# Load existing
df_existing = pd.read_csv('data/zeppelin_data.csv')
df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])

print(f"\n📁 Current dataset: {len(df_existing)} crashes")

# Add new
last_timestamp = df_existing['timestamp'].iloc[-1]
new_data = []

for i, crash in enumerate(crashes):
    timestamp = last_timestamp + timedelta(minutes=i + 1)
    new_data.append({
        'timestamp': timestamp,
        'value': crash
    })

new_df = pd.DataFrame(new_data)
combined_df = pd.concat([df_existing, new_df], ignore_index=True)
combined_df.to_csv('data/zeppelin_data.csv', index=False)

print(f"\n✅ Updated dataset: {len(combined_df)} total crashes")
print(f"   Last crash: {combined_df['value'].iloc[-1]}x")
print(f"\n📈 Statistics:")
print(f"   Min: {combined_df['value'].min():.2f}x")
print(f"   Max: {combined_df['value'].max():.2f}x")
print(f"   Mean: {combined_df['value'].mean():.2f}x")
print(f"   Median: {combined_df['value'].median():.2f}x")
