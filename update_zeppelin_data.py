#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime, timedelta

# Chronological order - OLDEST to NEWEST (23.84x is the most recent)
data_text = """23.84x
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
1.00x"""

# Extract numbers
lines = data_text.strip().split('\n')
values = [float(line.replace('x', '')) for line in lines if line.strip()]

print(f"📊 Extracted {len(values)} Zeppelin crash points")
print(f"   Oldest: {values[0]}x")
print(f"   Latest: {values[-1]}x (balloon just popped here)")

# Create timestamps (chronological order)
base_time = datetime.now()
data = []

for i, value in enumerate(values):
    timestamp = base_time - timedelta(minutes=len(values) - i)
    data.append({
        'timestamp': timestamp,
        'value': value
    })

df = pd.DataFrame(data)

# Save to NEW file (overwrite old data)
df.to_csv('data/zeppelin_data.csv', index=False)

print(f"\n✅ Saved {len(df)} crash points to data/zeppelin_data.csv")
print(f"\n📈 Statistics:")
print(f"   Min crash: {df['value'].min()}x")
print(f"   Max crash: {df['value'].max()}x")
print(f"   Mean crash: {df['value'].mean():.2f}x")
print(f"   Median crash: {df['value'].median():.2f}x")
print(f"\n🎯 Last 5 crashes: {df['value'].tail(5).tolist()}")
print(f"\n🔮 Ready to predict next crash point!")
