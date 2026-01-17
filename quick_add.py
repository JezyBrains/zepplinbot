#!/usr/bin/env python3
import pandas as pd
from datetime import timedelta

# Load current dataset
df = pd.read_csv('data/zeppelin_data.csv')
last_crash = df['value'].iloc[-1]

print(f"Last crash in dataset: {last_crash}x\n")

# Your pasted data
data_text = """2.11x
1.31x
4.67x
1.01x
2.09x
1.03x
1.00x
20.10x
9.06x
1.12x
1.00x
1.24x
7.81x
194.56x
1.15x
4.20x
1.32x
1.72x
1.03x
1.36x
2.10x
1.61x
13.01x
2.32x
1.27x
6.01x
1.00x
1.16x
40.32x
2.15x
1.16x
5.88x
5.36x
1.34x
36.27x
1.00x
1.19x
8.92x
2.47x
1.30x
8.67x
3.07x
2.95x
1.26x
5.96x
1.15x
1.21x
1.07x
78.92x
1.63x
2.84x
1.72x
1.42x
9.10x
1.06x
1.37x
1.00x
1.00x
1.21x
1.69x
1.00x
1.18x
1.07x
1.08x
1.91x
4.41x
10.54x
2.67x
1.22x
1.00x
4.62x
2.18x
1.24x
6.40x
3.96x
2.64x
94.69x
2.36x
2.04x
1.00x
7.83x
3.35x
3.33x
2.65x
5.40x
15.67x
1.00x
58.14x
3.31x
1.19x
4.26x
3.36x
4.51x
1.05x
19.29x
1.71x
4.32x
5.28x
4.14x"""

crashes = [float(line.replace('x', '').strip()) for line in data_text.strip().split('\n')]

# Find where last_crash appears
found_idx = -1
for i, crash in enumerate(crashes):
    if crash == last_crash:
        found_idx = i
        break

if found_idx >= 0:
    new_crashes = crashes[:found_idx]
    print(f"✅ Found last crash ({last_crash}x) at position {found_idx}\n")
    print(f"🆕 NEW crashes: {len(new_crashes)}")
    for i, crash in enumerate(new_crashes):
        print(f"   {i+1}. {crash}x")
    
    # Add them
    new_crashes_chrono = list(reversed(new_crashes))
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    last_timestamp = df['timestamp'].iloc[-1]
    
    new_data = []
    for i, crash in enumerate(new_crashes_chrono):
        timestamp = last_timestamp + timedelta(minutes=i + 1)
        new_data.append({'timestamp': timestamp, 'value': crash})
    
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df.to_csv('data/zeppelin_data.csv', index=False)
    
    print(f"\n✅ Added {len(new_crashes)} crashes")
    print(f"   Total: {len(combined_df)} crashes")
    print(f"   Last crash: {combined_df['value'].iloc[-1]}x")
else:
    print(f"⚠️  Last crash ({last_crash}x) not found")
