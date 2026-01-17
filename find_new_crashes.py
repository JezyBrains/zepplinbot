#!/usr/bin/env python3
import pandas as pd

# Load current dataset
df = pd.read_csv('data/zeppelin_data.csv')
last_crash = df['value'].iloc[-1]

print(f"Last crash in dataset: {last_crash}x\n")

# Your pasted data
data_text = """5.88x
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
4.14x
1.00x
1.34x
1.96x
3.10x
18.45x
1.46x
2.96x
2.04x
20.02x
1.95x
1.02x
1.94x
1.00x
1.38x
1.84x
1.02x
3.32x
3.02x
4.29x
2.17x
1.13x
1.00x
8.22x
1.15x
2.84x
2.13x
1.55x
5.37x
4.01x
8.01x
2.33x"""

crashes = [float(line.replace('x', '').strip()) for line in data_text.strip().split('\n')]

print(f"Total crashes in paste: {len(crashes)}")
print(f"First (newest): {crashes[0]}x")
print(f"Last (oldest): {crashes[-1]}x\n")

# Find where last_crash appears
found_idx = -1
for i, crash in enumerate(crashes):
    if crash == last_crash:
        found_idx = i
        break

if found_idx >= 0:
    new_crashes = crashes[:found_idx]
    print(f"✅ Found last crash ({last_crash}x) at position {found_idx}\n")
    print(f"🆕 NEW crashes (newest to oldest):")
    for i, crash in enumerate(new_crashes):
        print(f"   {i+1}. {crash}x")
    print(f"\n📊 Total NEW crashes: {len(new_crashes)}")
else:
    print(f"⚠️  Last crash ({last_crash}x) not found in pasted data")
    print(f"   Showing all crashes - you may need to identify manually")
