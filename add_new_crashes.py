#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime, timedelta

# Read existing data
existing_df = pd.read_csv('data/zeppelin_data.csv')
existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

print(f"📊 Current dataset: {len(existing_df)} crash points")
print(f"   Last crash in dataset: {existing_df['value'].iloc[-1]}x")

# New crash points (in chronological order)
new_crashes = [3.96, 1.56, 1.43, 1.21, 8.58, 1.51, 3.28, 2.35, 5.94, 1.58, 1.28, 2.16]

print(f"\n🆕 Adding {len(new_crashes)} new crash points")
print(f"   Actual crashes: {new_crashes}")
print(f"   Previous prediction was: 3.38x")
print(f"   First actual crash was: {new_crashes[0]}x")

# Create new entries
last_timestamp = existing_df['timestamp'].iloc[-1]
new_data = []

for i, value in enumerate(new_crashes):
    timestamp = last_timestamp + timedelta(minutes=i + 1)
    new_data.append({
        'timestamp': timestamp,
        'value': value
    })

new_df = pd.DataFrame(new_data)

# Combine and save
combined_df = pd.concat([existing_df, new_df], ignore_index=True)
combined_df.to_csv('data/zeppelin_data.csv', index=False)

print(f"\n✅ Updated dataset: {len(combined_df)} total crash points")
print(f"\n📈 New statistics:")
print(f"   Min crash: {combined_df['value'].min()}x")
print(f"   Max crash: {combined_df['value'].max()}x")
print(f"   Mean crash: {combined_df['value'].mean():.2f}x")
print(f"   Median crash: {combined_df['value'].median():.2f}x")
print(f"\n🎯 Last 10 crashes: {combined_df['value'].tail(10).tolist()}")
