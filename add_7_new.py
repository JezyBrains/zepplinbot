#!/usr/bin/env python3
import pandas as pd
from datetime import timedelta

# 7 new crashes (newest to oldest)
new_crashes = [1.00, 1.19, 8.92, 2.47, 1.30, 8.67, 3.07]

# Reverse to chronological order (oldest to newest)
new_crashes = list(reversed(new_crashes))

# Load existing
df = pd.read_csv('data/zeppelin_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
last_timestamp = df['timestamp'].iloc[-1]

# Add new
new_data = []
for i, crash in enumerate(new_crashes):
    timestamp = last_timestamp + timedelta(minutes=i + 1)
    new_data.append({'timestamp': timestamp, 'value': crash})

new_df = pd.DataFrame(new_data)
combined_df = pd.concat([df, new_df], ignore_index=True)
combined_df.to_csv('data/zeppelin_data.csv', index=False)

print(f"✅ Added 7 new crashes")
print(f"   Total: {len(combined_df)} crashes")
print(f"   Last crash: {combined_df['value'].iloc[-1]}x")
