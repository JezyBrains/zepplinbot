#!/usr/bin/env python3
"""
Quick Data Collection Tool

Easiest way to add crash data:
1. Copy crash values from game
2. Paste into this tool
3. Automatically added to dataset

No OBS setup needed!
"""

import pandas as pd
from datetime import datetime, timedelta
import sys

def parse_crashes(text):
    """Parse crash values from text input."""
    crashes = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove 'x' suffix and any extra characters
        try:
            value = line.replace('x', '').replace('X', '').strip()
            crash = float(value)
            crashes.append(crash)
        except ValueError:
            print(f"⚠️  Skipping invalid value: {line}")
            continue
    
    return crashes

def add_to_dataset(crashes):
    """Add crashes to dataset.
    
    Assumes crashes[0] is NEWEST, crashes[-1] is OLDEST.
    Reverses them to add in chronological order.
    """
    # Reverse: newest-first → oldest-first (chronological)
    crashes = list(reversed(crashes))
    
    # Load existing data
    try:
        df_existing = pd.read_csv('data/zeppelin_data.csv')
        df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
        last_timestamp = df_existing['timestamp'].iloc[-1]
        last_value = df_existing['value'].iloc[-1]
    except FileNotFoundError:
        # Create new dataset
        df_existing = pd.DataFrame(columns=['timestamp', 'value'])
        last_timestamp = datetime.now()
        last_value = None
    
    # Check for duplicates - skip crashes already in dataset
    if last_value is not None:
        # Find where new data starts (skip duplicates)
        start_idx = 0
        for i, crash in enumerate(crashes):
            if crash == last_value:
                start_idx = i + 1
                break
        
        if start_idx > 0:
            print(f"   ⚠️  Skipping {start_idx} duplicate(s) already in dataset")
            crashes = crashes[start_idx:]
    
    if not crashes:
        print("   ℹ️  All crashes already in dataset")
        return df_existing
    
    # Add new crashes
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
    
    return combined_df

def main():
    print("="*70)
    print("QUICK DATA COLLECTION")
    print("="*70)
    print("\n📋 How to use:")
    print("   1. Copy crash values from Zeppelin (e.g., 2.05x, 1.47x, etc.)")
    print("   2. Paste them here (one per line or all together)")
    print("   3. Type 'done' on a new line when finished")
    print("   4. Data automatically added to dataset")
    print("\n💡 Tip: Copy the most recent crashes first")
    print("="*70)
    
    print("\n📝 Paste crash values (type 'done' when finished):\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'done':
                break
            lines.append(line)
        except EOFError:
            break
    
    if not lines:
        print("\n❌ No data entered")
        return
    
    text = '\n'.join(lines)
    crashes = parse_crashes(text)
    
    if not crashes:
        print("\n❌ No valid crash values found")
        return
    
    print(f"\n📊 Parsed {len(crashes)} crash values")
    print(f"   First (newest): {crashes[0]}x")
    print(f"   Last (oldest): {crashes[-1]}x")
    
    print(f"\n💡 Assuming {crashes[0]}x is the MOST RECENT crash")
    print(f"   (Data will be added in chronological order)")
    
    # Add to dataset
    df = add_to_dataset(crashes)
    
    print(f"\n✅ Added {len(crashes)} crashes to dataset")
    print(f"   Total dataset: {len(df)} crashes")
    print(f"   Last crash: {df['value'].iloc[-1]}x")
    
    print("\n📈 Quick stats:")
    print(f"   Min: {df['value'].min():.2f}x")
    print(f"   Max: {df['value'].max():.2f}x")
    print(f"   Mean: {df['value'].mean():.2f}x")
    print(f"   Median: {df['value'].median():.2f}x")
    
    print("\n🚀 Next steps:")
    print("   python3 monitor_dashboard.py --bankroll 100 --target 2.0")
    print("   python3 kelly_dashboard.py --bankroll 100")

if __name__ == "__main__":
    main()
