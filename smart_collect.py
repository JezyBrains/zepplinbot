#!/usr/bin/env python3
"""
Smart Data Collection

Understands your data format:
- First line = NEWEST crash (most recent)
- Last line = OLDEST crash (historical)

Automatically detects duplicates and only adds new data!
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
            # Remove any trailing dashes or extra chars
            value = value.rstrip('-').strip()
            crash = float(value)
            if 1.0 <= crash <= 1000.0:  # Reasonable range
                crashes.append(crash)
        except ValueError:
            # Skip invalid values silently
            continue
    
    return crashes

def add_to_dataset(crashes):
    """Add crashes to dataset.
    
    Assumes crashes[0] is NEWEST, crashes[-1] is OLDEST.
    Automatically detects and skips duplicates.
    """
    if not crashes:
        return None
    
    # Load existing data
    try:
        df_existing = pd.read_csv('data/zeppelin_data.csv')
        df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
        existing_values = df_existing['value'].tolist()
        last_timestamp = df_existing['timestamp'].iloc[-1]
    except FileNotFoundError:
        # Create new dataset
        df_existing = pd.DataFrame(columns=['timestamp', 'value'])
        existing_values = []
        last_timestamp = datetime.now()
    
    # Reverse: newest-first → oldest-first (chronological)
    crashes_chronological = list(reversed(crashes))
    
    # Find new crashes (not in existing dataset)
    new_crashes = []
    for crash in crashes_chronological:
        if crash not in existing_values:
            new_crashes.append(crash)
            existing_values.append(crash)  # Track to avoid duplicates within batch
    
    if not new_crashes:
        return df_existing, 0
    
    # Add new crashes
    new_data = []
    for i, crash in enumerate(new_crashes):
        timestamp = last_timestamp + timedelta(minutes=i + 1)
        new_data.append({
            'timestamp': timestamp,
            'value': crash
        })
    
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([df_existing, new_df], ignore_index=True)
    combined_df.to_csv('data/zeppelin_data.csv', index=False)
    
    return combined_df, len(new_crashes)

def main():
    print("="*70)
    print("SMART DATA COLLECTION")
    print("="*70)
    print("\n📋 How to use:")
    print("   1. Copy crash values from Zeppelin")
    print("   2. Paste them here (first line = newest crash)")
    print("   3. Type 'done' on a new line when finished")
    print("   4. Script auto-detects duplicates!")
    print("\n💡 Format understood:")
    print("   1.26x - last crash    ← NEWEST")
    print("   5.96x")
    print("   1.15x")
    print("   ...                   ← OLDEST")
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
    print(f"   Newest: {crashes[0]}x (will be last in dataset)")
    print(f"   Oldest: {crashes[-1]}x (will be added first)")
    
    # Add to dataset
    result = add_to_dataset(crashes)
    
    if result is None:
        print("\n❌ Error adding to dataset")
        return
    
    df, new_count = result
    
    if new_count == 0:
        print(f"\n⚠️  No new crashes added (all {len(crashes)} already in dataset)")
        print(f"   Total dataset: {len(df)} crashes")
        print(f"   Last crash: {df['value'].iloc[-1]}x")
    else:
        skipped = len(crashes) - new_count
        print(f"\n✅ Added {new_count} NEW crashes to dataset")
        if skipped > 0:
            print(f"   Skipped {skipped} duplicate(s)")
        print(f"   Total dataset: {len(df)} crashes")
        print(f"   Last crash: {df['value'].iloc[-1]}x")
        
        print("\n📈 Quick stats:")
        print(f"   Min: {df['value'].min():.2f}x")
        print(f"   Max: {df['value'].max():.2f}x")
        print(f"   Mean: {df['value'].mean():.2f}x")
        print(f"   Median: {df['value'].median():.2f}x")
    
    print("\n🚀 Next steps:")
    print("   python3 monitor_dashboard.py --bankroll 100 --target 2.0")

if __name__ == "__main__":
    main()
