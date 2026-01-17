#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from datetime import datetime
from src.data_scraper import DataManager


def manual_data_entry():
    print("="*60)
    print("MANUAL DATA COLLECTION - Zeppelin Game")
    print("="*60)
    print("\nEnter the game results as you see them.")
    print("Commands:")
    print("  - Type a number and press Enter to add it")
    print("  - Type 'undo' to remove the last entry")
    print("  - Type 'show' to see all entries")
    print("  - Type 'done' when finished")
    print("="*60)
    
    data = []
    
    while True:
        try:
            user_input = input("\nEnter result: ").strip().lower()
            
            if user_input == 'done':
                break
            elif user_input == 'undo':
                if data:
                    removed = data.pop()
                    print(f"❌ Removed: {removed['value']}")
                    print(f"   Total entries: {len(data)}")
                else:
                    print("⚠️  No data to remove")
                continue
            elif user_input == 'show':
                if data:
                    print(f"\n📊 Current entries ({len(data)} total):")
                    recent = data[-10:] if len(data) > 10 else data
                    for i, entry in enumerate(recent, 1):
                        print(f"   {i}. {entry['value']}")
                    if len(data) > 10:
                        print(f"   ... and {len(data) - 10} more")
                else:
                    print("⚠️  No data collected yet")
                continue
            elif user_input == '':
                continue
            
            try:
                value = float(user_input)
                data.append({
                    'timestamp': datetime.now(),
                    'value': value
                })
                print(f"✅ Added: {value} (Total: {len(data)} entries)")
            except ValueError:
                print(f"❌ Invalid input. Please enter a number.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'done' to save or continue entering data.")
            continue
    
    if data:
        df = pd.DataFrame(data)
        
        data_manager = DataManager(data_file='data/zeppelin_data.csv')
        data_manager.save_data(df, append=True)
        
        print("\n" + "="*60)
        print("✅ DATA SAVED SUCCESSFULLY")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Total entries: {len(df)}")
        print(f"   Min value: {df['value'].min()}")
        print(f"   Max value: {df['value'].max()}")
        print(f"   Mean value: {df['value'].mean():.2f}")
        print(f"   File: data/zeppelin_data.csv")
        
        print(f"\n📈 Latest 10 values:")
        print(f"   {df['value'].tail(10).tolist()}")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1️⃣  Make predictions:")
        print("   python3 main.py --data-file data/zeppelin_data.csv --steps 5 --visualize")
        
        print("\n2️⃣  Collect more data:")
        print("   python3 manual_collect.py")
        
        print("\n3️⃣  Evaluate accuracy:")
        print("   python3 main.py --data-file data/zeppelin_data.csv --evaluate")
        
        if len(df) < 50:
            print(f"\n💡 TIP: You have {len(df)} entries. Collect at least 50-100 for better predictions.")
        
    else:
        print("\n⚠️  No data collected.")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    try:
        manual_data_entry()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
