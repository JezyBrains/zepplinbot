#!/usr/bin/env python3
"""
Auto Data Collection from Clipboard

Monitors clipboard for crash values and auto-adds them.
Just copy from game, script detects and adds automatically!
"""

import pandas as pd
from datetime import datetime, timedelta
import time
import re
import pyperclip
import sys

class ClipboardMonitor:
    """Monitors clipboard for crash values."""
    
    def __init__(self):
        self.last_clipboard = ""
        self.collected_crashes = []
    
    def parse_crashes(self, text):
        """Extract crash values from text."""
        # Pattern: number followed by 'x' (e.g., 2.05x, 1.47x)
        pattern = r'(\d+\.?\d*)x'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        crashes = []
        for match in matches:
            try:
                crash = float(match)
                if 1.0 <= crash <= 100.0:  # Reasonable range
                    crashes.append(crash)
            except ValueError:
                continue
        
        return crashes
    
    def add_to_dataset(self, crashes):
        """Add crashes to dataset."""
        if not crashes:
            return None
        
        # Load existing data
        try:
            df_existing = pd.read_csv('data/zeppelin_data.csv')
            df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
            last_timestamp = df_existing['timestamp'].iloc[-1]
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=['timestamp', 'value'])
            last_timestamp = datetime.now()
        
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
    
    def monitor(self):
        """Monitor clipboard for crash values."""
        print("="*70)
        print("AUTO CLIPBOARD MONITOR")
        print("="*70)
        print("\n🎯 Monitoring clipboard for crash values...")
        print("   Just copy crash values from game!")
        print("   Script will auto-detect and add them")
        print("\n💡 Tips:")
        print("   - Copy multiple values at once (e.g., 2.05x 1.47x 4.13x)")
        print("   - Or copy one at a time")
        print("   - Press Ctrl+C to stop monitoring")
        print("="*70)
        
        try:
            while True:
                try:
                    clipboard = pyperclip.paste()
                    
                    if clipboard != self.last_clipboard:
                        self.last_clipboard = clipboard
                        
                        # Try to parse crashes
                        crashes = self.parse_crashes(clipboard)
                        
                        if crashes:
                            print(f"\n✅ Detected {len(crashes)} crash(es): {crashes}")
                            
                            # Add to dataset
                            df = self.add_to_dataset(crashes)
                            
                            if df is not None:
                                self.collected_crashes.extend(crashes)
                                print(f"   Added to dataset (Total: {len(df)} crashes)")
                                print(f"   Session collected: {len(self.collected_crashes)} crashes")
                
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("MONITORING STOPPED")
            print("="*70)
            print(f"\n📊 Session Summary:")
            print(f"   Collected: {len(self.collected_crashes)} crashes")
            
            if self.collected_crashes:
                print(f"   Values: {self.collected_crashes}")
                print("\n🚀 Run analysis:")
                print("   python3 monitor_dashboard.py --bankroll 100 --target 2.0")

if __name__ == "__main__":
    try:
        import pyperclip
        monitor = ClipboardMonitor()
        monitor.monitor()
    except ImportError:
        print("❌ Error: pyperclip not installed")
        print("\nInstall with: pip3 install pyperclip")
        print("\nOr use quick_collect.py instead:")
        print("   python3 quick_collect.py")
