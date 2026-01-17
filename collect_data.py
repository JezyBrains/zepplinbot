import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import pandas as pd
from src.selenium_scraper import SeleniumScraper, GameDataCollector
from src.data_scraper import DataManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Data Collection Tool for Secured Websites')
    parser.add_argument('--url', type=str, 
                       default='https://zeppelin-tz.betsolutions.com/?sessionId=f1a5dbea-1b31-49e6-96cf-1e9592e3358c&Lang=en-US&MerchantId=40001&IsFreeplay=False&platform=desktop&gameId=7001&flags=undefined',
                       help='Website URL')
    parser.add_argument('--mode', type=str, choices=['interactive', 'auto', 'manual', 'monitor'],
                       default='interactive',
                       help='Data collection mode')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in minutes for auto/monitor mode')
    parser.add_argument('--interval', type=int, default=10,
                       help='Check interval in seconds for monitor mode')
    parser.add_argument('--selector', type=str,
                       help='Custom CSS selector for data extraction')
    parser.add_argument('--output', type=str, default='data/collected_data.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SECURED WEBSITE DATA COLLECTOR")
    print("="*60)
    print(f"\nMode: {args.mode}")
    print(f"URL: {args.url}")
    
    data_manager = DataManager(data_file=args.output)
    df = pd.DataFrame()
    
    if args.mode == 'interactive':
        print("\n📊 INTERACTIVE MODE")
        print("You'll be able to interact with the page and capture data manually")
        
        scraper = SeleniumScraper(headless=False)
        df = scraper.interactive_scraping_session(args.url)
        
    elif args.mode == 'auto':
        print("\n🤖 AUTO MODE")
        print("Browser will open, wait 30 seconds for you to complete verification")
        
        scraper = SeleniumScraper(headless=False)
        numbers = scraper.manual_data_collection(args.url, wait_seconds=30)
        
        if numbers:
            df = pd.DataFrame({
                'timestamp': pd.Timestamp.now(),
                'value': numbers
            })
        
        scraper.close()
        
    elif args.mode == 'manual':
        print("\n✍️  MANUAL ENTRY MODE")
        print("Enter data manually without opening browser")
        
        collector = GameDataCollector(args.url)
        df = collector.manual_entry_mode()
        
    elif args.mode == 'monitor':
        print("\n👁️  MONITOR MODE")
        print(f"Will monitor the page for {args.duration} minutes")
        
        collector = GameDataCollector(args.url)
        df = collector.collect_game_results(
            duration_minutes=args.duration,
            check_interval=args.interval
        )
    
    if not df.empty:
        data_manager.save_data(df, append=True)
        print(f"\n✅ Saved {len(df)} data points to {args.output}")
        
        print("\n📈 Data Summary:")
        print(f"   Total points: {len(df)}")
        if 'value' in df.columns:
            print(f"   Min value: {df['value'].min()}")
            print(f"   Max value: {df['value'].max()}")
            print(f"   Mean value: {df['value'].mean():.2f}")
            print(f"\n   Latest values: {df['value'].tail(10).tolist()}")
    else:
        print("\n⚠️  No data collected")
    
    print("\n" + "="*60)
    print("Collection Complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review collected data: {args.output}")
    print(f"2. Run prediction: python main.py --data-file {args.output} --steps 5 --visualize")


if __name__ == '__main__':
    main()
