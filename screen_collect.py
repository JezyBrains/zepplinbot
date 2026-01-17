#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from src.screen_capture import ScreenCaptureAnalyzer, OBSIntegration, live_preview_mode
from src.data_scraper import DataManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Screen Capture Data Collection')
    parser.add_argument('--mode', type=str, 
                       choices=['setup', 'monitor', 'preview', 'obs'],
                       default='setup',
                       help='Operation mode')
    parser.add_argument('--region', type=str, help='Region name for preview mode')
    parser.add_argument('--duration', type=int, default=10, 
                       help='Duration in minutes for monitor mode')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in seconds')
    parser.add_argument('--video', type=str, help='Path to OBS video file')
    parser.add_argument('--output', type=str, default='data/screen_data.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SCREEN CAPTURE DATA COLLECTOR")
    print("="*60)
    
    analyzer = ScreenCaptureAnalyzer()
    
    if args.mode == 'setup':
        print("\n🔧 SETUP MODE")
        print("\nYou'll define screen regions for the game elements.")
        print("Make sure the Zeppelin game is visible on your screen!")
        
        input("\nPress Enter to start setup...")
        analyzer.setup_wizard()
        
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Test with preview: python3 screen_collect.py --mode preview --region result")
        print("2. Start monitoring: python3 screen_collect.py --mode monitor --duration 15")
        
    elif args.mode == 'monitor':
        print("\n👁️  MONITOR MODE")
        
        if not analyzer.regions:
            print("⚠️  No regions defined. Run setup first:")
            print("   python3 screen_collect.py --mode setup")
            return
        
        print(f"\nMonitoring regions: {list(analyzer.regions.keys())}")
        print("Make sure the game is visible on screen!")
        
        input("\nPress Enter to start monitoring...")
        
        df = analyzer.monitor_regions(
            list(analyzer.regions.keys()),
            duration_minutes=args.duration,
            check_interval=args.interval
        )
        
        if not df.empty:
            data_manager = DataManager(data_file=args.output)
            
            result_df = df[df['region'] == 'result'] if 'result' in df['region'].values else df
            result_df = result_df[['timestamp', 'value']].copy()
            
            data_manager.save_data(result_df, append=True)
            
            print(f"\n✅ Saved {len(result_df)} data points to {args.output}")
            print(f"\n📊 Summary:")
            print(f"   Min: {result_df['value'].min()}")
            print(f"   Max: {result_df['value'].max()}")
            print(f"   Mean: {result_df['value'].mean():.2f}")
            
            print(f"\n📈 Latest values: {result_df['value'].tail(10).tolist()}")
            
            print("\n🎯 Make predictions:")
            print(f"   python3 main.py --data-file {args.output} --steps 5 --visualize")
        else:
            print("\n⚠️  No data collected")
    
    elif args.mode == 'preview':
        print("\n🔴 PREVIEW MODE")
        
        if not args.region:
            if analyzer.regions:
                args.region = list(analyzer.regions.keys())[0]
                print(f"Using region: {args.region}")
            else:
                print("⚠️  No regions defined. Run setup first.")
                return
        
        live_preview_mode(analyzer, args.region)
    
    elif args.mode == 'obs':
        print("\n📹 OBS VIDEO PROCESSING MODE")
        
        if not args.video:
            print("⚠️  Please specify video file with --video")
            return
        
        if not analyzer.regions:
            print("⚠️  No regions defined. Run setup first.")
            return
        
        obs = OBSIntegration()
        obs.analyzer = analyzer
        
        df = obs.process_obs_recording(args.video, sample_rate=30)
        
        if not df.empty:
            data_manager = DataManager(data_file=args.output)
            result_df = df[df['region'] == 'result'] if 'result' in df['region'].values else df
            result_df = result_df[['timestamp', 'value']].copy()
            
            data_manager.save_data(result_df, append=True)
            print(f"\n✅ Saved {len(result_df)} data points to {args.output}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
