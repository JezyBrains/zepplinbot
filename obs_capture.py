#!/usr/bin/env python3
"""
OBS Windowed Projector Data Capture Tool
Treats website as video feed to bypass anti-bot measures
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from src.obs_projector_capture import OBSProjectorCapture
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description='OBS Windowed Projector Data Capture - Bypass bot detection by treating site as video'
    )
    parser.add_argument('--mode', type=str, 
                       choices=['setup', 'monitor', 'preview', 'validate'],
                       default='setup',
                       help='Operation mode')
    parser.add_argument('--region', type=str, default='result',
                       help='ROI region name (default: result)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Monitoring duration in minutes (default: 30)')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Capture interval in seconds (default: 2.0)')
    parser.add_argument('--output', type=str, default='data/obs_capture_data.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎥 OBS WINDOWED PROJECTOR DATA CAPTURE")
    print("="*70)
    print("\nBypass bot detection by treating website as video feed!")
    print()
    
    capture = OBSProjectorCapture()
    
    if args.mode == 'setup':
        print("🔧 SETUP MODE - Define ROI Regions")
        print("\nSteps:")
        print("1. Open OBS Studio")
        print("2. Add Browser Source with Zeppelin URL")
        print("3. Right-click source → Windowed Projector (Source)")
        print("4. Position projector window where you can see it")
        print("5. Return here and press Enter")
        
        input("\nPress Enter when OBS Windowed Projector is ready...")
        
        projector_bounds = capture.find_obs_projector_window()
        if projector_bounds:
            print(f"✅ Found OBS Projector at: {projector_bounds}")
        else:
            print("⚠️  Auto-detection failed. Will use full screen selection.")
        
        regions_to_setup = []
        print("\nDefine ROI regions (e.g., 'result', 'multiplier')")
        
        while True:
            region_name = input("Enter region name (or 'done'): ").strip()
            if region_name.lower() == 'done':
                break
            if region_name:
                regions_to_setup.append(region_name)
        
        if not regions_to_setup:
            regions_to_setup = ['result']
            print("Using default region: 'result'")
        
        for region_name in regions_to_setup:
            success = capture.setup_roi_interactive(region_name, projector_bounds)
            if success:
                print(f"✅ {region_name} configured successfully")
        
        capture.validate_setup()
        
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print(f"1. Test: python3 obs_capture.py --mode preview --region {regions_to_setup[0]}")
        print(f"2. Monitor: python3 obs_capture.py --mode monitor --duration 30")
        
    elif args.mode == 'monitor':
        print("👁️  MONITOR MODE - Continuous Data Capture")
        
        if not capture.validate_setup():
            print("\n❌ No ROI regions configured. Run setup first:")
            print("   python3 obs_capture.py --mode setup")
            return
        
        print(f"\nMonitoring region: {args.region}")
        print(f"Duration: {args.duration} minutes")
        print(f"Capture interval: {args.interval} seconds")
        print("\nEnsure:")
        print("✓ OBS Windowed Projector is open and visible")
        print("✓ Game is running in the projector")
        print("✓ Numbers are clearly visible")
        
        input("\nPress Enter to start monitoring...")
        
        df = capture.monitor_continuous(
            region_name=args.region,
            duration_minutes=args.duration,
            capture_interval=args.interval,
            output_file=args.output
        )
        
        if not df.empty:
            print("\n" + "="*70)
            print("🎯 READY FOR PREDICTIONS")
            print("="*70)
            print(f"\nRun: python3 main.py --data-file {args.output} --steps 5 --visualize")
        
    elif args.mode == 'preview':
        print("🔴 PREVIEW MODE - Live OCR Testing")
        
        if not capture.validate_setup():
            print("\n❌ No ROI regions configured. Run setup first.")
            return
        
        print(f"\nPreviewing region: {args.region}")
        print("Press 'q' to quit, 's' to save frame")
        
        capture.live_preview(args.region)
        
    elif args.mode == 'validate':
        print("✅ VALIDATE MODE - Check Configuration")
        
        if capture.validate_setup():
            print("\n✅ Configuration is valid")
            
            print("\nTesting capture...")
            for region_name in capture.roi_regions.keys():
                value = capture.capture_roi_value(region_name)
                print(f"   {region_name}: {value}")
        else:
            print("\n❌ Configuration invalid. Run setup first.")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
