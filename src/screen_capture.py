import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab
import mss
import time
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class ScreenCaptureAnalyzer:
    def __init__(self, config_file: str = 'screen_mapping.json'):
        self.config_file = config_file
        self.regions = {}
        self.last_values = {}
        self.load_config()
        
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.regions = json.load(f)
            logger.info(f"Loaded screen mapping from {self.config_file}")
        except FileNotFoundError:
            logger.info("No config file found, will create new mapping")
            self.regions = {}
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.regions, f, indent=2)
        logger.info(f"Saved screen mapping to {self.config_file}")
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        with mss.mss() as sct:
            if region:
                x, y, width, height = region
                monitor = {"top": y, "left": x, "width": width, "height": height}
            else:
                monitor = sct.monitors[1]
            
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    
    def extract_text_from_image(self, img: np.ndarray, preprocess: bool = True) -> str:
        if preprocess:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            gray = cv2.medianBlur(gray, 3)
        else:
            gray = img
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.x'
        text = pytesseract.image_to_string(gray, config=custom_config)
        
        return text.strip()
    
    def extract_numbers_from_text(self, text: str) -> List[float]:
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) for n in numbers if n]
    
    def define_region_interactive(self, region_name: str):
        print(f"\n📍 Defining region: {region_name}")
        print("Instructions:")
        print("1. A screenshot will be taken")
        print("2. Click and drag to select the region containing the number")
        print("3. Press 'c' to confirm, 'r' to retry, 'q' to quit")
        
        img = self.capture_screen()
        
        clone = img.copy()
        roi = cv2.selectROI("Select Region", clone, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            self.regions[region_name] = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
            
            test_img = img[y:y+h, x:x+w]
            text = self.extract_text_from_image(test_img)
            numbers = self.extract_numbers_from_text(text)
            
            print(f"✅ Region saved: {region_name}")
            print(f"   Coordinates: x={x}, y={y}, w={w}, h={h}")
            print(f"   Test extraction: '{text}'")
            print(f"   Numbers found: {numbers}")
            
            self.save_config()
            return True
        
        return False
    
    def capture_region(self, region_name: str) -> Optional[float]:
        if region_name not in self.regions:
            logger.error(f"Region '{region_name}' not defined")
            return None
        
        region = self.regions[region_name]
        img = self.capture_screen()
        
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        roi = img[y:y+h, x:x+w]
        
        text = self.extract_text_from_image(roi)
        numbers = self.extract_numbers_from_text(text)
        
        if numbers:
            return numbers[0]
        
        return None
    
    def monitor_regions(self, region_names: List[str], duration_minutes: int = 10, 
                       check_interval: int = 5) -> pd.DataFrame:
        print(f"\n👁️  Monitoring regions for {duration_minutes} minutes")
        print(f"   Checking every {check_interval} seconds")
        print(f"   Regions: {', '.join(region_names)}")
        
        data = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            timestamp = datetime.now()
            
            for region_name in region_names:
                value = self.capture_region(region_name)
                
                if value is not None:
                    if region_name not in self.last_values or self.last_values[region_name] != value:
                        data.append({
                            'timestamp': timestamp,
                            'region': region_name,
                            'value': value
                        })
                        self.last_values[region_name] = value
                        print(f"📊 {timestamp.strftime('%H:%M:%S')} - {region_name}: {value}")
            
            time.sleep(check_interval)
        
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    
    def setup_wizard(self):
        print("\n" + "="*60)
        print("SCREEN CAPTURE SETUP WIZARD")
        print("="*60)
        print("\nThis wizard will help you map screen regions to game elements.")
        print("\nTips:")
        print("- Make sure the game is visible on screen")
        print("- Define regions for each number you want to track")
        print("- Common regions: 'result', 'multiplier', 'next_number'")
        
        regions_to_define = []
        
        while True:
            region_name = input("\nEnter region name (or 'done' to finish): ").strip()
            if region_name.lower() == 'done':
                break
            if region_name:
                regions_to_define.append(region_name)
        
        if not regions_to_define:
            print("⚠️  No regions defined")
            return
        
        print(f"\n📍 Will define {len(regions_to_define)} regions")
        input("Press Enter when the game is visible on screen...")
        
        for region_name in regions_to_define:
            success = self.define_region_interactive(region_name)
            if not success:
                print(f"⚠️  Skipped {region_name}")
        
        print("\n✅ Setup complete!")
        print(f"   Regions defined: {list(self.regions.keys())}")
        print(f"   Config saved to: {self.config_file}")


class OBSIntegration:
    def __init__(self, obs_output_folder: str = None):
        self.obs_output_folder = obs_output_folder
        self.analyzer = ScreenCaptureAnalyzer()
    
    def process_obs_recording(self, video_path: str, sample_rate: int = 30) -> pd.DataFrame:
        print(f"📹 Processing OBS recording: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   FPS: {fps}, Total frames: {total_frames}")
        
        frame_interval = int(fps * sample_rate)
        data = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                for region_name, region in self.analyzer.regions.items():
                    x, y, w, h = region['x'], region['y'], region['width'], region['height']
                    roi = frame[y:y+h, x:x+w]
                    
                    text = self.analyzer.extract_text_from_image(roi)
                    numbers = self.analyzer.extract_numbers_from_text(text)
                    
                    if numbers:
                        timestamp = datetime.now()
                        data.append({
                            'timestamp': timestamp,
                            'frame': frame_count,
                            'region': region_name,
                            'value': numbers[0]
                        })
            
            frame_count += 1
        
        cap.release()
        
        if data:
            df = pd.DataFrame(data)
            print(f"✅ Extracted {len(df)} data points from video")
            return df
        
        return pd.DataFrame()
    
    def watch_obs_folder(self, check_interval: int = 10):
        import os
        from pathlib import Path
        
        if not self.obs_output_folder:
            print("⚠️  OBS output folder not specified")
            return
        
        print(f"👁️  Watching OBS folder: {self.obs_output_folder}")
        processed_files = set()
        
        while True:
            files = list(Path(self.obs_output_folder).glob("*.mp4"))
            
            for file in files:
                if file not in processed_files:
                    print(f"\n📹 New recording found: {file.name}")
                    df = self.process_obs_recording(str(file))
                    
                    if not df.empty:
                        output_file = f"data/obs_{file.stem}.csv"
                        df.to_csv(output_file, index=False)
                        print(f"✅ Saved to {output_file}")
                    
                    processed_files.add(file)
            
            time.sleep(check_interval)


def live_preview_mode(analyzer: ScreenCaptureAnalyzer, region_name: str):
    print(f"\n🔴 LIVE PREVIEW MODE - {region_name}")
    print("Press 'q' to quit")
    
    if region_name not in analyzer.regions:
        print(f"⚠️  Region '{region_name}' not defined")
        return
    
    region = analyzer.regions[region_name]
    
    while True:
        img = analyzer.capture_screen()
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        roi = img[y:y+h, x:x+w]
        
        roi_large = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        
        text = analyzer.extract_text_from_image(roi)
        numbers = analyzer.extract_numbers_from_text(text)
        
        cv2.putText(roi_large, f"Value: {numbers[0] if numbers else 'N/A'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f"Live Preview - {region_name}", roi_large)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
