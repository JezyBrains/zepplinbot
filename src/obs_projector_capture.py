import cv2
import numpy as np
import pytesseract
import pyautogui
from PIL import Image, ImageGrab
import time
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OBSProjectorCapture:
    """
    Captures data from OBS Windowed Projector using OCR.
    Treats the website as a video feed to bypass anti-bot measures.
    """
    
    def __init__(self, config_file: str = 'obs_roi_config.json'):
        self.config_file = config_file
        self.roi_regions = {}
        self.last_values = {}
        self.data_history = []
        self.load_config()
        
    def load_config(self):
        """Load ROI configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                self.roi_regions = json.load(f)
            logger.info(f"Loaded ROI config from {self.config_file}")
            logger.info(f"Regions: {list(self.roi_regions.keys())}")
        except FileNotFoundError:
            logger.info("No config file found. Run setup to create ROI regions.")
            self.roi_regions = {}
    
    def save_config(self):
        """Save ROI configuration to JSON file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.roi_regions, f, indent=2)
        logger.info(f"Saved ROI config to {self.config_file}")
    
    def find_obs_projector_window(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate OBS Windowed Projector window on screen.
        Returns (x, y, width, height) or None if not found.
        
        Note: Auto-detection not available on macOS, will use manual selection.
        """
        try:
            import pygetwindow as gw
            
            # Check if we're on macOS (pygetwindow has limited functionality)
            if hasattr(gw, 'getWindowsWithTitle'):
                windows = gw.getWindowsWithTitle('Windowed Projector')
                
                if windows:
                    win = windows[0]
                    return (win.left, win.top, win.width, win.height)
            else:
                logger.info("Window auto-detection not available on macOS. Using manual selection.")
                return None
            
            logger.info("OBS Windowed Projector not found. Will use manual selection.")
            return None
            
        except (ImportError, AttributeError) as e:
            logger.info("Window detection not available. Using manual selection.")
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Capture a specific region of the screen.
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
            
        Returns:
            Captured image as numpy array
        """
        # Get screen size to validate coordinates
        screen_width, screen_height = pyautogui.size()
        
        # Clamp coordinates to screen bounds
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        width = min(width, screen_width - x)
        height = min(height, screen_height - y)
        
        # Ensure minimum size
        if width < 1 or height < 1:
            raise ValueError(f"Invalid region size: {width}x{height}")
        
        try:
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            logger.error(f"Failed to capture region ({x}, {y}, {width}, {height}): {e}")
            raise
    
    def preprocess_for_ocr(self, img: np.ndarray, scale_factor: int = 3) -> np.ndarray:
        """
        Apply image preprocessing to maximize OCR accuracy.
        
        Steps:
        1. Convert to grayscale
        2. Scale up for better recognition
        3. Apply adaptive thresholding
        4. Denoise
        
        Args:
            img: Input image
            scale_factor: Upscaling factor for better OCR
            
        Returns:
            Preprocessed image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                         interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def extract_numbers_from_image(self, img: np.ndarray, 
                                   preprocess: bool = True) -> List[float]:
        """
        Extract numeric values from image using Tesseract OCR.
        
        Args:
            img: Input image
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of extracted numbers
        """
        if preprocess:
            processed = self.preprocess_for_ocr(img)
        else:
            processed = img
        
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.x'
        
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        text = text.strip()
        
        numbers = re.findall(r'\d+\.?\d*', text)
        result = [float(n) for n in numbers if n and float(n) > 0]
        
        return result
    
    def setup_roi_interactive(self, region_name: str, 
                             projector_bounds: Optional[Tuple] = None):
        """
        Interactive ROI setup - user selects region on screen.
        
        Args:
            region_name: Name for this ROI (e.g., 'result', 'multiplier')
            projector_bounds: Optional OBS projector window bounds
        """
        print(f"\n📍 Setting up ROI: {region_name}")
        print("Instructions:")
        print("1. Make sure OBS Windowed Projector is visible")
        print("2. Click and drag to select the region containing the number")
        print("3. Press SPACE or ENTER to confirm")
        print("4. Press ESC to cancel")
        
        input("\nPress Enter when ready to capture...")
        
        if projector_bounds:
            x, y, w, h = projector_bounds
            screenshot = self.capture_region(x, y, w, h)
        else:
            screenshot = pyautogui.screenshot()
            screenshot = np.array(screenshot)
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        
        roi = cv2.selectROI("Select ROI - Press SPACE to confirm", 
                           screenshot, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            
            if projector_bounds:
                x += projector_bounds[0]
                y += projector_bounds[1]
            
            # Validate coordinates are within screen bounds
            screen_width, screen_height = pyautogui.size()
            
            if x < 0 or y < 0 or x + w > screen_width or y + h > screen_height:
                print(f"\n⚠️  Warning: Selected region is outside screen bounds")
                print(f"   Screen size: {screen_width}x{screen_height}")
                print(f"   Selected: x={x}, y={y}, w={w}, h={h}")
                print(f"   Adjusting to fit screen...")
                
                x = max(0, min(x, screen_width - 10))
                y = max(0, min(y, screen_height - 10))
                w = min(w, screen_width - x)
                h = min(h, screen_height - y)
            
            self.roi_regions[region_name] = {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
            
            try:
                test_img = self.capture_region(x, y, w, h)
                numbers = self.extract_numbers_from_image(test_img)
                
                print(f"\n✅ ROI saved: {region_name}")
                print(f"   Coordinates: x={x}, y={y}, w={w}, h={h}")
                print(f"   Test extraction: {numbers}")
                
                self.save_config()
                return True
            except Exception as e:
                print(f"\n❌ Error testing ROI: {e}")
                print(f"   Region may be invalid. Try selecting again.")
                return False
        
        print("❌ ROI setup cancelled")
        return False
    
    def capture_roi_value(self, region_name: str) -> Optional[float]:
        """
        Capture and extract value from a defined ROI.
        
        Args:
            region_name: Name of the ROI to capture
            
        Returns:
            Extracted numeric value or None
        """
        if region_name not in self.roi_regions:
            logger.error(f"ROI '{region_name}' not defined")
            return None
        
        roi = self.roi_regions[region_name]
        img = self.capture_region(roi['x'], roi['y'], roi['width'], roi['height'])
        
        numbers = self.extract_numbers_from_image(img)
        
        if numbers:
            return numbers[0]
        
        return None
    
    def monitor_continuous(self, region_name: str = 'result', 
                          duration_minutes: int = 30,
                          capture_interval: float = 2.0,
                          output_file: str = 'data/obs_capture_data.csv') -> pd.DataFrame:
        """
        Continuously monitor OBS projector and extract data.
        
        Args:
            region_name: ROI to monitor
            duration_minutes: How long to monitor
            capture_interval: Seconds between captures
            output_file: Where to save data
            
        Returns:
            DataFrame with collected data
        """
        if region_name not in self.roi_regions:
            logger.error(f"ROI '{region_name}' not defined. Run setup first.")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("🎥 OBS PROJECTOR MONITORING")
        print("="*60)
        print(f"Region: {region_name}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Interval: {capture_interval} seconds")
        print(f"Output: {output_file}")
        print("\nMake sure OBS Windowed Projector is visible and stable!")
        print("="*60)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        capture_count = 0
        success_count = 0
        
        while time.time() < end_time:
            try:
                value = self.capture_roi_value(region_name)
                capture_count += 1
                
                if value is not None:
                    if region_name not in self.last_values or self.last_values[region_name] != value:
                        timestamp = datetime.now()
                        
                        data_point = {
                            'timestamp': timestamp,
                            'value': value,
                            'region': region_name
                        }
                        
                        self.data_history.append(data_point)
                        self.last_values[region_name] = value
                        success_count += 1
                        
                        elapsed = int(time.time() - start_time)
                        remaining = int(end_time - time.time())
                        
                        print(f"[{elapsed:04d}s] 📊 New value: {value} | "
                              f"Total: {success_count} | Remaining: {remaining}s")
                
                time.sleep(capture_interval)
                
            except KeyboardInterrupt:
                print("\n⚠️  Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(capture_interval)
        
        print("\n" + "="*60)
        print("📊 MONITORING COMPLETE")
        print("="*60)
        print(f"Total captures: {capture_count}")
        print(f"Successful extractions: {success_count}")
        print(f"Success rate: {(success_count/capture_count*100):.1f}%")
        
        if self.data_history:
            df = pd.DataFrame(self.data_history)
            
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ Saved {len(self.data_history)} new data points to {output_file}")
            print(f"\n📈 Statistics:")
            print(f"   Min: {df['value'].min()}")
            print(f"   Max: {df['value'].max()}")
            print(f"   Mean: {df['value'].mean():.2f}")
            print(f"   Unique values: {df['value'].nunique()}")
            
            return df
        
        return pd.DataFrame()
    
    def live_preview(self, region_name: str):
        """
        Show live preview of ROI capture and OCR results.
        Press 'q' to quit.
        
        Args:
            region_name: ROI to preview
        """
        if region_name not in self.roi_regions:
            logger.error(f"ROI '{region_name}' not defined")
            return
        
        print(f"\n🔴 LIVE PREVIEW - {region_name}")
        print("Press 'q' to quit, 's' to save current frame")
        
        roi = self.roi_regions[region_name]
        frame_count = 0
        
        while True:
            img = self.capture_region(roi['x'], roi['y'], roi['width'], roi['height'])
            
            processed = self.preprocess_for_ocr(img)
            
            numbers = self.extract_numbers_from_image(img)
            
            display = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            
            value_text = f"Value: {numbers[0]:.2f}" if numbers else "Value: N/A"
            cv2.putText(display, value_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            cv2.putText(display, f"Frame: {frame_count}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow(f"Live Preview - {region_name}", display)
            
            processed_display = cv2.resize(processed, None, fx=2, fy=2)
            cv2.imshow("Preprocessed", processed_display)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{region_name}_{frame_count}.png"
                cv2.imwrite(filename, img)
                print(f"💾 Saved frame to {filename}")
            
            frame_count += 1
        
        cv2.destroyAllWindows()
    
    def validate_setup(self) -> bool:
        """
        Validate that ROI regions are properly configured.
        
        Returns:
            True if setup is valid
        """
        if not self.roi_regions:
            print("❌ No ROI regions defined")
            return False
        
        print("\n✅ ROI Configuration:")
        for name, roi in self.roi_regions.items():
            print(f"   {name}: x={roi['x']}, y={roi['y']}, "
                  f"w={roi['width']}, h={roi['height']}")
        
        return True
