#!/usr/bin/env python3
"""
Zeppelin Auto-Capture Script

Monitors the Zeppelin game page via Selenium and automatically captures
crash values, adding them to the dashboard data file.

Usage:
    python auto_capture.py

Requirements:
    pip install selenium webdriver-manager
"""

import time
import os
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.error("Selenium not installed. Run: pip install selenium webdriver-manager")


DATA_FILE = 'data/crash_data.csv'


def load_existing_data():
    """Load existing crash data."""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            return df['value'].tolist() if 'value' in df.columns else []
        except:
            return []
    return []


def save_crash(value: float):
    """Save a new crash value."""
    os.makedirs('data', exist_ok=True)
    
    crashes = load_existing_data()
    crashes.append(value)
    
    df = pd.DataFrame({
        'timestamp': [datetime.now().isoformat() for _ in crashes],
        'value': crashes
    })
    df.to_csv(DATA_FILE, index=False)
    logger.info(f"💾 Saved crash: {value:.2f}x (Total: {len(crashes)})")


class ZeppelinMonitor:
    """Monitors Zeppelin game for crash values."""
    
    def __init__(self, url: str = "https://zeppelin-tz.betsolutions.com"):
        self.url = url
        self.driver = None
        self.last_crashes = set()
        self.captured_count = 0
        
    def start(self):
        """Start the browser."""
        logger.info("🚀 Starting browser...")
        
        options = Options()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('--headless')  # Uncomment for headless mode
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        except:
            # Fallback
            self.driver = webdriver.Chrome(options=options)
        
        self.driver.set_window_size(800, 600)
        logger.info("✅ Browser started")
    
    def navigate_to_game(self):
        """Navigate to the game page."""
        logger.info(f"🌐 Navigating to {self.url}")
        self.driver.get(self.url)
        time.sleep(5)  # Wait for page load
        logger.info("✅ Page loaded")
    
    def get_crash_history(self) -> list:
        """Extract crash history from the page."""
        crashes = []
        
        try:
            # Try to find crash history elements
            # These selectors may need to be updated based on actual page structure
            elements = self.driver.find_elements(By.CSS_SELECTOR, '[class*="history"] span, [class*="crash"] span')
            
            for el in elements:
                text = el.text.strip()
                if 'x' in text.lower():
                    try:
                        value = float(text.lower().replace('x', '').strip())
                        if 1.0 <= value <= 1000:  # Reasonable range
                            crashes.append(value)
                    except:
                        pass
            
            # Also try getting from visible multiplier elements
            multipliers = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'x') or contains(text(), 'X')]")
            
            for el in multipliers:
                text = el.text.strip()
                if text and 'x' in text.lower():
                    try:
                        value = float(text.lower().replace('x', '').strip())
                        if 1.0 <= value <= 1000:
                            crashes.append(value)
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Error extracting crashes: {e}")
        
        return list(set(crashes))  # Remove duplicates
    
    def get_current_multiplier(self) -> float:
        """Get the current/last multiplier displayed."""
        try:
            # Look for large multiplier display
            large_text = self.driver.find_elements(By.CSS_SELECTOR, '[class*="multiplier"], [class*="value"]')
            
            for el in large_text:
                text = el.text.strip()
                if 'x' in text.lower():
                    try:
                        value = float(text.lower().replace('x', '').strip())
                        if 1.0 <= value <= 1000:
                            return value
                    except:
                        pass
        except:
            pass
        
        return None
    
    def monitor(self, duration_minutes: int = 30, check_interval: int = 3):
        """
        Monitor the game for new crashes.
        
        Args:
            duration_minutes: How long to monitor
            check_interval: Seconds between checks
        """
        logger.info(f"👀 Starting monitor for {duration_minutes} minutes...")
        logger.info("Press Ctrl+C to stop\n")
        
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                # Get current crashes from page
                history = self.get_crash_history()
                current = self.get_current_multiplier()
                
                # Check for new crashes
                new_crashes = set(history) - self.last_crashes
                
                if new_crashes:
                    for crash in new_crashes:
                        save_crash(crash)
                        self.captured_count += 1
                        logger.info(f"🎯 NEW CRASH: {crash:.2f}x")
                    
                    self.last_crashes.update(new_crashes)
                
                if current:
                    logger.debug(f"Current display: {current:.2f}x")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Monitoring stopped by user")
        
        logger.info(f"\n📊 Session complete: Captured {self.captured_count} crashes")
    
    def stop(self):
        """Stop the browser."""
        if self.driver:
            self.driver.quit()
            logger.info("🛑 Browser closed")


def manual_capture_mode():
    """Fallback mode for manual capture without Selenium."""
    print("=" * 60)
    print("📝 MANUAL CAPTURE MODE")
    print("=" * 60)
    print("\nSelenium not available. Use manual mode instead.")
    print("Enter crash values as they occur (e.g., '2.35')")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            value = input("Enter crash value: ").strip()
            
            if value.lower() == 'quit':
                break
            
            # Parse value
            value = float(value.replace('x', '').replace('X', ''))
            
            if 1.0 <= value <= 1000:
                save_crash(value)
                print(f"✅ Added {value:.2f}x")
            else:
                print("❌ Value must be between 1.0 and 1000")
                
        except ValueError:
            print("❌ Invalid number. Try again.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
    
    crashes = load_existing_data()
    print(f"\n📊 Total crashes saved: {len(crashes)}")


def run_with_existing_browser():
    """
    Alternative: Read from existing browser using debugging port.
    
    Start Chrome with: 
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
    """
    if not SELENIUM_AVAILABLE:
        manual_capture_mode()
        return
    
    logger.info("🔗 Connecting to existing Chrome browser...")
    
    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    
    try:
        driver = webdriver.Chrome(options=options)
        logger.info("✅ Connected to existing browser")
        
        # Monitor mode
        monitor = ZeppelinMonitor()
        monitor.driver = driver
        monitor.monitor(duration_minutes=60)
        
    except Exception as e:
        logger.error(f"Could not connect to browser: {e}")
        logger.info("Falling back to manual mode...")
        manual_capture_mode()


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ZEPPELIN AUTO-CAPTURE")
    print("=" * 60)
    
    print("\nOptions:")
    print("1. Manual capture (type values)")
    print("2. Connect to existing browser (requires Chrome debugging port)")
    print("3. Open new browser (requires Selenium setup)")
    
    choice = input("\nChoice [1/2/3]: ").strip()
    
    if choice == "1":
        manual_capture_mode()
    elif choice == "2":
        run_with_existing_browser()
    elif choice == "3":
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not installed. Run: pip install selenium webdriver-manager")
        else:
            monitor = ZeppelinMonitor()
            try:
                monitor.start()
                monitor.navigate_to_game()
                print("\n⚠️  Please log in to the game if needed, then press Enter...")
                input()
                monitor.monitor(duration_minutes=60)
            finally:
                monitor.stop()
    else:
        print("Invalid choice. Using manual mode...")
        manual_capture_mode()
