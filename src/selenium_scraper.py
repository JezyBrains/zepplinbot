from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


class SeleniumScraper:
    def __init__(self, headless: bool = False, wait_time: int = 10):
        self.headless = headless
        self.wait_time = wait_time
        self.driver = None
        
    def initialize_driver(self):
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logger.info("Selenium WebDriver initialized")
        
    def open_url(self, url: str):
        if not self.driver:
            self.initialize_driver()
        
        self.driver.get(url)
        logger.info(f"Opened URL: {url}")
        time.sleep(3)
        
    def wait_for_element(self, selector: str, by: By = By.CSS_SELECTOR, timeout: int = None):
        timeout = timeout or self.wait_time
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except Exception as e:
            logger.error(f"Element not found: {selector} - {e}")
            return None
    
    def extract_numbers_from_page(self, selectors: List[str] = None) -> List[float]:
        if not selectors:
            selectors = [
                '.number', '.result', '.outcome', '.value', 
                '[class*="number"]', '[class*="result"]',
                'span', 'div', 'p'
            ]
        
        numbers = []
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text:
                        found_numbers = re.findall(r'\d+\.?\d*', text)
                        numbers.extend([float(n) for n in found_numbers if n])
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        return numbers
    
    def manual_data_collection(self, url: str, wait_seconds: int = 30) -> List[float]:
        self.open_url(url)
        
        print("\n" + "="*60)
        print("MANUAL DATA COLLECTION MODE")
        print("="*60)
        print(f"\nBrowser opened. You have {wait_seconds} seconds to:")
        print("1. Complete any CAPTCHA or verification")
        print("2. Navigate to the data you want to collect")
        print("3. Let the page fully load")
        print("\nWaiting...")
        
        time.sleep(wait_seconds)
        
        print("\nAttempting to extract numbers from the page...")
        numbers = self.extract_numbers_from_page()
        
        if numbers:
            print(f"\nFound {len(numbers)} numbers: {numbers[:10]}...")
        else:
            print("\nNo numbers found automatically.")
        
        return numbers
    
    def interactive_scraping_session(self, url: str) -> pd.DataFrame:
        self.open_url(url)
        
        print("\n" + "="*60)
        print("INTERACTIVE SCRAPING SESSION")
        print("="*60)
        print("\nBrowser is open. Instructions:")
        print("1. Interact with the page as needed (login, navigate, etc.)")
        print("2. When you see the data you want to collect, press Enter")
        print("3. The system will attempt to extract numbers")
        print("4. You can repeat this process multiple times")
        print("5. Type 'done' when finished")
        
        all_data = []
        
        while True:
            user_input = input("\nPress Enter to capture data (or 'done' to finish): ").strip().lower()
            
            if user_input == 'done':
                break
            
            numbers = self.extract_numbers_from_page()
            
            if numbers:
                print(f"Captured {len(numbers)} numbers: {numbers}")
                
                for num in numbers:
                    all_data.append({
                        'timestamp': datetime.now(),
                        'value': num
                    })
            else:
                print("No numbers found. Try a different selector or manual entry.")
                manual_input = input("Enter number manually (or press Enter to skip): ").strip()
                if manual_input:
                    try:
                        num = float(manual_input)
                        all_data.append({
                            'timestamp': datetime.now(),
                            'value': num
                        })
                        print(f"Added: {num}")
                    except ValueError:
                        print("Invalid number format")
        
        self.close()
        
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"\nTotal data points collected: {len(df)}")
            return df
        
        return pd.DataFrame()
    
    def scrape_with_custom_selector(self, url: str, selector: str, 
                                    wait_for_load: int = 5) -> List[float]:
        self.open_url(url)
        time.sleep(wait_for_load)
        
        numbers = []
        
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            logger.info(f"Found {len(elements)} elements with selector: {selector}")
            
            for element in elements:
                text = element.text.strip()
                if text:
                    found_numbers = re.findall(r'\d+\.?\d*', text)
                    numbers.extend([float(n) for n in found_numbers if n])
        except Exception as e:
            logger.error(f"Error with selector {selector}: {e}")
        
        return numbers
    
    def take_screenshot(self, filepath: str = 'screenshot.png'):
        if self.driver:
            self.driver.save_screenshot(filepath)
            logger.info(f"Screenshot saved to {filepath}")
    
    def get_page_source(self) -> str:
        if self.driver:
            return self.driver.page_source
        return ""
    
    def execute_javascript(self, script: str):
        if self.driver:
            return self.driver.execute_script(script)
        return None
    
    def extract_from_javascript(self, js_variable: str) -> any:
        script = f"return {js_variable};"
        try:
            result = self.execute_javascript(script)
            return result
        except Exception as e:
            logger.error(f"Error extracting JS variable {js_variable}: {e}")
            return None
    
    def close(self):
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")


class GameDataCollector:
    def __init__(self, url: str):
        self.url = url
        self.scraper = SeleniumScraper(headless=False)
        self.data_history = []
        
    def collect_game_results(self, duration_minutes: int = 5, 
                            check_interval: int = 10) -> pd.DataFrame:
        self.scraper.open_url(self.url)
        
        print(f"\nCollecting data for {duration_minutes} minutes...")
        print(f"Checking every {check_interval} seconds")
        print("Please ensure the game is visible in the browser window.")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            numbers = self.scraper.extract_numbers_from_page()
            
            if numbers:
                latest_number = numbers[0]
                
                if not self.data_history or latest_number != self.data_history[-1]['value']:
                    data_point = {
                        'timestamp': datetime.now(),
                        'value': latest_number
                    }
                    self.data_history.append(data_point)
                    print(f"New value: {latest_number} at {data_point['timestamp']}")
            
            time.sleep(check_interval)
        
        self.scraper.close()
        
        if self.data_history:
            return pd.DataFrame(self.data_history)
        return pd.DataFrame()
    
    def manual_entry_mode(self) -> pd.DataFrame:
        print("\n" + "="*60)
        print("MANUAL DATA ENTRY MODE")
        print("="*60)
        print("\nEnter game results one by one.")
        print("Type 'done' when finished, 'undo' to remove last entry")
        
        data = []
        
        while True:
            user_input = input("\nEnter result (or 'done'/'undo'): ").strip().lower()
            
            if user_input == 'done':
                break
            elif user_input == 'undo':
                if data:
                    removed = data.pop()
                    print(f"Removed: {removed['value']}")
                else:
                    print("No data to remove")
                continue
            
            try:
                value = float(user_input)
                data.append({
                    'timestamp': datetime.now(),
                    'value': value
                })
                print(f"Added: {value} (Total: {len(data)} entries)")
            except ValueError:
                print("Invalid number. Please enter a numeric value.")
        
        if data:
            df = pd.DataFrame(data)
            print(f"\nTotal entries: {len(df)}")
            return df
        
        return pd.DataFrame()
