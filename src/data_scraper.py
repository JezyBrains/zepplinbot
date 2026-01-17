import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebDataScraper:
    def __init__(self, url: str, selector: str, headers: Optional[Dict] = None):
        self.url = url
        self.selector = selector
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        
    def extract_numbers(self, text: str) -> List[float]:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers if n]
    
    def scrape_data(self, retry_attempts: int = 3, timeout: int = 30) -> List[Dict]:
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Attempting to scrape data from {self.url} (attempt {attempt + 1})")
                response = self.session.get(self.url, headers=self.headers, timeout=timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                elements = soup.select(self.selector)
                
                data = []
                for element in elements:
                    text = element.get_text(strip=True)
                    numbers = self.extract_numbers(text)
                    
                    if numbers:
                        data.append({
                            'timestamp': datetime.now(),
                            'value': numbers[0],
                            'raw_text': text,
                            'all_numbers': numbers
                        })
                
                logger.info(f"Successfully scraped {len(data)} data points")
                return data
                
            except requests.RequestException as e:
                logger.error(f"Error scraping data (attempt {attempt + 1}): {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        return []
    
    def scrape_to_dataframe(self, **kwargs) -> pd.DataFrame:
        data = self.scrape_data(**kwargs)
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    
    def scrape_historical(self, url_pattern: str, date_range: List[str]) -> pd.DataFrame:
        all_data = []
        
        for date in date_range:
            url = url_pattern.format(date=date)
            temp_scraper = WebDataScraper(url, self.selector, self.headers)
            data = temp_scraper.scrape_data()
            all_data.extend(data)
            time.sleep(1)
        
        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()


class APIDataExtractor:
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def fetch_data(self, params: Optional[Dict] = None) -> pd.DataFrame:
        try:
            response = requests.get(self.api_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    return pd.DataFrame(data['data'])
                return pd.DataFrame([data])
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return pd.DataFrame()


class DataManager:
    def __init__(self, data_file: str = 'data/historical_data.csv'):
        self.data_file = data_file
        
    def save_data(self, df: pd.DataFrame, append: bool = True):
        import os
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if append and os.path.exists(self.data_file):
            existing_df = pd.read_csv(self.data_file)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        df.to_csv(self.data_file, index=False)
        logger.info(f"Saved {len(df)} records to {self.data_file}")
    
    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            logger.warning(f"Data file {self.data_file} not found")
            return pd.DataFrame()
    
    def get_latest_values(self, n: int = 100) -> List[float]:
        df = self.load_data()
        if not df.empty and 'value' in df.columns:
            return df['value'].tail(n).tolist()
        return []
