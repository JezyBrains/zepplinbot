import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

from models.lstm_model import LSTMPredictor
from models.prophet_model import ProphetPredictor
from models.statistical_models import ARIMAPredictor, MovingAveragePredictor
from models.ml_models import XGBoostPredictor, LightGBMPredictor
from ensemble_predictor import EnsemblePredictor
from data_scraper import WebDataScraper, DataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictionEngine:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.ensemble = None
        self.data_manager = DataManager()
        self.scraper = None
        
    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def initialize_scraper(self):
        website_config = self.config.get('website', {})
        scraping_config = self.config.get('scraping', {})
        
        self.scraper = WebDataScraper(
            url=website_config.get('url'),
            selector=website_config.get('selector'),
            headers=scraping_config.get('headers')
        )
        logger.info("Web scraper initialized")
    
    def scrape_and_save_data(self):
        if not self.scraper:
            self.initialize_scraper()
        
        df = self.scraper.scrape_to_dataframe()
        if not df.empty:
            self.data_manager.save_data(df)
            logger.info(f"Scraped and saved {len(df)} data points")
            return df
        return None
    
    def load_data(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        if data is not None:
            return data
        
        values = self.data_manager.get_latest_values(n=1000)
        if values:
            return np.array(values)
        
        raise ValueError("No data available. Please provide data or scrape from website.")
    
    def initialize_models(self):
        pred_config = self.config.get('prediction', {})
        sequence_length = pred_config.get('sequence_length', 50)
        
        model_names = pred_config.get('models', ['lstm', 'prophet', 'arima', 'xgboost'])
        
        if 'lstm' in model_names:
            lstm_config = pred_config.get('lstm', {})
            self.models['lstm'] = LSTMPredictor(
                sequence_length=sequence_length,
                units=lstm_config.get('units', [128, 64, 32]),
                dropout=lstm_config.get('dropout', 0.2),
                learning_rate=lstm_config.get('learning_rate', 0.001)
            )
            logger.info("LSTM model initialized")
        
        if 'prophet' in model_names:
            prophet_config = pred_config.get('prophet', {})
            self.models['prophet'] = ProphetPredictor(
                changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
                seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative')
            )
            logger.info("Prophet model initialized")
        
        if 'arima' in model_names:
            arima_config = pred_config.get('arima', {})
            self.models['arima'] = ARIMAPredictor(
                auto_order=arima_config.get('auto_order', True)
            )
            logger.info("ARIMA model initialized")
        
        if 'xgboost' in model_names:
            xgb_config = pred_config.get('xgboost', {})
            self.models['xgboost'] = XGBoostPredictor(
                n_estimators=xgb_config.get('n_estimators', 1000),
                max_depth=xgb_config.get('max_depth', 7),
                learning_rate=xgb_config.get('learning_rate', 0.01),
                sequence_length=sequence_length
            )
            logger.info("XGBoost model initialized")
        
        if 'lightgbm' in model_names:
            self.models['lightgbm'] = LightGBMPredictor(
                sequence_length=sequence_length
            )
            logger.info("LightGBM model initialized")
        
        if 'moving_average' in model_names:
            self.models['moving_average'] = MovingAveragePredictor()
            logger.info("Moving Average model initialized")
    
    def train_models(self, data: np.ndarray):
        pred_config = self.config.get('prediction', {})
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                if model_name == 'lstm':
                    lstm_config = pred_config.get('lstm', {})
                    model.train(
                        data,
                        epochs=lstm_config.get('epochs', 100),
                        batch_size=lstm_config.get('batch_size', 32),
                        validation_split=pred_config.get('validation_split', 0.2)
                    )
                elif model_name in ['xgboost', 'lightgbm', 'arima', 'prophet']:
                    model.train(data)
                
                logger.info(f"{model_name} training completed")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
    
    def initialize_ensemble(self, data: np.ndarray):
        ensemble_config = self.config.get('prediction', {}).get('ensemble', {})
        
        self.ensemble = EnsemblePredictor(
            models=self.models,
            method=ensemble_config.get('method', 'weighted_average')
        )
        
        if ensemble_config.get('weights') == 'auto':
            self.ensemble.calculate_auto_weights(data)
        
        logger.info("Ensemble predictor initialized")
    
    def predict(self, data: Optional[np.ndarray] = None, steps: int = 1) -> Dict:
        data = self.load_data(data)
        
        if not self.models:
            self.initialize_models()
            self.train_models(data)
        
        if not self.ensemble:
            self.initialize_ensemble(data)
        
        result = self.ensemble.predict_next(data, steps=steps, use_auto_weights=True)
        confidence_interval = self.ensemble.get_confidence_interval(data, steps=steps)
        
        return {
            'prediction': result['ensemble'],
            'individual_predictions': result['individual'],
            'weights': result['weights'],
            'confidence_interval': confidence_interval,
            'statistics': self.ensemble.get_prediction_statistics(result['individual'])
        }
    
    def save_models(self, directory: str = 'models/saved'):
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                filepath = os.path.join(directory, model_name)
                if hasattr(model, 'save_model'):
                    model.save_model(filepath)
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
    
    def load_models(self, directory: str = 'models/saved'):
        for model_name, model in self.models.items():
            try:
                filepath = os.path.join(directory, model_name)
                if hasattr(model, 'load_model'):
                    model.load_model(filepath)
                    logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
    
    def evaluate_models(self, data: np.ndarray, test_size: float = 0.2):
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                predictions = []
                
                if hasattr(model, 'predict_batch'):
                    predictions = model.predict_batch(train_data, test_data)
                else:
                    for i in range(len(test_data)):
                        recent_data = np.concatenate([train_data, test_data[:i]]) if i > 0 else train_data
                        pred = model.predict_next(recent_data, steps=1)[0]
                        predictions.append(pred)

                    predictions = np.array(predictions)
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                results[model_name] = {
                    'mse': mean_squared_error(test_data, predictions),
                    'mae': mean_absolute_error(test_data, predictions),
                    'rmse': np.sqrt(mean_squared_error(test_data, predictions)),
                    'r2': r2_score(test_data, predictions)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return results
