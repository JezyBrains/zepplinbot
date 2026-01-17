import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as auto_arima
import logging
import joblib

logger = logging.getLogger(__name__)


class ARIMAPredictor:
    def __init__(self, order: tuple = None, auto_order: bool = True):
        self.order = order
        self.auto_order = auto_order
        self.model = None
        self.fitted_model = None
        
    def train(self, data: np.ndarray):
        if self.auto_order:
            logger.info("Finding optimal ARIMA parameters...")
            try:
                import pmdarima as pm
                auto_model = pm.auto_arima(
                    data,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    d=None,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
                self.order = auto_model.order
                logger.info(f"Optimal ARIMA order: {self.order}")
            except:
                self.order = (1, 1, 1)
                logger.warning("Auto ARIMA failed, using default order (1,1,1)")
        
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        logger.info(f"ARIMA{self.order} model training completed")
        
    def predict_next(self, steps: int = 1) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def get_confidence_interval(self, steps: int = 1, alpha: float = 0.05):
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return forecast.values, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
    
    def save_model(self, filepath: str):
        joblib.dump(self.fitted_model, f"{filepath}_arima.pkl")
        logger.info(f"ARIMA model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.fitted_model = joblib.load(f"{filepath}_arima.pkl")
        logger.info(f"ARIMA model loaded from {filepath}")


class SARIMAXPredictor:
    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def train(self, data: np.ndarray):
        self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        logger.info(f"SARIMAX model training completed")
        
    def predict_next(self, steps: int = 1) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def save_model(self, filepath: str):
        joblib.dump(self.fitted_model, f"{filepath}_sarimax.pkl")
    
    def load_model(self, filepath: str):
        self.fitted_model = joblib.load(f"{filepath}_sarimax.pkl")


class ExponentialSmoothingPredictor:
    def __init__(self, seasonal_periods: int = 12, trend: str = 'add', seasonal: str = 'add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        
    def train(self, data: np.ndarray):
        self.model = ExponentialSmoothing(
            data,
            seasonal_periods=self.seasonal_periods,
            trend=self.trend,
            seasonal=self.seasonal
        ).fit()
        logger.info("Exponential Smoothing model training completed")
        
    def predict_next(self, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        forecast = self.model.forecast(steps=steps)
        return np.array(forecast)
    
    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_exp_smoothing.pkl")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_exp_smoothing.pkl")


class MovingAveragePredictor:
    def __init__(self, window_sizes: list = None):
        self.window_sizes = window_sizes or [3, 5, 10, 20]
        
    def predict_next(self, data: np.ndarray) -> float:
        predictions = []
        
        for window in self.window_sizes:
            if len(data) >= window:
                ma = np.mean(data[-window:])
                predictions.append(ma)
        
        return np.mean(predictions) if predictions else data[-1]
    
    def weighted_moving_average(self, data: np.ndarray, window: int = 10) -> float:
        if len(data) < window:
            window = len(data)
        
        weights = np.arange(1, window + 1)
        wma = np.sum(weights * data[-window:]) / np.sum(weights)
        return wma
