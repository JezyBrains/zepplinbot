import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import joblib

logger = logging.getLogger(__name__)


class ProphetPredictor:
    def __init__(self, changepoint_prior_scale: float = 0.05, 
                 seasonality_mode: str = 'multiplicative'):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None
        
    def prepare_data(self, data: np.ndarray, timestamps: pd.DatetimeIndex = None) -> pd.DataFrame:
        if timestamps is None:
            timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(data), freq='D')
        
        df = pd.DataFrame({
            'ds': timestamps,
            'y': data
        })
        return df
    
    def train(self, data: np.ndarray, timestamps: pd.DatetimeIndex = None):
        df = self.prepare_data(data, timestamps)
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        self.model.fit(df)
        logger.info("Prophet model training completed")
        
    def predict_next(self, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        predictions = forecast['yhat'].tail(steps).values
        return predictions
    
    def get_forecast_with_intervals(self, steps: int = 1) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
        return result
    
    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_prophet.pkl")
        logger.info(f"Prophet model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_prophet.pkl")
        logger.info(f"Prophet model loaded from {filepath}")
