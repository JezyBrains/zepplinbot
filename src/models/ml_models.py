import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    def __init__(self, n_estimators: int = 1000, max_depth: int = 7, 
                 learning_rate: float = 0.01, sequence_length: int = 50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def create_features(self, data: np.ndarray):
        # Vectorized feature creation
        if len(data) <= self.sequence_length:
            return np.array([]), np.array([])
            
        # Create windows of size sequence_length
        # windows shape: (n_windows, sequence_length)
        windows = sliding_window_view(data, window_shape=self.sequence_length)

        # We need y to be the value *after* each sequence.
        # The last window in `windows` ends at the very last element of data.
        # So it has no "next" element. We exclude it.
        X_windows = windows[:-1]
        y = data[self.sequence_length:]

        if len(X_windows) == 0:
             return np.array([]), np.array([])

        # Compute features vectorized
        # 1. Basic stats
        f_mean = np.mean(X_windows, axis=1, keepdims=True)
        f_std = np.std(X_windows, axis=1, keepdims=True)
        f_min = np.min(X_windows, axis=1, keepdims=True)
        f_max = np.max(X_windows, axis=1, keepdims=True)

        # 2. Last element
        f_last = X_windows[:, -1:]

        # 3. Difference (last - second_last)
        if self.sequence_length > 1:
            f_diff = (X_windows[:, -1] - X_windows[:, -2]).reshape(-1, 1)
        else:
            f_diff = np.zeros((X_windows.shape[0], 1))
            
        # 4. Mean of last 5
        f_mean5 = np.mean(X_windows[:, -5:], axis=1, keepdims=True)

        # 5. Mean of last 10 (specific to XGBoostPredictor)
        f_mean10 = np.mean(X_windows[:, -10:], axis=1, keepdims=True)

        # 6. Last 10 elements (raw)
        f_last10 = X_windows[:, -10:]
        
        # Combine all features
        X = np.hstack([
            f_mean, f_std, f_min, f_max, f_last, f_diff, f_mean5, f_mean10, f_last10
        ])

        return X, y
    
    def train(self, data: np.ndarray):
        X, y = self.create_features(data)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create features")
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42
        )
        
        self.model.fit(X_scaled, y, verbose=False)
        logger.info("XGBoost model training completed")
        
    def predict_next(self, recent_data: np.ndarray, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(steps):
            sequence = current_data[-self.sequence_length:]
            
            features = [
                np.mean(sequence),
                np.std(sequence),
                np.min(sequence),
                np.max(sequence),
                sequence[-1],
                sequence[-1] - sequence[-2] if len(sequence) > 1 else 0,
                np.mean(sequence[-5:]) if len(sequence) >= 5 else np.mean(sequence),
                np.mean(sequence[-10:]) if len(sequence) >= 10 else np.mean(sequence),
            ]
            
            features.extend(sequence[-10:].tolist() if len(sequence) >= 10 else sequence.tolist())
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            current_data = np.append(current_data, pred)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_xgboost.pkl")
        joblib.dump(self.scaler, f"{filepath}_xgboost_scaler.pkl")
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_xgboost.pkl")
        self.scaler = joblib.load(f"{filepath}_xgboost_scaler.pkl")
        logger.info(f"XGBoost model loaded from {filepath}")


class LightGBMPredictor:
    def __init__(self, n_estimators: int = 1000, max_depth: int = 7, 
                 learning_rate: float = 0.01, sequence_length: int = 50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
    def create_features(self, data: np.ndarray):
        # Vectorized feature creation
        if len(data) <= self.sequence_length:
            return np.array([]), np.array([])
            
        windows = sliding_window_view(data, window_shape=self.sequence_length)
        X_windows = windows[:-1]
        y = data[self.sequence_length:]

        if len(X_windows) == 0:
             return np.array([]), np.array([])

        # 1. Basic stats
        f_mean = np.mean(X_windows, axis=1, keepdims=True)
        f_std = np.std(X_windows, axis=1, keepdims=True)
        f_min = np.min(X_windows, axis=1, keepdims=True)
        f_max = np.max(X_windows, axis=1, keepdims=True)

        # 2. Last element
        f_last = X_windows[:, -1:]

        # 3. Difference
        if self.sequence_length > 1:
            f_diff = (X_windows[:, -1] - X_windows[:, -2]).reshape(-1, 1)
        else:
            f_diff = np.zeros((X_windows.shape[0], 1))
            
        # 4. Mean of last 5
        f_mean5 = np.mean(X_windows[:, -5:], axis=1, keepdims=True)

        # Note: LightGBMPredictor does NOT have mean of last 10

        # 5. Last 10 elements
        f_last10 = X_windows[:, -10:]

        X = np.hstack([
            f_mean, f_std, f_min, f_max, f_last, f_diff, f_mean5, f_last10
        ])
        
        return X, y
    
    def train(self, data: np.ndarray):
        X, y = self.create_features(data)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X_scaled, y)
        logger.info("LightGBM model training completed")
        
    def predict_next(self, recent_data: np.ndarray, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(steps):
            sequence = current_data[-self.sequence_length:]
            
            features = [
                np.mean(sequence),
                np.std(sequence),
                np.min(sequence),
                np.max(sequence),
                sequence[-1],
                sequence[-1] - sequence[-2] if len(sequence) > 1 else 0,
                np.mean(sequence[-5:]) if len(sequence) >= 5 else np.mean(sequence),
            ]
            
            features.extend(sequence[-10:].tolist() if len(sequence) >= 10 else sequence.tolist())
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            current_data = np.append(current_data, pred)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_lightgbm.pkl")
        joblib.dump(self.scaler, f"{filepath}_lightgbm_scaler.pkl")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_lightgbm.pkl")
        self.scaler = joblib.load(f"{filepath}_lightgbm_scaler.pkl")
