import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging

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
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            
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
            
            X.append(features)
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
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

    def _compute_features_batch(self, windows: np.ndarray) -> np.ndarray:
        """Vectorized feature computation for sliding windows."""
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        mins = np.min(windows, axis=1)
        maxs = np.max(windows, axis=1)
        last_vals = windows[:, -1]

        # sequence[-1] - sequence[-2]
        diffs = windows[:, -1] - windows[:, -2]

        if self.sequence_length >= 10:
             means_5 = np.mean(windows[:, -5:], axis=1)
             means_10 = np.mean(windows[:, -10:], axis=1)
             last_10 = windows[:, -10:]
        else:
             # Fallback logic if sequence length is small
             means_5 = np.mean(windows[:, -5:], axis=1) if self.sequence_length >= 5 else means
             means_10 = means
             last_10 = windows

        features_base = np.column_stack([
            means, stds, mins, maxs, last_vals, diffs, means_5, means_10
        ])

        X = np.hstack([features_base, last_10])
        return X

    def predict_batch(self, full_data: np.ndarray, train_size: int) -> np.ndarray:
        """
        Batch prediction for evaluation on test set.
        Avoids iterative concatenation and single-step inference.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # We need input history for the test set.
        # Target: full_data[train_size:]
        # First input window ends at train_size-1 (last training point)
        # Last input window ends at len(full_data)-2 (second to last point)

        # Slice full_data to cover all required history
        relevant_data = full_data[train_size - self.sequence_length : -1]

        if len(relevant_data) < self.sequence_length:
             return np.array([])

        windows = sliding_window_view(relevant_data, window_shape=self.sequence_length)

        X = self._compute_features_batch(windows)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        return predictions
    
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
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            
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
            
            X.append(features)
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
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
    
    def _compute_features_batch(self, windows: np.ndarray) -> np.ndarray:
        """Vectorized feature computation for sliding windows."""
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        mins = np.min(windows, axis=1)
        maxs = np.max(windows, axis=1)
        last_vals = windows[:, -1]

        # sequence[-1] - sequence[-2]
        diffs = windows[:, -1] - windows[:, -2]

        if self.sequence_length >= 5:
             means_5 = np.mean(windows[:, -5:], axis=1)
        else:
             means_5 = means

        if self.sequence_length >= 10:
             last_10 = windows[:, -10:]
        else:
             last_10 = windows

        features_base = np.column_stack([
            means, stds, mins, maxs, last_vals, diffs, means_5
        ])

        X = np.hstack([features_base, last_10])
        return X

    def predict_batch(self, full_data: np.ndarray, train_size: int) -> np.ndarray:
        """
        Batch prediction for evaluation on test set.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        relevant_data = full_data[train_size - self.sequence_length : -1]

        if len(relevant_data) < self.sequence_length:
             return np.array([])

        windows = sliding_window_view(relevant_data, window_shape=self.sequence_length)

        X = self._compute_features_batch(windows)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        return predictions

    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_lightgbm.pkl")
        joblib.dump(self.scaler, f"{filepath}_lightgbm_scaler.pkl")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_lightgbm.pkl")
        self.scaler = joblib.load(f"{filepath}_lightgbm_scaler.pkl")
