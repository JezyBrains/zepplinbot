import numpy as np
import pandas as pd
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
        
    def _extract_features(self, sequence: np.ndarray) -> list:
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
        return features

    def create_features(self, data: np.ndarray):
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            X.append(self._extract_features(sequence))
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
            features = self._extract_features(sequence)
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            current_data = np.append(current_data, pred)
        
        return np.array(predictions)

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")

        features_list = [self._extract_features(seq) for seq in sequences]
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
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
        
    def _extract_features(self, sequence: np.ndarray) -> list:
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
        return features

    def create_features(self, data: np.ndarray):
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            X.append(self._extract_features(sequence))
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
            features = self._extract_features(sequence)
            
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            current_data = np.append(current_data, pred)
        
        return np.array(predictions)

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")

        features_list = [self._extract_features(seq) for seq in sequences]
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filepath: str):
        joblib.dump(self.model, f"{filepath}_lightgbm.pkl")
        joblib.dump(self.scaler, f"{filepath}_lightgbm_scaler.pkl")
    
    def load_model(self, filepath: str):
        self.model = joblib.load(f"{filepath}_lightgbm.pkl")
        self.scaler = joblib.load(f"{filepath}_lightgbm_scaler.pkl")
