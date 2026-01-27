import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class LSTMPredictor:
    def __init__(self, sequence_length: int = 50, units: list = None, 
                 dropout: float = 0.2, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.units = units or [128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = keras.Sequential()
        
        model.add(layers.LSTM(self.units[0], return_sequences=True, 
                             input_shape=input_shape))
        model.add(layers.Dropout(self.dropout))
        
        for units in self.units[1:]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(self.dropout))
        
        model.add(layers.LSTM(self.units[-1]))
        model.add(layers.Dropout(self.dropout))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2):
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = self.create_sequences(data_scaled)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        self.model = self.build_model((X.shape[1], 1))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.6f}")
        return history
    
    def predict_next(self, recent_data: np.ndarray, steps: int = 1) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = []
        current_sequence = recent_data[-self.sequence_length:].copy()
        current_sequence_scaled = self.scaler.transform(current_sequence.reshape(-1, 1))
        
        for _ in range(steps):
            X = current_sequence_scaled.reshape(1, self.sequence_length, 1)
            pred_scaled = self.model.predict(X, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            
            predictions.append(pred)
            
            current_sequence_scaled = np.append(current_sequence_scaled[1:], pred_scaled)
            current_sequence_scaled = current_sequence_scaled.reshape(-1, 1)
        
        return np.array(predictions)

    def predict_batch(self, data: np.ndarray, start_index: int) -> np.ndarray:
        """
        Predicts values starting from start_index using history from data.
        Returns predictions corresponding to data[start_index:].
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if start_index < self.sequence_length:
             raise ValueError("start_index must be >= sequence_length to have sufficient history")

        # Transform all data first (vectorized)
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()

        # We use sliding_window_view to get all windows of size sequence_length
        # windows[j] corresponds to data[j : j + sequence_length]
        # Input for target at index i is data[i - sequence_length : i]
        # This corresponds to windows[i - sequence_length]

        windows = sliding_window_view(data_scaled, window_shape=self.sequence_length)

        # Calculate indices into 'windows'
        # We want targets from start_index to len(data)-1
        # So we want windows from index (start_index - sequence_length) to (len(data) - 1 - sequence_length)

        start_win_idx = start_index - self.sequence_length
        # sliding_window_view returns shape (N - k + 1, k)
        # So maximum valid index is len(data) - sequence_length

        # We want input for target at len(data)-1, which is at index len(data)-1-seq_len
        end_win_idx = len(data) - self.sequence_length

        X_batch = windows[start_win_idx:end_win_idx]

        if len(X_batch) == 0:
            return np.array([])

        # Reshape for LSTM: (samples, timesteps, features)
        X_batch = X_batch.reshape(-1, self.sequence_length, 1)

        # Batch prediction
        pred_scaled = self.model.predict(X_batch, verbose=0)
        predictions = self.scaler.inverse_transform(pred_scaled)

        return predictions.flatten()
    
    def get_confidence_interval(self, recent_data: np.ndarray, 
                               n_simulations: int = 100, confidence: float = 0.95):
        predictions = []
        for _ in range(n_simulations):
            pred = self.predict_next(recent_data, steps=1)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
        upper = np.percentile(predictions, (1 + confidence) / 2 * 100)
        mean = np.mean(predictions)
        
        return mean, lower, upper
    
    def save_model(self, filepath: str):
        self.model.save(f"{filepath}_model.h5")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        logger.info(f"LSTM model loaded from {filepath}")
