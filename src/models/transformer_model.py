#!/usr/bin/env python3
"""
Transformer-based Sequence Predictor for Cryptographic Pattern Analysis

Uses attention mechanism to detect long-range dependencies in crash sequences.
Designed to exploit autocorrelation patterns detected in cryptanalysis.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import List, Tuple, Optional
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head attention.
    
    This allows the model to "see" relationships between any two points
    in the sequence, not just adjacent ones.
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    """
    Adds positional information to the sequence.
    
    This helps the model understand the ORDER of crashes, which is
    critical for detecting patterns at specific lags (15, 20, 35, etc.)
    """
    
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length

    def call(self, x):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = self.position_embedding(positions)
        return x + positions


class TransformerPredictor:
    """
    Transformer-based crash point predictor.
    
    Architecture:
    - Input: Sequence of previous crash points
    - Positional embedding: Adds sequence order information
    - Transformer blocks: Multi-head attention to find patterns
    - Output: Predicted next crash point(s)
    """
    
    def __init__(self, sequence_length: int = 50, embed_dim: int = 64,
                 num_heads: int = 4, ff_dim: int = 128, num_blocks: int = 2):
        """
        Args:
            sequence_length: How many previous crashes to consider
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            num_blocks: Number of transformer blocks
        """
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def _build_model(self):
        """Build the Transformer architecture."""
        inputs = layers.Input(shape=(self.sequence_length, 1))
        
        # Project input to embedding dimension
        x = layers.Dense(self.embed_dim)(inputs)
        
        # Add positional information
        x = PositionalEmbedding(self.sequence_length, self.embed_dim)(x)
        
        # Stack transformer blocks
        for _ in range(self.num_blocks):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers for prediction
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training.
        
        Example: [1.5, 2.0, 3.0, 1.2, 4.5] with sequence_length=3
        X: [[1.5, 2.0, 3.0], [2.0, 3.0, 1.2]]
        y: [1.2, 4.5]
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: np.ndarray, epochs: int = 100, 
              validation_split: float = 0.2, verbose: int = 0):
        """
        Train the Transformer model.
        
        Args:
            data: Array of crash points
            epochs: Training epochs
            validation_split: Fraction for validation
            verbose: Verbosity level
        """
        if len(data) < self.sequence_length + 10:
            raise ValueError(f"Need at least {self.sequence_length + 10} data points")
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._prepare_sequences(data_scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        logger.info(f"Training Transformer with {len(X)} sequences...")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Attention heads: {self.num_heads}")
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        
        logger.info(f"Transformer training completed. Final MAE: {final_mae:.4f}")
        
        self.is_trained = True
        return history
    
    def predict_next(self, recent_data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict next crash point(s).
        
        Args:
            recent_data: Recent crash points (at least sequence_length)
            steps: Number of future points to predict
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} recent data points")
        
        # Use last sequence_length points
        sequence = recent_data[-self.sequence_length:]
        
        # Normalize
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = sequence_scaled.copy()
        
        for _ in range(steps):
            # Reshape for model
            X = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict
            pred_scaled = self.model.predict(X, verbose=0)[0, 0]
            
            # Inverse transform
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save model and scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        self.model.save(f"{filepath}_transformer.keras")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_blocks': self.num_blocks
        }
        joblib.dump(config, f"{filepath}_config.pkl")
        
        logger.info(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and scaler."""
        self.model = keras.models.load_model(
            f"{filepath}_transformer.keras",
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'PositionalEmbedding': PositionalEmbedding
            }
        )
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        # Load config
        config = joblib.load(f"{filepath}_config.pkl")
        self.sequence_length = config['sequence_length']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.ff_dim = config['ff_dim']
        self.num_blocks = config['num_blocks']
        
        self.is_trained = True
        logger.info(f"Transformer model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Simulate data with autocorrelation (like Zeppelin)
    data = []
    for i in range(200):
        if i < 50:
            val = np.random.exponential(2.5)
        else:
            # Add autocorrelation
            val = 0.3 * data[i-15] + 0.7 * np.random.exponential(2.5)
        data.append(val)
    
    data = np.array(data)
    
    # Train
    predictor = TransformerPredictor(sequence_length=50)
    predictor.train(data[:150], epochs=50, verbose=1)
    
    # Predict
    predictions = predictor.predict_next(data[100:150], steps=5)
    
    print(f"\nPredictions: {predictions}")
    print(f"Actual next 5: {data[150:155]}")
