# Bolt's Journal

## 2024-05-22 - Initial Setup
**Learning:** Project involves data collection and prediction.
**Action:** Look for opportunities in data processing loops or model inference.

## 2026-01-26 - Vectorized LSTM Inference
**Learning:** Iterative `model.predict` calls in Keras/TensorFlow are extremely slow due to Python-to-C++ overhead.
**Action:** Implemented `predict_batch` in `LSTMPredictor` using `sliding_window_view`. Evaluation speedup ~5x (4.5s -> 0.9s). Always batch model inference!
