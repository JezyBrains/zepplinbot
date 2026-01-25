## 2024-05-23 - Prediction Engine Bottleneck
**Learning:** The `PredictionEngine.evaluate_models` method used an iterative loop with `np.concatenate` to reconstruct historical windows for each test point. This created an O(N^2) data copying bottleneck and forced single-item inference, preventing vectorization benefits of ML models.
**Action:** Always look for rolling window operations that reconstruct arrays inside loops. Use `numpy.lib.stride_tricks.sliding_window_view` to create zero-copy views and enable batch inference for orders of magnitude speedup.
