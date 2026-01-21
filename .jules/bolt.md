## 2026-01-21 - [Validation Vectorization]
**Learning:** Iterative prediction in validation loops (one step at a time) is a massive bottleneck for models with high inference overhead (like LSTM/XGBoost). Batch prediction reduces overhead significantly (~23x speedup).
**Action:** Always check if models support batch prediction for evaluation tasks. Vectorize input construction to avoid repeated concatenations.
