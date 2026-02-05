## 2026-02-05 - Caching CSV Reads in Feature Engine
**Learning:** The `FeatureEngine` was reading a CSV file on every call to `load_round_data`, which is used frequently by feature calculation methods. This caused a significant IO bottleneck (~45ms per call for 50k rows).
**Action:** Implemented file-stat based caching (mtime + size). Always return `df.copy()` from cached dataframes to prevent side-effect corruption. Benchmarked a ~34x speedup.
