## 2026-02-03 - CSV Reloading Bottleneck
**Learning:** `FeatureEngine.load_round_data` was reading the full CSV file on every call, causing significant latency (~133ms for 100k rows).
**Action:** Implemented file-stat based caching (mtime/size). Always return a copy of cached mutable data (DataFrames) to prevent side effects.
