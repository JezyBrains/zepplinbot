## 2024-05-22 - FeatureEngine CSV Caching
**Learning:** Repeatedly reading static or slowly changing CSV files (`pd.read_csv`) for every feature calculation creates significant overhead.
**Action:** Implement file metadata-based caching (checking `st_mtime` and `st_size`) to skip redundant I/O. Always return `.copy()` of cached DataFrames to prevent external mutation.
