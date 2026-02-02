## 2026-02-02 - Data Loading Performance
**Learning:** `pd.read_csv` in `FeatureEngine.load_round_data` was a critical bottleneck, taking ~130ms for 100k rows. In-memory caching reduced this to ~0.2ms (650x speedup).
**Action:** Always verify if data loading functions are called frequently and implement caching (with mtime/size invalidation) for file-based data sources.
