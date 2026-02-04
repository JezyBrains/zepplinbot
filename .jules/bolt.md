## 2024-05-22 - [Optimized FeatureEngine Data Loading]
**Learning:** Frequent file I/O operations (`pd.read_csv`) in critical paths (like feature generation) can cause massive bottlenecks. Simple mtime-based caching can provide orders of magnitude speedup (33x in this case).
**Action:** Always check if static or semi-static data is being reloaded unnecessarily in loops or high-frequency methods. Use `os.stat` for cheap invalidation checks.
