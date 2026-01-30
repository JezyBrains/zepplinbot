## 2024-05-23 - Caching in FeatureEngine
**Learning:** Frequent file I/O in `load_round_data` was a significant bottleneck (~150ms per call for 100k rows). Caching based on file mtime/size reduced this to <2ms, a 100x improvement.
**Action:** When working with frequently accessed data from files, always implement caching with invalidation checks (mtime/size) to avoid redundant I/O, especially in real-time paths. Also, use `.copy()` on cached mutable objects to prevent side-effect corruption.
