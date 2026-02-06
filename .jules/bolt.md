## 2025-02-06 - Caching heavy CSV loads
**Learning:** Frequent calls to `pd.read_csv` for data that changes infrequently causes massive bottlenecks (0.25s+ per call for 100k rows).
**Action:** Implement caching based on `os.stat` (mtime + size) for any file-based data loader. Ensure to return `.copy()` to protect cache integrity.
