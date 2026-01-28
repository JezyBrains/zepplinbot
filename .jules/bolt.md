## 2024-05-22 - Caching Mutable Dataframes
**Learning:** When implementing in-memory caching for large objects like Pandas DataFrames, returning a reference to the cached object is dangerous if the caller modifies it. This corrupts the cache for all future calls.
**Action:** Always return `.copy()` of cached mutable objects unless strict immutability is enforced or performance profiling explicitly demands zero-copy access (and you trust the caller).
