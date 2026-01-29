## 2026-01-29 - File I/O Caching
**Learning:** Repetitive reading of static or slowly changing CSV files is a major bottleneck (found 4.4ms/call overhead). Caching with `mtime` and `size` invalidation reduced this to 0.2ms/call (~22x speedup).
**Action:** Always check if file reads in frequently called methods can be cached. Ensure cached mutable objects (DataFrames) are copied before returning.
