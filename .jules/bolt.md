## 2024-05-23 - Rolling Window Concatenation Anti-Pattern
**Learning:** Found an O(N^2) anti-pattern in rolling window evaluation where `np.concatenate` was used inside a loop to append one new data point to the history. This resulted in excessive memory copying (GBs of data for moderate datasets).
**Action:** Use array slicing `data[:i]` instead of concatenation `np.concatenate([history, new_point])` when the full dataset is available. This reduces complexity to O(N) (or O(1) per step) regarding data preparation.
