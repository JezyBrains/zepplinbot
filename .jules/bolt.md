## 2024-05-23 - Vectorized Time Series Feature Creation
**Learning:** Using `numpy.lib.stride_tricks.sliding_window_view` allows for fully vectorized feature creation (mean, std, etc. over rolling windows) which is significantly faster (~50-75x) than iterating through the data in a Python loop.
**Action:** When implementing rolling window operations for feature engineering, always use vectorized numpy operations or pandas rolling windows instead of explicit loops.
