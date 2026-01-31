## 2024-05-23 - Import Path Patching
**Learning:** When `src` is added to `sys.path` and contains `__init__.py`, importing `src.module` and `module` creates two different module objects. `unittest.mock.patch` must target the exact path used by the code under test (e.g. `module.GLOBAL_VAR` instead of `src.module.GLOBAL_VAR` if the test imports `module`).
**Action:** Always check how the module is imported in the test environment vs application code when patching globals.

## 2024-05-23 - Ljung-Box Test Edge Case
**Learning:** The Ljung-Box test implementation crashed when `max_lag` (default 20) was smaller than the hardcoded slice `autocorr[:10]`. It requires dynamic slicing `min(len(autocorr), 10)` to handle short time series or small lags.
**Action:** Always verify array shapes match before broadcasting operations in numpy, especially when dealing with variable length inputs.
