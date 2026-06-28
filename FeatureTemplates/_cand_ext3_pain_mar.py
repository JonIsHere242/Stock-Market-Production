"""
Pain Index & MAR ratio feature block.

Rolling 120-day Pain Index (mean drawdown depth from running peak),
MAR-style ratio (trailing return / max drawdown), and Martin ratio
(trailing return / RMS drawdown, analogous to Ulcer Performance Index).
Pure OHLCV -- no external helpers needed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_pain_mar",
    "description": (
        "Rolling 120-day drawdown-risk metrics computed per ticker from daily Close. "
        "(1) ext3_pain_mar_pain: mean fractional depth below running 120d peak "
        "(Pain Index -- higher = more sustained pain). "
        "(2) ext3_pain_mar_mar: trailing 120d return divided by max drawdown over "
        "that window (MAR-style ratio -- higher = better risk-adjusted return). "
        "(3) ext3_pain_mar_martin: trailing return divided by RMS drawdown depth "
        "(Martin / Ulcer-Performance-Index proxy). "
        "All windows are causal (rolling, no lookahead). Guard against zero max-dd "
        "with np.nan. Orthogonal to momentum/vol -- captures sustained underwater "
        "duration and max-pain signal axes."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_pain_mar_pain",    # mean drawdown depth (Pain Index), 0..1
        "ext3_pain_mar_mar",     # MAR-style ratio (ret / max_dd)
        "ext3_pain_mar_martin",  # Martin ratio (ret / rms_dd)
    ],
    "tags": ["drawdown", "risk", "pain_index", "mar", "martin", "ulcer"],
    "version": "1.0.0",
    "author": "Round-4 expansion (xdom2_downside_beta); spec ext3_pain_mar",
}

_WINDOW = 120  # trading days


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    pain = np.full(n, np.nan)
    mar = np.full(n, np.nan)
    martin = np.full(n, np.nan)

    # We need at least WINDOW bars to produce a value (first valid at index WINDOW-1).
    # Use a sliding approach: for each bar t (0-indexed), the window is
    # close[t-WINDOW+1 : t+1].  We compute the running peak within that
    # window and derive drawdown depths.

    # Vectorised via stride tricks is tricky for running-peak-within-window;
    # use an efficient numpy cummax trick instead.

    # Pre-compute for every possible right-endpoint t:
    # We iterate once and maintain a deque of peak candidates is O(n) but
    # complex. Simpler and fast enough (<100ms on 700 rows):
    # For each endpoint t, window slice is W rows.  Use numpy on each slice
    # but avoid a Python loop over every single row -- instead batch with
    # numpy operations on the full 2-D strided view.

    # sliding_window_view is available in numpy >= 1.20
    if n < _WINDOW:
        df["ext3_pain_mar_pain"] = np.nan
        df["ext3_pain_mar_mar"] = np.nan
        df["ext3_pain_mar_martin"] = np.nan
        return df

    from numpy.lib.stride_tricks import sliding_window_view  # numpy >= 1.20

    # shape (n - WINDOW + 1, WINDOW)
    windows = sliding_window_view(close, _WINDOW)  # 2-D, no copy

    # Running peak within each window row: cummax along axis=1
    # np.maximum.accumulate works along axis=1
    running_peak = np.maximum.accumulate(windows, axis=1)  # same shape

    # Drawdown depth at each step within the window: (peak - price) / peak
    # Guard: if peak == 0 -> nan
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_depth = np.where(
            running_peak == 0,
            np.nan,
            (running_peak - windows) / running_peak,
        )  # shape (M, WINDOW), values in [0, 1]

    M = windows.shape[0]  # n - WINDOW + 1

    # Pain Index = mean dd_depth across the window (nanmean per row)
    pain_vals = np.nanmean(dd_depth, axis=1)  # (M,)

    # Max drawdown = max dd_depth in window
    max_dd = np.nanmax(dd_depth, axis=1)  # (M,)

    # Trailing return = (last price / first price) - 1 within window
    trailing_ret = np.where(
        windows[:, 0] == 0,
        np.nan,
        (windows[:, -1] / windows[:, 0]) - 1.0,
    )  # (M,)

    # MAR = trailing_ret / max_dd  (nan when max_dd == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mar_vals = np.where(max_dd == 0.0, np.nan, trailing_ret / max_dd)

    # Martin (Ulcer Performance Index proxy) = trailing_ret / rms_dd
    rms_dd = np.sqrt(np.nanmean(dd_depth ** 2, axis=1))  # (M,)
    with np.errstate(divide="ignore", invalid="ignore"):
        martin_vals = np.where(rms_dd == 0.0, np.nan, trailing_ret / rms_dd)

    # Map back: valid results occupy indices [WINDOW-1 .. n-1]
    start = _WINDOW - 1
    pain[start:] = pain_vals
    mar[start:] = mar_vals
    martin[start:] = martin_vals

    # Replace any inf/-inf with nan (safety)
    pain = np.where(np.isfinite(pain), pain, np.nan)
    mar = np.where(np.isfinite(mar), mar, np.nan)
    martin = np.where(np.isfinite(martin), martin, np.nan)

    df["ext3_pain_mar_pain"] = pain
    df["ext3_pain_mar_mar"] = mar
    df["ext3_pain_mar_martin"] = martin

    return df
