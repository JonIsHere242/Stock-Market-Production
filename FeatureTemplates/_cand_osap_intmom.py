"""
Intermediate Momentum (Novy-Marx 2012) -- per-ticker OHLCV implementation.

The cross-sectional factor ranks stocks by their cumulative return from
month t-12 to t-6 (skipping the most recent 6 months).  This per-ticker
block captures the same signal as a time-series level (the raw 6-month
log-return ending 6 months ago), a 63-day rolling z-score of that level
(normalises the signal across its own history), and a 21-day slope of the
level (captures whether the intermediate-horizon momentum is accelerating or
fading).  All computations are strictly causal (rolling look-back only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_intmom",
    "description": (
        "Intermediate Momentum (Novy-Marx 2012): cumulative log-return from "
        "approx t-252 to t-126 trading days (months t-12 to t-6), computed "
        "per ticker from daily Close prices.  Produces: (1) osap_intmom_ret "
        "= raw 6-month log-return in the [t-252, t-126] window; "
        "(2) osap_intmom_zscore = 63-day rolling z-score of that level "
        "(self-normalised, lookahead-free); (3) osap_intmom_slope = linear "
        "trend slope of the level over the most recent 21 observations "
        "(momentum of momentum).  Cross-sectional ranking is not available "
        "per-ticker; the raw return is the closest faithful proxy -- "
        "downstream XS-rank overlays can be applied at ensemble time."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_intmom_ret",
        "osap_intmom_zscore",
        "osap_intmom_slope",
    ],
    "tags": ["momentum", "intermediate_momentum", "novy_marx", "price"],
    "version": "1.0",
    "author": "OpenSourceAP (Chen-Zimmermann); Novy-Marx 2012",
}

# Trading-day approximations
_DAYS_12M = 252   # t-12 months look-back start
_DAYS_6M  = 126   # t-6  months look-back end (skip recent 6 months)
_ZSCORE_WIN = 63  # rolling window for z-score normalisation (~3 months)
_SLOPE_WIN  = 21  # rolling window for slope (~1 month)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    ret    = np.full(n, np.nan)
    zscore = np.full(n, np.nan)
    slope  = np.full(n, np.nan)

    # --- 1. Intermediate-momentum return: log(Close[t-126] / Close[t-252]) ---
    # Requires at least _DAYS_12M + 1 rows.
    for i in range(_DAYS_12M, n):
        c_end   = close[i - _DAYS_6M]    # price at ~t-6 months
        c_start = close[i - _DAYS_12M]   # price at ~t-12 months
        if c_start > 0 and c_end > 0:
            ret[i] = np.log(c_end / c_start)
        # else remains NaN

    # --- 2. Rolling z-score of the level series ---
    # Use pandas rolling for clean NaN-aware mean/std
    ret_s = pd.Series(ret)
    roll  = ret_s.rolling(_ZSCORE_WIN, min_periods=max(2, _ZSCORE_WIN // 2))
    rmean = roll.mean().to_numpy()
    rstd  = roll.std(ddof=1).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(rstd > 0, (ret - rmean) / rstd, np.nan)
    zscore = np.where(np.isfinite(z), z, np.nan)

    # --- 3. Slope of the level over last _SLOPE_WIN observations ---
    # OLS slope via vectorised sliding sums (avoids Python loop overhead).
    w = _SLOPE_WIN
    if n >= w:
        # x values 0..w-1, centred: pre-compute once
        x = np.arange(w, dtype=np.float64)
        x -= x.mean()
        ss_x = float((x ** 2).sum())

        # Build (n-w+1) x w matrix via stride tricks
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(ret, window_shape=w)   # shape (n-w+1, w)

        # Mask rows that have any NaN (OLS undefined)
        valid = ~np.any(np.isnan(windows), axis=1)

        slopes_out = np.full(windows.shape[0], np.nan)
        if valid.any():
            y = windows[valid]          # shape (k, w)
            y_dm = y - y.mean(axis=1, keepdims=True)
            cov_xy = (y_dm * x).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                slopes_out[valid] = np.where(ss_x > 0, cov_xy / ss_x, np.nan)

        # Align: sliding_window_view index i corresponds to df row i + w - 1
        slope[w - 1:] = slopes_out

    df["osap_intmom_ret"]    = ret
    df["osap_intmom_zscore"] = zscore
    df["osap_intmom_slope"]  = slope

    return df
