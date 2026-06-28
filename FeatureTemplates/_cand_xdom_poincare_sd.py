"""
Poincare plot SD1/SD2 nonlinear geometry from HRV / signal-processing.

Per-ticker rolling Poincare scatter: for each day t, pair (r[t-1], r[t])
inside a trailing 80-day window. SD1 = spread along the anti-diagonal
(y = -x), reflecting short-term consecutive variability. SD2 = spread
along the main diagonal (y = x), reflecting long-term trend variability.
The ratio SD1/SD2 captures nonlinear autocorrelation shape; a low ratio
means returns are autocorrelated / trendy; a high ratio means consecutive
returns are noisy/mean-reverting relative to the level of variability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

METADATA = {
    "name": "xdom_poincare_sd",
    "description": (
        "Rolling 80-day Poincare plot SD1/SD2 on daily log-returns. "
        "SD1 = std of (r[t] - r[t-1]) / sqrt(2) (anti-diagonal, short-term variability). "
        "SD2 = std of (r[t] + r[t-1]) / sqrt(2) (main diagonal, long-term variability). "
        "Produces: ratio SD1/SD2 (nonlinear autocorrelation shape), raw SD1, raw SD2. "
        "Per-ticker proxy; method originates in HRV / econophysics (see AUTHORS). "
        "Cross-sectional ranking deferred to downstream framework."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_poincare_sd_ratio_80",
        "xdom_poincare_sd_sd1_80",
        "xdom_poincare_sd_sd2_80",
    ],
    "tags": ["cross-domain", "nonlinear", "autocorrelation", "poincare", "hrv", "volatility"],
    "version": "1.0.0",
    "author": "Poincare plot SD1/SD2 (heart-rate-variability nonlinear geometry); Cross-domain method transfer (signal processing / econophysics / HRV / DSP). Spec: xdom_poincare_sd.",
}

_WINDOW = 80


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling Poincare SD1, SD2, and their ratio over a 80-day window.

    For each bar t within the rolling window we have pairs (r[t-1], r[t]).
    In Poincare-plot geometry:
      SD1 = std of projections onto the anti-diagonal direction = std((r[t] - r[t-1]) / sqrt(2))
      SD2 = std of projections onto the main diagonal direction  = std((r[t] + r[t-1]) / sqrt(2))

    Both are computed on the trailing window of consecutive-return pairs, with
    no lookahead: at each row t, we use pairs (r[t-w+1],r[t-w+2]), ..., (r[t-1],r[t]).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        close = df["Close"].to_numpy(dtype=np.float64)
        n = len(close)

        # Log returns; r[0] = NaN (no prior bar)
        log_ret = np.full(n, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = close[1:] / close[:-1]
            valid = ratio > 0
            log_ret[1:] = np.where(valid, np.log(ratio), np.nan)

        # Poincare pairs: d1[t] = (r[t] - r[t-1]) / sqrt2  (anti-diagonal)
        #                 d2[t] = (r[t] + r[t-1]) / sqrt2  (main diagonal)
        # d1[t] and d2[t] are defined for t >= 2 (need two consecutive returns).
        sqrt2 = np.sqrt(2.0)
        d1 = np.full(n, np.nan, dtype=np.float64)
        d2 = np.full(n, np.nan, dtype=np.float64)
        # r[t-1] is log_ret[t-1], r[t] is log_ret[t]
        for t in range(2, n):
            rt = log_ret[t]
            rt1 = log_ret[t - 1]
            if np.isfinite(rt) and np.isfinite(rt1):
                d1[t] = (rt - rt1) / sqrt2
                d2[t] = (rt + rt1) / sqrt2

        # Rolling std over _WINDOW bars (min_periods = max(20, _WINDOW//4))
        min_p = max(20, _WINDOW // 4)

        sd1_arr = np.full(n, np.nan, dtype=np.float64)
        sd2_arr = np.full(n, np.nan, dtype=np.float64)

        # Use pandas rolling for efficiency
        d1_s = pd.Series(d1)
        d2_s = pd.Series(d2)

        roll_sd1 = d1_s.rolling(window=_WINDOW, min_periods=min_p).std()
        roll_sd2 = d2_s.rolling(window=_WINDOW, min_periods=min_p).std()

        sd1_arr = roll_sd1.to_numpy(dtype=np.float64)
        sd2_arr = roll_sd2.to_numpy(dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_arr = np.where(sd2_arr > 0, sd1_arr / sd2_arr, np.nan)
            # Clamp out any inf (shouldn't happen but guard anyway)
            ratio_arr = np.where(np.isfinite(ratio_arr), ratio_arr, np.nan)
            sd1_arr = np.where(np.isfinite(sd1_arr), sd1_arr, np.nan)
            sd2_arr = np.where(np.isfinite(sd2_arr), sd2_arr, np.nan)

        df["xdom_poincare_sd_ratio_80"] = ratio_arr
        df["xdom_poincare_sd_sd1_80"] = sd1_arr
        df["xdom_poincare_sd_sd2_80"] = sd2_arr

    return df
