"""
Wald-Wolfowitz runs statistic (nonparametric randomness test) applied per-ticker
on a rolling window of daily return signs.

Negative z  => too few runs (trending / persistent price motion)
Positive z  => too many runs (choppy / mean-reverting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_runs_test",
    "description": (
        "Rolling Wald-Wolfowitz runs statistic on daily-return sign sequences. "
        "For each bar we look back 60 days and compute the standardised runs count: "
        "z = (R - mu_R) / sigma_R where mu_R and sigma_R are the analytic mean/sd "
        "under the null of iid returns given n+ and n- (number of positive and "
        "negative returns). Negative z indicates too few runs (trending/persistent); "
        "positive z indicates too many runs (choppy/mean-reverting). "
        "We also produce a 20-bar EWM slope of runs_z to capture regime acceleration, "
        "and a short 20-day window version for higher-frequency signal. "
        "Strictly per-ticker time-series proxy (no cross-sectional component)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_runs_test_z60",       # standardised runs statistic, 60-day window
        "xdom_runs_test_z20",       # standardised runs statistic, 20-day window
        "xdom_runs_test_z60_slope", # 10-bar EWM slope of z60 (acceleration)
    ],
    "tags": ["cross-domain", "runs-test", "randomness", "trend", "mean-reversion", "nonparametric"],
    "version": "1.0",
    "author": (
        "Wald-Wolfowitz runs statistic (nonparametric randomness test); "
        "spec source: Cross-domain method transfer (signal processing / econophysics / HRV / DSP)"
    ),
}


def _runs_z(sign_arr: np.ndarray) -> float:
    """
    Compute the Wald-Wolfowitz standardised runs statistic for a 1-D binary
    sequence of +1 / -1 values.  Returns nan when the analytic sigma is zero
    (degenerate cases: all same sign, or n < 2).
    """
    n = len(sign_arr)
    if n < 2:
        return np.nan

    n_pos = int(np.sum(sign_arr > 0))
    n_neg = int(np.sum(sign_arr < 0))
    # Zeros are excluded (flat days); update effective n
    n_eff = n_pos + n_neg
    if n_eff < 2 or n_pos == 0 or n_neg == 0:
        return np.nan

    # Count runs on the sign sequence (ignoring zeros)
    seq = sign_arr[sign_arr != 0]
    if len(seq) < 2:
        return np.nan

    runs = 1 + int(np.sum(seq[1:] != seq[:-1]))

    # Analytic mean and variance under null of randomness
    mu = (2.0 * n_pos * n_neg) / n_eff + 1.0
    var_num = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_eff)
    var_den = (n_eff ** 2) * (n_eff - 1.0)
    if var_den <= 0:
        return np.nan
    var = var_num / var_den
    if var <= 0:
        return np.nan

    return (runs - mu) / np.sqrt(var)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # Daily log-returns; sign: +1, 0, -1
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = np.diff(np.log(np.where(close > 0, close, np.nan)))
    signs = np.sign(ret)  # +1, 0, -1; nan propagates to 0 via np.sign(nan)=0 in old numpy,
                          # but np.sign(np.nan) gives nan in modern numpy, so force:
    signs = np.where(np.isnan(ret), 0.0, np.sign(ret))

    # --- 60-day rolling runs z ---
    W60 = 60
    z60 = np.full(n, np.nan, dtype=np.float64)
    for i in range(W60 - 1, n):
        window = signs[i - W60 + 1: i + 1]
        z60[i] = _runs_z(window)

    # --- 20-day rolling runs z ---
    W20 = 20
    z20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(W20 - 1, n):
        window = signs[i - W20 + 1: i + 1]
        z20[i] = _runs_z(window)

    # --- 10-bar EWM slope of z60 ---
    s_z60 = pd.Series(z60)
    ewm_z60 = s_z60.ewm(span=10, min_periods=5).mean()
    # slope = first difference of the smoothed series
    slope = ewm_z60.diff()

    df["xdom_runs_test_z60"] = z60
    df["xdom_runs_test_z20"] = z20
    df["xdom_runs_test_z60_slope"] = slope.to_numpy()

    return df
