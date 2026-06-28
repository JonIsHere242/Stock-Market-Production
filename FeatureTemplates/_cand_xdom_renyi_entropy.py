"""
Renyi quadratic entropy of the return distribution (per-ticker, rolling).

H2 = -log(sum p_i^2) over a 10-bin histogram of daily close returns built from
rolling 80-day windows.  More sensitive to distribution concentration / fat tails
than Shannon entropy; low H2 = returns clustering into few magnitude bins.

Cross-domain method transfer: signal processing / econophysics / HRV / DSP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_renyi_entropy",
    "description": (
        "Rolling 80-day Renyi quadratic entropy (order 2) of daily log-return "
        "distribution, approximated via a 10-bin histogram built from rolling "
        "sample quantiles.  H2 = -log(sum p_i^2); low H2 means returns cluster "
        "into few magnitude bins (fat-tail concentration).  Also emits a 20-day "
        "z-score of H2 as a dynamic/change variant.  Per-ticker; no cross-sectional "
        "data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_renyi_entropy_h2",        # rolling 80d Renyi H2
        "xdom_renyi_entropy_h2_z20",    # 20-day z-score of H2 (trend/change)
        "xdom_renyi_entropy_h2_slope",  # 10-day slope of H2 (sign of recent shift)
    ],
    "tags": ["entropy", "distribution", "fat-tail", "cross-domain", "econophysics"],
    "version": "1.0",
    "author": (
        "Renyi quadratic entropy of the return distribution (generalized entropy); "
        "SOURCE: Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "AUTHORS/CITATION: Renyi quadratic entropy of the return distribution (generalized entropy)"
    ),
}

_N_BINS = 10
_WIN = 80
_Z_WIN = 20
_SLOPE_WIN = 10


def _renyi2_from_window(window: np.ndarray) -> float:
    """Compute Renyi H2 for a 1-D array of returns using a 10-bin histogram."""
    if len(window) < _N_BINS + 1:
        return np.nan
    # Build uniform histogram over [min, max] of the window
    counts, _ = np.histogram(window, bins=_N_BINS)
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    sq_sum = float(np.dot(p, p))
    if sq_sum <= 0.0:
        return np.nan
    return float(-np.log(sq_sum))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Daily log returns; first row is NaN
    close = df["Close"].values.astype(np.float64)
    n = len(close)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.empty(n, dtype=np.float64)
        log_ret[0] = np.nan
        prev = close[:-1]
        curr = close[1:]
        ratio = np.where(prev > 0.0, curr / prev, np.nan)
        log_ret[1:] = np.where(ratio > 0.0, np.log(ratio), np.nan)

    # Rolling Renyi H2 over _WIN bars
    h2 = np.full(n, np.nan, dtype=np.float64)
    for i in range(_WIN - 1, n):
        window = log_ret[i - _WIN + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= _N_BINS + 1:
            h2[i] = _renyi2_from_window(valid)

    df["xdom_renyi_entropy_h2"] = h2

    # 20-day rolling z-score of H2
    h2_s = pd.Series(h2, index=df.index)
    roll_mean = h2_s.rolling(_Z_WIN, min_periods=_Z_WIN).mean()
    roll_std  = h2_s.rolling(_Z_WIN, min_periods=_Z_WIN).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z20 = np.where(roll_std.values > 0.0,
                       (h2_s.values - roll_mean.values) / roll_std.values,
                       np.nan)
    df["xdom_renyi_entropy_h2_z20"] = z20

    # 10-day OLS slope of H2 (captures rate of entropy change)
    x = np.arange(_SLOPE_WIN, dtype=np.float64)
    x_dm = x - x.mean()
    x_ss = float(np.dot(x_dm, x_dm))

    slope = np.full(n, np.nan, dtype=np.float64)
    if x_ss > 0.0:
        for i in range(_SLOPE_WIN - 1, n):
            y = h2[i - _SLOPE_WIN + 1 : i + 1]
            if not np.any(np.isnan(y)):
                y_dm = y - y.mean()
                slope[i] = float(np.dot(x_dm, y_dm)) / x_ss

    df["xdom_renyi_entropy_h2_slope"] = slope

    return df
