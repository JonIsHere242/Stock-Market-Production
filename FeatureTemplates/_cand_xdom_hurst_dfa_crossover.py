"""
Candidate feature block: xdom_hurst_dfa_crossover

DFA short/long crossover (multi-scale persistence; Peng 1994 extended).
Computes Detrended Fluctuation Analysis fluctuation F(s) at small scales
(4-16d) and large scales (16-64d) on a rolling 250d window, then produces:
  - dfa_crossover_250: slope_large - slope_small (in log-log space)
  - dfa_slope_small_250: DFA slope at small scales
  - dfa_slope_large_250: DFA slope at large scales

The crossover value signals regime transitions: when short-scale persistence
diverges from long-scale persistence, a structural break may be imminent.
Positive crossover = long-scale more persistent (trend regime).
Negative crossover = short-scale more persistent (mean-reversion regime).

Per-ticker proxy -- faithful to the original econophysics method
(Peng 1994; extended two-scale DFA) applied to log-return series.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple

METADATA = {
    "name": "xdom_hurst_dfa_crossover",
    "description": (
        "DFA short/long crossover: detrended fluctuation analysis F(s) computed "
        "at small scales (4-16d) and large scales (16-64d) within a rolling 250d "
        "window on log-returns. Produces the slope difference (large - small) in "
        "log-log space, plus the two individual slopes. A crossover/divergence "
        "between the two scale regimes signals a market-structure transition. "
        "Per-ticker proxy (econophysics DFA, Peng 1994 extended)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_hurst_dfa_crossover_val",
        "xdom_hurst_dfa_crossover_small",
        "xdom_hurst_dfa_crossover_large",
    ],
    "tags": ["cross-domain", "hurst", "dfa", "persistence", "regime", "rolling"],
    "version": "1.0.0",
    "author": "DFA short/long crossover (multi-scale persistence; Peng 1994 extended); cross-domain-method transfer from signal processing / econophysics / HRV / DSP",
}


def _dfa_fluctuation(series: np.ndarray, scale: int) -> float:
    """
    Compute DFA fluctuation F(scale) for a cumulative-sum series.
    Splits the series into non-overlapping segments of `scale` length,
    fits a linear trend in each segment, computes RMS of residuals.
    Returns F(scale) = RMS of detrended fluctuations.
    """
    n = len(series)
    if n < scale * 2:
        return np.nan
    # Number of complete segments
    n_seg = n // scale
    if n_seg < 2:
        return np.nan
    # Use only complete segments
    y = series[: n_seg * scale]
    # Reshape into segments
    segments = y.reshape(n_seg, scale)
    # For each segment, fit linear trend and compute residual variance
    t = np.arange(scale, dtype=np.float64)
    # Linear trend via least squares: t_mean, y_mean, then slope/intercept
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0:
        return np.nan
    # Vectorised across all segments
    seg_means = segments.mean(axis=1, keepdims=True)  # (n_seg, 1)
    t_centered = t - t_mean  # (scale,)
    # slopes shape: (n_seg,)
    slopes = ((segments - seg_means) * t_centered).sum(axis=1) / t_var
    intercepts = seg_means[:, 0] - slopes * t_mean
    # Fitted values: (n_seg, scale)
    fitted = slopes[:, np.newaxis] * t[np.newaxis, :] + intercepts[:, np.newaxis]
    residuals = segments - fitted
    rms = np.sqrt((residuals ** 2).mean())
    return rms


def _dfa_slope(cum_series: np.ndarray, scales: List[int]) -> float:
    """
    Compute the DFA scaling exponent (slope in log-log F(s) vs s plot)
    across the given list of scales. Returns NaN if fewer than 2 valid points.
    """
    log_s = []
    log_f = []
    for s in scales:
        f = _dfa_fluctuation(cum_series, s)
        if f is not None and not np.isnan(f) and f > 0:
            log_s.append(np.log(s))
            log_f.append(np.log(f))
    if len(log_s) < 2:
        return np.nan
    log_s_arr = np.array(log_s)
    log_f_arr = np.array(log_f)
    # OLS slope
    ls_mean = log_s_arr.mean()
    lf_mean = log_f_arr.mean()
    denom = ((log_s_arr - ls_mean) ** 2).sum()
    if denom == 0:
        return np.nan
    slope = ((log_s_arr - ls_mean) * (log_f_arr - lf_mean)).sum() / denom
    return slope


# Small and large scale lists (integer scales)
_SMALL_SCALES = [4, 6, 8, 10, 12, 16]   # 4-16 days
_LARGE_SCALES = [16, 20, 25, 32, 40, 50, 64]  # 16-64 days
_WINDOW = 250


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    # Initialise output arrays with NaN
    vals = np.full(n, np.nan, dtype=np.float64)
    small_slopes = np.full(n, np.nan, dtype=np.float64)
    large_slopes = np.full(n, np.nan, dtype=np.float64)

    close = df["Close"].values.astype(np.float64)

    # Need at least _WINDOW rows for the first valid output
    if n < _WINDOW:
        df["xdom_hurst_dfa_crossover_val"] = np.nan
        df["xdom_hurst_dfa_crossover_small"] = np.nan
        df["xdom_hurst_dfa_crossover_large"] = np.nan
        return df

    # Compute log returns once for the entire series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Guard: replace non-positive closes with nan
        safe_close = np.where(close > 0, close, np.nan)
        log_ret = np.full(n, np.nan, dtype=np.float64)
        log_ret[1:] = np.log(safe_close[1:] / safe_close[:-1])

    # Rolling over each window position
    for i in range(_WINDOW - 1, n):
        window_ret = log_ret[i - _WINDOW + 1 : i + 1]
        # Skip if too many NaNs
        if np.sum(~np.isnan(window_ret)) < _WINDOW * 0.8:
            continue
        # Fill NaN log-returns with 0 for cumsum (conservative)
        filled = np.where(np.isnan(window_ret), 0.0, window_ret)
        # DFA requires cumulative sum (profile) of zero-mean series
        demeaned = filled - filled.mean()
        cum = np.cumsum(demeaned)

        sl_small = _dfa_slope(cum, _SMALL_SCALES)
        sl_large = _dfa_slope(cum, _LARGE_SCALES)

        small_slopes[i] = sl_small
        large_slopes[i] = sl_large
        if not np.isnan(sl_small) and not np.isnan(sl_large):
            vals[i] = sl_large - sl_small

    df["xdom_hurst_dfa_crossover_val"] = vals
    df["xdom_hurst_dfa_crossover_small"] = small_slopes
    df["xdom_hurst_dfa_crossover_large"] = large_slopes
    return df
