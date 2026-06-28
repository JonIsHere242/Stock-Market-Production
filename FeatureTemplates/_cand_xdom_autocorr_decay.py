"""
xdom_autocorr_decay — Autocorrelation decay rate / first-zero lag (correlation time).

Per-ticker proxy for the cross-domain DSP/econophysics method described in the spec.
Cross-sectional comparisons are not possible per-block, but the per-ticker rolling
autocorrelation structure (first-zero lag and log-ACF slope) faithfully captures the
same economic signal: short correlation time = fast mean-reversion, long = momentum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_autocorr_decay",
    "description": (
        "Rolling 90-day autocorrelation decay of daily log-returns. Produces: "
        "(1) acf_first_zero_90 = smallest lag (1..20) at which the ACF first crosses "
        "zero — short values indicate fast mean-reversion, long values indicate "
        "momentum persistence; "
        "(2) acf_decay_rate_90 = slope of log|ACF| over lags 1..5 (OLS on log-linear "
        "fit) — more-negative slope = faster exponential decay of serial correlation; "
        "(3) acf_lag1_90 = rolling lag-1 autocorrelation of log-returns (level signal). "
        "Per-ticker proxy for the cross-domain signal-processing / econophysics "
        "'correlation time' concept (HRV / DSP). Inherently a per-ticker measure; "
        "cross-sectional ranking is left to the framework."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_autocorr_decay_first_zero_90",
        "xdom_autocorr_decay_rate_90",
        "xdom_autocorr_decay_lag1_90",
    ],
    "tags": ["autocorrelation", "mean-reversion", "momentum", "cross-domain", "dsp", "econophysics"],
    "version": "1.0.0",
    "author": (
        "Spec: Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Authors/Citation: Autocorrelation decay rate / first-zero lag (correlation time)."
    ),
}

# Maximum lag for first-zero search and for decay-rate fit
_MAX_LAG = 20
_DECAY_LAGS = 5   # lags 1..5 used for log-linear slope
_WINDOW = 90      # rolling window in trading days


def _acf_stats(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Given a 1-D array of returns (length == _WINDOW, no NaN guaranteed by caller),
    compute:
      - lag-1 autocorrelation
      - first zero-crossing lag (1.._MAX_LAG), or NaN if none found
      - log-linear decay slope over lags 1.._DECAY_LAGS
    Returns (lag1_acf, first_zero_lag, decay_slope).
    """
    n = len(returns)
    if n < _MAX_LAG + 2:
        return np.nan, np.nan, np.nan

    mean = returns.mean()
    demeaned = returns - mean
    var = np.dot(demeaned, demeaned)
    if var == 0.0:
        return np.nan, np.nan, np.nan

    # Compute ACF for lags 1.._MAX_LAG using dot-product (O(n*max_lag) but
    # max_lag=20 and n=90 so at most 1800 mults — fast enough vectorised).
    lags = np.arange(1, _MAX_LAG + 1)
    acf_vals = np.empty(len(lags))
    for i, k in enumerate(lags):
        acf_vals[i] = np.dot(demeaned[k:], demeaned[:-k]) / var

    lag1_acf = float(acf_vals[0])

    # First zero crossing: smallest lag where ACF <= 0
    first_zero = np.nan
    for i in range(len(acf_vals)):
        if acf_vals[i] <= 0.0:
            first_zero = float(lags[i])
            break

    # Log-linear decay slope: OLS of log|ACF| on lags 1.._DECAY_LAGS
    decay_vals = acf_vals[:_DECAY_LAGS]
    abs_vals = np.abs(decay_vals)
    # Require at least 3 non-zero values for a meaningful slope
    valid = abs_vals > 1e-10
    if valid.sum() < 3:
        decay_slope = np.nan
    else:
        x = lags[:_DECAY_LAGS][valid].astype(float)
        y = np.log(abs_vals[valid])
        # OLS slope: (sum(x*y) - n*xbar*ybar) / (sum(x^2) - n*xbar^2)
        xbar = x.mean()
        ybar = y.mean()
        num = np.dot(x - xbar, y - ybar)
        den = np.dot(x - xbar, x - xbar)
        decay_slope = float(num / den) if den != 0.0 else np.nan

    return lag1_acf, first_zero, decay_slope


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Initialise output columns with NaN
    col_zero = "xdom_autocorr_decay_first_zero_90"
    col_rate = "xdom_autocorr_decay_rate_90"
    col_lag1 = "xdom_autocorr_decay_lag1_90"

    df[col_zero] = np.nan
    df[col_rate] = np.nan
    df[col_lag1] = np.nan

    if n < _WINDOW + 1:
        return df

    # Log-returns (shift(1) is safe — only uses past data)
    log_ret = np.log(df["Close"].values / np.where(
        df["Close"].shift(1).values == 0, np.nan, df["Close"].shift(1).values
    ))
    # log_ret[0] will be NaN due to shift; that's fine

    first_zero_out = np.full(n, np.nan)
    decay_rate_out = np.full(n, np.nan)
    lag1_out = np.full(n, np.nan)

    # Rolling window: index i is the LAST bar of each window.
    # We need _WINDOW returns ending at i → rows [i-_WINDOW+1 .. i].
    # log_ret[0] is NaN (shift), so the earliest usable window starts at i=_WINDOW
    # (which has log_ret[1.._WINDOW] — _WINDOW valid values).
    for i in range(_WINDOW, n):
        window = log_ret[i - _WINDOW + 1 : i + 1]   # length _WINDOW
        # Skip if too many NaNs
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < _WINDOW // 2:
            continue
        # Use only valid (non-NaN) entries for ACF; fill with valid subset
        w = window[valid_mask]
        if len(w) < _MAX_LAG + 2:
            continue

        lag1, fz, slope = _acf_stats(w)
        first_zero_out[i] = fz
        decay_rate_out[i] = slope
        lag1_out[i] = lag1

    df[col_zero] = first_zero_out
    df[col_rate] = decay_rate_out
    df[col_lag1] = lag1_out

    return df
