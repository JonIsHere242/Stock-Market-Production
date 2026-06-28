"""
Petrosian Fractal Dimension (fast waveform-complexity proxy).

Spec: xdom_petrosian_fd
Source: Cross-domain method transfer (signal processing / econophysics / HRV / DSP)
Citation: Petrosian 1995 – "A new fractal dimension measure for classification of EEG signals"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_petrosian_fd",
    "description": (
        "Rolling Petrosian Fractal Dimension (PFD) of log-returns on a per-ticker basis. "
        "PFD = log10(n) / (log10(n) + log10(n / (n + 0.4*Nd))) "
        "where Nd is the number of sign changes in the first difference of returns over the window. "
        "A cheap waveform-complexity index: high PFD = oscillatory/noisy; low PFD = trending/smooth. "
        "Produces: (1) xdom_petrosian_fd_60  – 60-day PFD level; "
        "(2) xdom_petrosian_fd_20  – shorter 20-day PFD for recency; "
        "(3) xdom_petrosian_fd_slope – z-scored change (60d PFD minus 20d PFD, normalised by rolling std) "
        "capturing whether complexity is rising or falling."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_petrosian_fd_60",
        "xdom_petrosian_fd_20",
        "xdom_petrosian_fd_slope",
    ],
    "tags": ["complexity", "fractal", "signal_processing", "cross_domain", "returns"],
    "version": "1.0.0",
    "author": "Petrosian (1995); cross-domain transfer from HRV/EEG signal processing",
}


def _petrosian_fd(arr: np.ndarray) -> float:
    """
    Compute Petrosian FD for a 1-D array of values.
    FD = log10(n) / (log10(n) + log10(n / (n + 0.4 * Nd)))
    where Nd = number of sign changes in first-difference of arr.
    Returns NaN if n < 2 or denominator is zero.
    """
    n = len(arr)
    if n < 2:
        return np.nan
    diff = np.diff(arr)
    # sign changes: consecutive elements of diff with opposite signs
    # ignore zeros (treat zero as continuing the prior direction)
    signs = np.sign(diff)
    # drop zeros to avoid counting pause→move as a sign-change artefact
    signs_nonzero = signs[signs != 0]
    if len(signs_nonzero) < 2:
        nd = 0
    else:
        nd = int(np.sum(signs_nonzero[1:] != signs_nonzero[:-1]))

    log_n = np.log10(n)
    denom_inner = n + 0.4 * nd
    if denom_inner <= 0:
        return np.nan
    log_ratio = np.log10(n / denom_inner)
    denominator = log_n + log_ratio
    if denominator == 0:
        return np.nan
    return log_n / denominator


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- log returns (no lookahead: shift(1) uses only past prices) ---
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    n = len(df)

    # Pre-allocate output arrays
    fd_60 = np.full(n, np.nan)
    fd_20 = np.full(n, np.nan)

    ret_vals = log_ret.to_numpy()

    WIN60 = 60
    WIN20 = 20

    # Vectorised rolling using numpy sliding_window_view to avoid O(n^2) Python loops
    # We still need a Python loop over windows but it is O(n) with fixed C-speed inner work.
    for i in range(WIN60 - 1, n):
        window = ret_vals[i - WIN60 + 1: i + 1]
        # skip if too many NaNs (e.g. at start of series)
        if np.sum(np.isnan(window)) > WIN60 // 2:
            continue
        # drop leading NaNs inside window
        w = window[~np.isnan(window)]
        fd_60[i] = _petrosian_fd(w)

    for i in range(WIN20 - 1, n):
        window = ret_vals[i - WIN20 + 1: i + 1]
        if np.sum(np.isnan(window)) > WIN20 // 2:
            continue
        w = window[~np.isnan(window)]
        fd_20[i] = _petrosian_fd(w)

    df["xdom_petrosian_fd_60"] = fd_60
    df["xdom_petrosian_fd_20"] = fd_20

    # Slope: difference between 60d and 20d PFD, normalised by rolling std of 60d PFD
    # Positive slope → complexity rising (regime becoming more noisy)
    fd_60_series = pd.Series(fd_60, index=df.index)
    fd_20_series = pd.Series(fd_20, index=df.index)

    raw_slope = fd_60_series - fd_20_series
    rolling_std = fd_60_series.rolling(window=WIN60, min_periods=WIN60 // 2).std()

    with np.errstate(invalid="ignore", divide="ignore"):
        slope_z = np.where(
            rolling_std.to_numpy() > 0,
            raw_slope.to_numpy() / rolling_std.to_numpy(),
            np.nan,
        )

    # Replace inf with nan
    slope_z = np.where(np.isfinite(slope_z), slope_z, np.nan)
    df["xdom_petrosian_fd_slope"] = slope_z

    return df
