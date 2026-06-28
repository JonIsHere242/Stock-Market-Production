"""
Katz Fractal Dimension of the log-price waveform (Katz 1988).

Rolling 60-day Katz FD applied per-ticker to the log-close series.
Also produces a shorter 20-day window (responsiveness) and a slope/momentum
of the 60-day FD to capture complexity trends.

FD = log10(n) / (log10(n) + log10(d / L))
where:
  n = number of steps (window - 1)
  L = summed point-to-point path length (total variation of the waveform)
  d = max Euclidean distance from the first point to any other point

High FD -> rough/complex waveform; Low FD -> smoother/trending series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_katz_fd",
    "description": (
        "Rolling Katz fractal dimension (Katz 1988) of the log-close price waveform. "
        "FD = log10(n)/(log10(n)+log10(d/L)) where L is summed path length, "
        "d is max displacement from the first point, n is the number of steps. "
        "Per-ticker proxy; captures waveform roughness/complexity. "
        "High FD = rough/mean-reverting; Low FD = smooth/trending. "
        "Produces: 60-day FD level, 20-day FD level, and 10-day slope of the 60-day FD."
    ),
    "requires": ["Close"],
    "produces": ["xdom_katz_fd_60", "xdom_katz_fd_20", "xdom_katz_fd_60_slope"],
    "tags": ["fractal", "complexity", "waveform", "cross-domain", "price-structure"],
    "version": "1.0",
    "author": "Katz fractal dimension (Katz 1988, waveform complexity); cross-domain-method transfer spec",
}


def _katz_fd_series(log_prices: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Katz FD over a 1-D array of log prices.
    Returns array of same length; first (window-1) entries are NaN.
    """
    n_total = len(log_prices)
    result = np.full(n_total, np.nan, dtype=np.float64)

    n_steps = window - 1  # number of point-to-point steps; constant across windows
    if n_steps < 2:
        return result

    log_n = np.log10(n_steps)

    # Sliding window using stride tricks for speed (O(n*w) but vectorised)
    # shape: (n_windows, window)
    from numpy.lib.stride_tricks import sliding_window_view
    if n_total < window:
        return result

    windows = sliding_window_view(log_prices, window_shape=window)  # (n_windows, window)

    # L = sum of absolute first differences within each window
    diffs = np.abs(np.diff(windows, axis=1))          # (n_windows, window-1)
    L = diffs.sum(axis=1)                              # (n_windows,)

    # d = max Euclidean distance from first point in each window to any other
    # Since we are on a 1-D series (time-axis is evenly spaced by 1 day),
    # distance from point 0 to point k = sqrt(k^2 + (x_k - x_0)^2).
    # Using the Katz formulation in its original waveform context (amplitude-only distance
    # is also common; here we include the time axis for fidelity).
    x0 = windows[:, 0:1]                              # (n_windows, 1)
    delta_x = windows - x0                            # (n_windows, window)
    time_idx = np.arange(window, dtype=np.float64)    # [0, 1, ..., window-1]
    dists = np.sqrt(time_idx[np.newaxis, :] ** 2 + delta_x ** 2)  # (n_windows, window)
    d = dists.max(axis=1)                             # (n_windows,)

    # FD = log10(n) / (log10(n) + log10(d / L))
    # Guard: L == 0 (flat series) or d == 0 -> undefined
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((L > 0) & (d > 0), d / L, np.nan)
        log_ratio = np.log10(ratio)
        fd = np.where(
            np.isfinite(log_ratio) & (log_n + log_ratio != 0),
            log_n / (log_n + log_ratio),
            np.nan,
        )

    # Place results: window of size `window` covers indices [i, i+window-1]; result lands at i+window-1
    result[window - 1:] = fd

    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2 or "Close" not in df.columns:
        df["xdom_katz_fd_60"] = np.nan
        df["xdom_katz_fd_20"] = np.nan
        df["xdom_katz_fd_60_slope"] = np.nan
        return df

    log_close = np.log(df["Close"].values.astype(np.float64))

    # Replace any inf/-inf from log(<=0) with NaN
    log_close = np.where(np.isfinite(log_close), log_close, np.nan)

    # Interpolate interior NaNs linearly so windows are not shattered
    # (sparse NaNs only; leading NaNs stay NaN)
    series_lc = pd.Series(log_close)
    log_close_filled = series_lc.interpolate(method="linear", limit_direction="forward").values

    fd_60 = _katz_fd_series(log_close_filled, window=60)
    fd_20 = _katz_fd_series(log_close_filled, window=20)

    df["xdom_katz_fd_60"] = fd_60
    df["xdom_katz_fd_20"] = fd_20

    # 10-day slope of the 60-day FD (captures trend in complexity)
    fd_60_series = pd.Series(fd_60, index=df.index)
    df["xdom_katz_fd_60_slope"] = fd_60_series.diff(10)

    return df
