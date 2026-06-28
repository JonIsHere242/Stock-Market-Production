"""
osap_indmom — Industry Momentum (per-ticker proxy)

Source: Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?"
        Journal of Finance 54(4): 1249–1290.
        Replicated in Chen & Zimmermann Open-Source Asset Pricing (OSAP) library.

True construction: assign stocks to SIC2/FF49 industries; compute equal- or
value-weighted industry return over the prior 12m skipping last month (t-12 to
t-2); long top-decile industries, short bottom-decile industries cross-
sectionally.

Per-ticker proxy (used here because cross-sectional industry classification is
not available at the block level):
  - 12-month momentum skipping the most recent month (classic WML window:
    days -252 to -21) — captures the same 11-month prior-period return signal
    that industry momentum is built on. This is the dominant component of the
    factor even at the single-stock level.
  - A 3-month variant (63-day) for a shorter look-back.
  - A "momentum acceleration" slope: whether the long-run momentum is
    accelerating or decelerating over the last quarter (sign of d/dt of
    rolling 6m return).

These are faithful, lookahead-free, per-ticker proxies for the industry
momentum signal.  They will not reproduce the exact cross-sectional industry
ranking but capture the same economic mechanism: stocks in trending sectors
tend to keep trending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_indmom",
    "description": (
        "Industry momentum per-ticker proxy (Moskowitz & Grinblatt 1999 / OSAP). "
        "True factor requires cross-sectional SIC2 industry assignment not available "
        "at block level. Proxy: 11-month skip-1-month price momentum (osap_indmom_12_1), "
        "3-month skip-1-month momentum (osap_indmom_3_1), and momentum acceleration "
        "(osap_indmom_accel) = slope of rolling 6m return over last quarter. "
        "All lookahead-free, per-ticker."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_indmom_12_1",
        "osap_indmom_3_1",
        "osap_indmom_accel",
    ],
    "tags": ["momentum", "industry", "osap", "price", "cross_sectional_proxy"],
    "version": "1.0.0",
    "author": "Moskowitz & Grinblatt (1999), OSAP (Chen & Zimmermann); per-ticker proxy impl.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # ------------------------------------------------------------------ #
    # 1. 12-1 momentum: return from 252 days ago to 21 days ago (skip 1m) #
    # ------------------------------------------------------------------ #
    # price 252 trading days ago
    close_252 = close.shift(252)
    # price 21 trading days ago (end of the skip-month)
    close_21 = close.shift(21)

    mom_12_1 = (close_21 / close_252.replace(0, np.nan)) - 1.0
    # guard inf
    mom_12_1 = mom_12_1.replace([np.inf, -np.inf], np.nan)
    df["osap_indmom_12_1"] = mom_12_1

    # ------------------------------------------------------------------ #
    # 2. 3-1 momentum: return from 63 days ago to 21 days ago             #
    # ------------------------------------------------------------------ #
    close_63 = close.shift(63)

    mom_3_1 = (close_21 / close_63.replace(0, np.nan)) - 1.0
    mom_3_1 = mom_3_1.replace([np.inf, -np.inf], np.nan)
    df["osap_indmom_3_1"] = mom_3_1

    # ------------------------------------------------------------------ #
    # 3. Momentum acceleration: slope of rolling 126-day (6m) return       #
    # over the last 63 trading days (1 quarter).                           #
    # accel > 0 => recent-past 6m momentum is improving.                  #
    # ------------------------------------------------------------------ #
    # rolling_6m[t] = Close[t-21] / Close[t-147] - 1   (skip 1m end)
    # We compute this series and then take the OLS slope over a 63-day window.
    # To keep this O(n) we use a rolling mean approach for the slope:
    #   slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # where x = integer index [0..62], y = rolling_6m values in that window.

    close_147 = close.shift(147)
    rolling_6m = (close_21 / close_147.replace(0, np.nan)) - 1.0
    rolling_6m = rolling_6m.replace([np.inf, -np.inf], np.nan)

    # Compute OLS slope of rolling_6m over last 63 bars via a rolling window.
    # We use the formula: slope = cov(x, y) / var(x)
    # With x = 0..n-1 this simplifies to a fixed x-variance and only requires
    # rolling mean(y) and rolling mean(x*y).
    n = 63
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()                     # (n-1)/2
    x_var = np.var(x, ddof=0)             # fixed for any window of length n

    y = rolling_6m.to_numpy(dtype=np.float64, na_value=np.nan)

    # rolling mean of y over n
    roll_y = rolling_6m.rolling(n, min_periods=max(n // 2, 10)).mean().to_numpy()

    # rolling mean of (i_within_window * y[i]) — we need sum(x_i * y_{t-n+1+i})
    # Build via convolution weights: weights[i] = i / n  (to get mean(x*y))
    # This is equivalent to a weighted rolling sum.
    # weights normalised so we get mean(x*y), not sum.
    w = x / n  # shape (n,), represents x_i / n  (to compute mean(x*y)/1)

    # Pad with NaN at the front to handle edge cases
    # Perform a 1D rolling dot via np.convolve equivalent using stride tricks
    # to stay O(n_rows * window_size) which is fine for n_rows~700, window=63.
    N = len(y)
    xy_means = np.full(N, np.nan)

    # Vectorised via stride approach
    from numpy.lib.stride_tricks import sliding_window_view

    # Only compute where we have enough data
    if N >= n:
        # sliding windows of y: shape (N-n+1, n)
        windows = sliding_window_view(y, window_shape=n)    # (M, n)
        # For each window, count valid entries
        valid_mask = ~np.isnan(windows)
        valid_count = valid_mask.sum(axis=1)                # (M,)

        # Compute mean(x * y) for each window, ignoring NaN
        # x is broadcast: shape (1, n)
        xy = x[np.newaxis, :] * np.where(valid_mask, windows, 0.0)  # (M, n)
        xy_sum = xy.sum(axis=1)                             # (M,)

        # mean(y) for each window (already have roll_y but recompute cleanly)
        y_sum = np.where(valid_mask, windows, 0.0).sum(axis=1)  # (M,)

        # Only use windows with at least min_periods valid values
        min_p = max(n // 2, 10)
        usable = valid_count >= min_p

        # cov(x, y) = mean(x*y) - mean(x)*mean(y)
        # var(x)    = x_var  (fixed)
        # We treat missing y as if replaced with 0 but only use usable windows
        with np.errstate(invalid="ignore", divide="ignore"):
            xy_mean_arr = np.where(usable, xy_sum / np.maximum(valid_count, 1), np.nan)
            y_mean_arr = np.where(usable, y_sum / np.maximum(valid_count, 1), np.nan)
            cov_xy = xy_mean_arr - x_mean * y_mean_arr
            slope_arr = cov_xy / x_var if x_var != 0 else np.full(len(cov_xy), np.nan)

        # Place into output array (first n-1 rows are NaN)
        accel = np.full(N, np.nan)
        accel[n - 1:] = slope_arr

        df["osap_indmom_accel"] = accel
    else:
        df["osap_indmom_accel"] = np.nan

    return df
