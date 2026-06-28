"""
_cand_ext2_allan_acceleration.py

Acceleration of the Allan noise-regime (2nd-order Allan change).
Computes rolling Allan deviation (tau=5) over an 80-bar window, then
derives velocity (10d change) and acceleration (10d change-of-change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext2_allan_acceleration",
    "description": (
        "Per-ticker Allan deviation (tau=5, 80d rolling window): non-overlapping "
        "5-bar block means of log-returns, sigma_A = sqrt(0.5 * mean(diff(block_means)^2)). "
        "Produces: (1) the rolling Allan deviation level, (2) its 10d first difference "
        "(velocity -- how fast the noise regime is shifting), and (3) its 10d "
        "second difference (acceleration -- how fast the velocity is changing). "
        "Acceleration is a strictly orthogonal axis from the parent xdom_allan_variance "
        "level/z-score features. Cross-sectional ranks not available per-ticker; "
        "all three columns are in native Allan-deviation units."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_allan_acceleration_level",
        "ext2_allan_acceleration_vel",
        "ext2_allan_acceleration_accel",
    ],
    "tags": ["volatility", "noise-regime", "allan", "acceleration", "time-series"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of xdom_allan_variance winner vein",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling Allan deviation (tau=5, window=80) and its 1st/2nd
    10-day finite differences (velocity and acceleration).
    """
    tau = 5
    window = 80
    diff_lag = 10

    close = df["Close"].values.astype(np.float64)
    n = len(close)

    # Log returns (length n-1, padded with nan at index 0 to keep alignment)
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.where(
            close[:-1] > 0,
            np.log(close[1:] / close[:-1]),
            np.nan,
        )

    # Rolling Allan deviation over `window` bars with tau=5 block size.
    # For each ending index t we look at log_ret[t-window+1 : t+1] (window bars).
    # Split that into floor(window/tau) non-overlapping 5-bar blocks,
    # compute block means, then sigma_A = sqrt(0.5 * mean(diff(block_means)^2)).
    # We use a stride-trick for efficiency: no O(n^2) python loop over rows.

    n_blocks = window // tau  # = 16 for window=80, tau=5
    # We only produce valid values where we have at least `window` bars of log_ret.
    # log_ret starts from index 1, so first valid window ends at index window (0-based).

    allan_dev = np.full(n, np.nan, dtype=np.float64)

    # Build the rolling computation using sliding_window_view for speed.
    # We need log_ret[1:] (length n-1) windowed with width=window.
    lr = log_ret[1:]  # length n-1; index i here = original index i+1
    if len(lr) >= window:
        # shape (n-1-window+1, window) = (n-window, window)
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(lr, window_shape=window)
        # wins[i] covers original indices [i+1 .. i+window] inclusive
        # Output index in allan_dev = i + window (original df row)
        m = wins.shape[0]  # = n - window

        # Reshape each window into (n_blocks, tau) and take block means
        # wins shape: (m, window) -> (m, n_blocks, tau)
        blocks = wins[:, : n_blocks * tau].reshape(m, n_blocks, tau)
        block_means = blocks.mean(axis=2)  # (m, n_blocks)

        # diff of block means along block axis
        d = np.diff(block_means, axis=1)  # (m, n_blocks-1)
        d_sq = d ** 2

        # mean of squared diffs per window, guard against all-nan
        mean_d_sq = np.nanmean(d_sq, axis=1)  # (m,)

        sigma_a = np.sqrt(0.5 * mean_d_sq)  # (m,)

        # Place results: window i ends at original index i + window
        allan_dev[window: window + m] = sigma_a

    # 10-day velocity and acceleration via simple finite differences
    allan_vel = np.full(n, np.nan, dtype=np.float64)
    allan_accel = np.full(n, np.nan, dtype=np.float64)

    valid = ~np.isnan(allan_dev)
    if valid.sum() >= diff_lag + 1:
        # velocity: allan_dev[t] - allan_dev[t - diff_lag]
        av = allan_dev.copy()
        # shift by diff_lag
        shifted_vel = np.full(n, np.nan)
        shifted_vel[diff_lag:] = av[: n - diff_lag]
        vel = av - shifted_vel

        # acceleration: vel[t] - vel[t - diff_lag]
        shifted_accel = np.full(n, np.nan)
        shifted_accel[diff_lag:] = vel[: n - diff_lag]
        accel = vel - shifted_accel

        # Guard against inf/-inf (shouldn't arise but be safe)
        vel = np.where(np.isfinite(vel), vel, np.nan)
        accel = np.where(np.isfinite(accel), accel, np.nan)

        allan_vel = vel
        allan_accel = accel

    df["ext2_allan_acceleration_level"] = allan_dev
    df["ext2_allan_acceleration_vel"] = allan_vel
    df["ext2_allan_acceleration_accel"] = allan_accel

    return df
