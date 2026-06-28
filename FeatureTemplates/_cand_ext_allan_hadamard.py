"""
Hadamard deviation of returns (drift-insensitive frequency stability).

Extension of xdom_allan_variance: applies the 3-sample Hadamard variance
formula to log-returns, which is insensitive to linear frequency drift.
The Hadamard/Allan ratio separates flicker/random-walk noise from
deterministic drift that plain Allan variance conflates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_allan_hadamard",
    "description": (
        "Hadamard deviation of log-returns (drift-insensitive frequency stability). "
        "Uses 3-sample Hadamard variance H(tau)^2 = (1/6)*mean((ybar_{k+2} - "
        "2*ybar_{k+1} + ybar_k)^2) over tau=5 blocks in an 80-day rolling window. "
        "Hadamard variance is insensitive to linear frequency drift, so its ratio "
        "to Allan variance separates flicker/random-walk from deterministic drift. "
        "Per-ticker OHLCV proxy; no cross-sectional component. "
        "Produces: Hadamard deviation (level), Allan deviation (level), "
        "and Hadamard/Allan ratio (separates noise regimes)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_allan_hadamard_dev",
        "ext_allan_hadamard_allan_dev",
        "ext_allan_hadamard_ratio",
    ],
    "tags": ["volatility", "frequency_stability", "noise_decomposition", "returns"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner xdom_allan_variance; "
        "Hadamard variance formulation from time-frequency metrology literature "
        "(cf. Howe, Allan & Barnes, NIST, 1981)."
    ),
}

# Rolling window and tau parameters
_WINDOW = 80   # total rolling window in bars
_TAU = 5       # block size (bars per block)
# Number of full blocks in window: 80 // 5 = 16 blocks
# Allan uses 2-sample differences -> need >=2 blocks
# Hadamard uses 3-sample differences -> need >=3 blocks
_N_BLOCKS = _WINDOW // _TAU  # 16


def _allan_variance_from_blocks(block_means: np.ndarray) -> float:
    """Allan variance from array of consecutive tau-block means.
    AVAR(tau) = (1/2) * mean((ybar_{k+1} - ybar_k)^2)
    """
    if len(block_means) < 2:
        return np.nan
    diffs = np.diff(block_means)
    return 0.5 * np.mean(diffs ** 2)


def _hadamard_variance_from_blocks(block_means: np.ndarray) -> float:
    """Hadamard variance from array of consecutive tau-block means.
    HVAR(tau) = (1/6) * mean((ybar_{k+2} - 2*ybar_{k+1} + ybar_k)^2)
    """
    if len(block_means) < 3:
        return np.nan
    # Second differences of block means
    bm = block_means
    second_diff = bm[2:] - 2.0 * bm[1:-1] + bm[:-2]
    return (1.0 / 6.0) * np.mean(second_diff ** 2)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Hadamard deviation, Allan deviation, and their ratio."""
    n = len(df)

    hdev = np.full(n, np.nan)
    adev = np.full(n, np.nan)
    ratio = np.full(n, np.nan)

    if n < _WINDOW:
        df["ext_allan_hadamard_dev"] = np.nan
        df["ext_allan_hadamard_allan_dev"] = np.nan
        df["ext_allan_hadamard_ratio"] = np.nan
        return df

    # Log returns: shape (n,)
    close = df["Close"].to_numpy(dtype=np.float64)

    # Guard against non-positive close prices
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[1:] > 0) & (close[:-1] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # log_ret has length n-1; index i maps to return between bar i and bar i+1
    # We attach log_ret[i] to bar i+1 (current bar uses past close)
    # Pad with nan at the start so log_ret_full[i] = log return ending at bar i
    log_ret_full = np.empty(n, dtype=np.float64)
    log_ret_full[0] = np.nan
    log_ret_full[1:] = log_ret

    # For each bar t from _WINDOW-1 onward, take the last _WINDOW returns
    # and partition into _N_BLOCKS blocks of size _TAU
    # block means -> apply Allan and Hadamard formulas

    # Vectorized approach using stride tricks on a 2D array of blocks
    # We use a sliding approach: for each window ending at t,
    # split into N_BLOCKS x TAU, compute block means, then apply formulas.

    # Build a 2D matrix: rows = windows, cols = _WINDOW values
    # Only feasible because _WINDOW=80 is small relative to n (~700 rows)
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view shape: (n - _WINDOW + 1, _WINDOW)
    windows = sliding_window_view(log_ret_full, window_shape=_WINDOW)
    # windows[i] corresponds to bars [i .. i+_WINDOW-1], result stored at bar i+_WINDOW-1

    # Reshape each window into (_N_BLOCKS, _TAU) and compute block means
    # Shape: (num_windows, _N_BLOCKS, _TAU)
    w3d = windows.reshape(windows.shape[0], _N_BLOCKS, _TAU)

    # Block means: (num_windows, _N_BLOCKS)
    # Use nanmean to handle any nan log-returns gracefully
    block_means = np.nanmean(w3d, axis=2)

    # Allan variance: (1/2) * mean of squared first differences along blocks axis
    # First differences of block means: shape (num_windows, _N_BLOCKS - 1)
    first_diffs = np.diff(block_means, axis=1)
    avar = 0.5 * np.nanmean(first_diffs ** 2, axis=1)  # (num_windows,)

    # Hadamard variance: (1/6) * mean of squared second differences along blocks axis
    # Second differences: shape (num_windows, _N_BLOCKS - 2)
    bm = block_means
    second_diffs = bm[:, 2:] - 2.0 * bm[:, 1:-1] + bm[:, :-2]
    hvar = (1.0 / 6.0) * np.nanmean(second_diffs ** 2, axis=1)  # (num_windows,)

    # Deviations (square roots); guard against negative due to floating point
    with np.errstate(invalid="ignore"):
        adev_vals = np.sqrt(np.where(avar >= 0, avar, np.nan))
        hdev_vals = np.sqrt(np.where(hvar >= 0, hvar, np.nan))

    # Ratio H/A: values > 1 indicate drift-dominated noise; < 1 indicate flicker/white
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_vals = np.where(adev_vals > 0, hdev_vals / adev_vals, np.nan)

    # Store results at the last bar of each window
    start_idx = _WINDOW - 1
    num_windows = windows.shape[0]
    hdev[start_idx: start_idx + num_windows] = hdev_vals
    adev[start_idx: start_idx + num_windows] = adev_vals
    ratio[start_idx: start_idx + num_windows] = ratio_vals

    df["ext_allan_hadamard_dev"] = hdev
    df["ext_allan_hadamard_allan_dev"] = adev
    df["ext_allan_hadamard_ratio"] = ratio

    return df
