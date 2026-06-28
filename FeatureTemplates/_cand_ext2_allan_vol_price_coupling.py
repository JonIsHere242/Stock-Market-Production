"""
ext2_allan_vol_price_coupling
Activity-vs-price stability coupling via Allan deviation ratio.

Allan deviation (sigma_A) at tau=5 over an 80d rolling window:
  - Split the series into non-overlapping 5-bar blocks
  - Take block means
  - sigma_A = sqrt(0.5 * mean(diff(block_means)^2))
Compute for log-return AND first-differenced log-volume; produce their ratio
(volume Allan dev / return Allan dev) and its 20d change.

Per-ticker proxy: fully computable from OHLCV; no cross-sectional data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext2_allan_vol_price_coupling",
    "description": (
        "Allan deviation ratio: volume-process stability / price-process stability. "
        "Allan deviation at tau=5 over an 80d rolling window; ratio captures "
        "relative stability of trading activity vs price. Also produces 20d change "
        "in the ratio as a dynamic/slope variant. Per-ticker OHLCV proxy; "
        "no cross-sectional data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext2_allan_vol_price_coupling_ratio",
        "ext2_allan_vol_price_coupling_ratio_chg20",
        "ext2_allan_vol_price_coupling_vol_adev",
    ],
    "tags": ["volume", "price", "stability", "allan_variance", "coupling"],
    "version": "1.0",
    "author": "Round-3 deep exploration of a rich winner vein (xdom_allan_variance)",
}

# Parameters
_TAU = 5          # block size in bars
_WIN = 80         # rolling window in bars (must be >= 2*TAU so we get >=2 block means)
_CHG_LAG = 20     # lookback for slope/change


def _allan_dev_ratio_at(log_ret: np.ndarray, log_vol_diff: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Allan deviation at tau=5 for both series over the supplied window.
    Returns (ratio, vol_adev, ret_adev).  NaN when computation is impossible.
    """
    n = len(log_ret)
    n_blocks = n // _TAU
    if n_blocks < 2:
        return np.nan, np.nan, np.nan

    usable = n_blocks * _TAU

    # -- return Allan dev --
    ret_blocks = log_ret[:usable].reshape(n_blocks, _TAU)
    ret_means = ret_blocks.mean(axis=1)
    ret_diffs = np.diff(ret_means)
    ret_adev_sq = 0.5 * np.nanmean(ret_diffs ** 2)
    if ret_adev_sq <= 0.0 or not np.isfinite(ret_adev_sq):
        return np.nan, np.nan, np.nan
    ret_adev = np.sqrt(ret_adev_sq)

    # -- volume Allan dev --
    vol_blocks = log_vol_diff[:usable].reshape(n_blocks, _TAU)
    vol_means = vol_blocks.mean(axis=1)
    vol_diffs = np.diff(vol_means)
    vol_adev_sq = 0.5 * np.nanmean(vol_diffs ** 2)
    if vol_adev_sq < 0.0 or not np.isfinite(vol_adev_sq):
        return np.nan, np.nan, np.nan
    vol_adev = np.sqrt(vol_adev_sq)

    ratio = vol_adev / ret_adev if ret_adev > 0 else np.nan
    return ratio, vol_adev, ret_adev


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # Pre-allocate output arrays
    ratio_arr = np.full(n, np.nan)
    vol_adev_arr = np.full(n, np.nan)

    # Build input series
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Log returns (shift by 1 — past to present, no lookahead)
    # log_ret[i] = log(close[i] / close[i-1]), set 0th to nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.where(close > 0, np.log(close), np.nan)
        log_ret = np.empty(n)
        log_ret[0] = np.nan
        log_ret[1:] = log_close[1:] - log_close[:-1]

    # First-differenced log-volume
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.where(volume > 0, np.log(volume), np.nan)
        log_vol_diff = np.empty(n)
        log_vol_diff[0] = np.nan
        log_vol_diff[1:] = log_vol[1:] - log_vol[:-1]

    # Rolling computation: only need _WIN bars of history
    # First valid output index: we need _WIN bars (indices 0.._WIN-1) to compute at index _WIN-1
    start = _WIN - 1
    for i in range(start, n):
        ret_window = log_ret[i - _WIN + 1: i + 1]
        vol_window = log_vol_diff[i - _WIN + 1: i + 1]

        # Skip if too many NaNs (tolerate a few leading NaNs in the window)
        if np.sum(np.isfinite(ret_window)) < _WIN // 2:
            continue
        if np.sum(np.isfinite(vol_window)) < _WIN // 2:
            continue

        # Replace NaN with 0 for block-mean computation (conservative)
        ret_clean = np.where(np.isfinite(ret_window), ret_window, 0.0)
        vol_clean = np.where(np.isfinite(vol_window), vol_window, 0.0)

        r, v, _ = _allan_dev_ratio_at(ret_clean, vol_clean)
        ratio_arr[i] = r
        vol_adev_arr[i] = v

    df["ext2_allan_vol_price_coupling_ratio"] = ratio_arr
    df["ext2_allan_vol_price_coupling_vol_adev"] = vol_adev_arr

    # 20-day change in ratio (slope/dynamic variant)
    ratio_series = pd.Series(ratio_arr, index=df.index)
    df["ext2_allan_vol_price_coupling_ratio_chg20"] = ratio_series.diff(_CHG_LAG)

    return df
