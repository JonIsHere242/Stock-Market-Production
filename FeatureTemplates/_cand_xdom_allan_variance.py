"""
Allan Variance / Deviation feature block.

Allan variance is a frequency-stability metric originally developed for atomic
clocks and adapted here from signal processing / econophysics.  For a
tau-bar averaging interval, the Allan variance is:
    sigma_A(tau)^2 = 0.5 * mean( (ybar_{k+1} - ybar_k)^2 )
where ybar_k is the mean of tau consecutive log-returns in block k.

Rolling 80-day window with tau=5: at each date we compute 80 days of
log-returns, split into 80//5=16 consecutive non-overlapping blocks of 5,
compute each block mean, then compute the Allan variance as 0.5 * mean of
squared first-differences of adjacent block means.

Allan deviation is the square root.  It distinguishes white noise
(sigma_A flat across tau) from random-walk drift (sigma_A grows with tau)
and flicker noise, providing a richer noise-regime characterisation than
plain rolling variance.

A 20-day rolling z-score of the Allan deviation is also produced to capture
the rate of change / regime shift.
"""

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_allan_variance",
    "description": (
        "Rolling Allan deviation of log-returns (tau=5 blocks, 80-day window). "
        "Adapted from atomic-clock frequency-stability analysis (Allan variance). "
        "sigma_A(tau)^2 = 0.5 * mean((ybar_{k+1}-ybar_k)^2) over consecutive "
        "tau-length block means.  Distinguishes white noise from random-walk / "
        "flicker-noise drift better than plain rolling variance.  Per-ticker "
        "time-series implementation (inherently single-asset; no cross-sectional "
        "ranking needed -- the metric is already absolute)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_allan_variance_dev5_80",   # Allan deviation, tau=5, window=80
        "xdom_allan_variance_ratio",      # Allan-dev / rolling-std ratio (noise-regime indicator)
        "xdom_allan_variance_zscore",     # 20-day rolling z-score of Allan deviation
    ],
    "tags": ["cross-domain", "signal-processing", "volatility", "noise-regime", "econophysics"],
    "version": "1.0",
    "author": (
        "Cross-domain method transfer -- Allan variance / deviation "
        "(frequency-stability metric from atomic clocks). "
        "Source: xdom_allan_variance spec; econophysics / HRV / DSP literature."
    ),
}

# ── constants ────────────────────────────────────────────────────────────────
_TAU   = 5   # block length (days)
_WIN   = 80  # rolling window (must be divisible by tau; 80/5=16 blocks)
_ZSCORE_WIN = 20  # window for z-score of Allan dev


def _allan_dev_from_returns(ret_window: np.ndarray, tau: int) -> float:
    """
    Compute Allan deviation for a 1-D array of returns using tau-length blocks.

    Returns NaN if there are fewer than 2*tau valid values.
    """
    n = len(ret_window)
    n_blocks = n // tau
    if n_blocks < 2:
        return np.nan

    # trim to exact multiple of tau
    ret_window = ret_window[: n_blocks * tau]

    # block means
    block_means = ret_window.reshape(n_blocks, tau).mean(axis=1)

    # Allan variance = 0.5 * mean of squared first-differences of block means
    diffs = np.diff(block_means)
    allan_var = 0.5 * np.mean(diffs ** 2)
    return float(np.sqrt(allan_var)) if allan_var >= 0 else np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Allan deviation features per ticker (single-stock call)."""

    # ── log-returns ──────────────────────────────────────────────────────────
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # safe log-return; zero or negative closes → nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (close[:-1] > 0) & (close[1:] > 0),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )
    # align with df index: first bar has no return
    log_ret_full = np.empty(n, dtype=np.float64)
    log_ret_full[0] = np.nan
    log_ret_full[1:] = log_ret

    # ── rolling Allan deviation (tau=5, win=80) ──────────────────────────────
    allan_dev = np.full(n, np.nan, dtype=np.float64)
    rolling_std = np.full(n, np.nan, dtype=np.float64)

    for i in range(_WIN - 1, n):
        window = log_ret_full[i - _WIN + 1 : i + 1]
        # drop nans inside window before computing
        valid = window[~np.isnan(window)]
        if len(valid) >= _TAU * 2:
            allan_dev[i] = _allan_dev_from_returns(valid[-_WIN:], _TAU)
            rolling_std[i] = float(np.nanstd(valid))

    # ── ratio: Allan-dev / rolling-std ───────────────────────────────────────
    # ratio < 1 => returns closer to white noise; > 1 => more random-walk-like
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rolling_std > 0, allan_dev / rolling_std, np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    # ── z-score of Allan deviation (20-day rolling) ──────────────────────────
    ad_series = pd.Series(allan_dev)
    roll_mean = ad_series.rolling(_ZSCORE_WIN, min_periods=_ZSCORE_WIN // 2).mean()
    roll_std  = ad_series.rolling(_ZSCORE_WIN, min_periods=_ZSCORE_WIN // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore_arr = np.where(
            roll_std.to_numpy() > 0,
            (ad_series.to_numpy() - roll_mean.to_numpy()) / roll_std.to_numpy(),
            np.nan,
        )
    zscore_arr = np.where(np.isfinite(zscore_arr), zscore_arr, np.nan)

    # ── assign columns ────────────────────────────────────────────────────────
    df["xdom_allan_variance_dev5_80"] = allan_dev
    df["xdom_allan_variance_ratio"]   = ratio
    df["xdom_allan_variance_zscore"]  = zscore_arr

    return df
