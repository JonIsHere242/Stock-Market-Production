"""
Allan deviation (tau=5) of log-returns over an 80-day rolling window.

Allan deviation is a frequency-stability metric borrowed from oscillator physics.
At tau=5 it captures medium-frequency drift instability: high values mean the
5-day mean return is non-stationary (trending/reversing noisily); low values
indicate a stable drift regime.

Per-ticker proxy (cross-sectional ranking not needed -- the economic signal is
about intra-ticker frequency stability).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06280034d_freq_stability_allan_ret_80",
    "description": (
        "Allan deviation at tau=5 over an 80d rolling window of daily log-returns. "
        "Non-overlapping 5-bar blocks; block means computed; "
        "sigma_A = sqrt(0.5 * mean(diff(block_means)^2)). "
        "Produces the level, its ratio to rolling std of returns, and a 20d z-score. "
        "Per-ticker causal proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034d_freq_stability_allan_ret_80_level",
        "ff06280034d_freq_stability_allan_ret_80_ratio",
        "ff06280034d_freq_stability_allan_ret_80_zscore",
    ],
    "tags": ["frequency_stability", "allan_deviation", "volatility", "regime"],
    "version": "1.0.0",
    "author": "feature-factory",
}

_TAU = 5       # block size in bars
_WIN = 80      # rolling window in bars (must be a multiple of TAU for clean blocks; 80/5=16 blocks)
_ZWIN = 20     # z-score lookback


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN on every code path
    col_level  = "ff06280034d_freq_stability_allan_ret_80_level"
    col_ratio  = "ff06280034d_freq_stability_allan_ret_80_ratio"
    col_zscore = "ff06280034d_freq_stability_allan_ret_80_zscore"
    df[col_level]  = np.nan
    df[col_ratio]  = np.nan
    df[col_zscore] = np.nan

    n = len(df)
    if n < _WIN + 1:
        return df

    # Log-returns (no lookahead: ret[i] uses Close[i] and Close[i-1])
    close = df["Close"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(close[:-1] > 0, np.log(close[1:] / close[:-1]), np.nan)
    # Prepend NaN so log_ret[i] corresponds to row i (ret[0] is always NaN)
    log_ret = np.concatenate([[np.nan], log_ret])

    # Rolling std of raw returns (for ratio denominator)
    ret_series = pd.Series(log_ret)
    rolling_std = ret_series.rolling(_WIN, min_periods=_WIN).std().to_numpy()

    # Allan deviation at tau=5 over 80-bar rolling window
    # For each bar t >= WIN-1, window = log_ret[t-WIN+1 : t+1] (80 bars)
    # Split into 80/5 = 16 non-overlapping blocks, compute block means,
    # then sigma_A = sqrt(0.5 * mean(diff(block_means)^2))
    n_blocks = _WIN // _TAU   # 16
    allan_level = np.full(n, np.nan)

    for t in range(_WIN - 1, n):
        window = log_ret[t - _WIN + 1 : t + 1]   # shape (80,)
        if np.any(np.isnan(window)):
            continue
        # Reshape into (n_blocks, TAU) and take row means
        blocks = window.reshape(n_blocks, _TAU)
        block_means = blocks.mean(axis=1)          # shape (16,)
        diffs = np.diff(block_means)               # shape (15,)
        denom = len(diffs)
        if denom == 0:
            continue
        var_a = 0.5 * np.mean(diffs ** 2)
        if var_a < 0:
            continue
        allan_level[t] = np.sqrt(var_a)

    # Ratio: allan_level / rolling_std (normalises by unconditional vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        allan_ratio = np.where(
            (rolling_std > 0) & ~np.isnan(rolling_std) & ~np.isnan(allan_level),
            allan_level / rolling_std,
            np.nan,
        )

    # Z-score of allan_level over 20-bar lookback
    level_series = pd.Series(allan_level)
    roll_mean = level_series.rolling(_ZWIN, min_periods=_ZWIN).mean()
    roll_sd   = level_series.rolling(_ZWIN, min_periods=_ZWIN).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(
            roll_sd.to_numpy() > 0,
            (allan_level - roll_mean.to_numpy()) / roll_sd.to_numpy(),
            np.nan,
        )

    # Replace inf/-inf with NaN
    allan_level  = np.where(np.isfinite(allan_level),  allan_level,  np.nan)
    allan_ratio  = np.where(np.isfinite(allan_ratio),  allan_ratio,  np.nan)
    zscore       = np.where(np.isfinite(zscore),        zscore,        np.nan)

    df[col_level]  = allan_level
    df[col_ratio]  = allan_ratio
    df[col_zscore] = zscore

    return df
