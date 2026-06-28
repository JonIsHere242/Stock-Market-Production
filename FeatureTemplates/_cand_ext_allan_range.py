"""
ext_allan_range: Allan deviation of the daily-range volatility process.

The parent feature (xdom_allan_variance) applied Allan variance to price-return
series. This block applies it to the Parkinson daily-range proxy
  r_t = ln(High / Low)
which characterises the intraday volatility process. Allan deviation at tau=5
over an 80-day rolling window measures the *stability* of that volatility
process (vol-of-vol regime). The ratio to rolling std(r_t) normalises for
level, capturing regime transitions rather than level.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name": "ext_allan_range",
    "description": (
        "Allan deviation of the Parkinson daily-range series r_t=ln(H/L) at "
        "tau=5 over an 80-bar rolling window (vol-of-vol regime stability). "
        "Produces: (1) ext_allan_range_adev — rolling Allan deviation level; "
        "(2) ext_allan_range_ratio — Allan deviation divided by rolling std of "
        "r_t (normalised vol-of-vol); (3) ext_allan_range_slope — 20-bar "
        "z-score of the Allan deviation (trend / acceleration signal). "
        "Per-ticker proxy; cross-sectional ranking not needed here."
    ),
    "requires": ["High", "Low"],
    "produces": [
        "ext_allan_range_adev",
        "ext_allan_range_ratio",
        "ext_allan_range_slope",
    ],
    "tags": ["volatility", "allan_deviation", "vol_of_vol", "range", "parkinson"],
    "version": "1.0.0",
    "author": "spec ext_allan_range — extension of gate-validated winner xdom_allan_variance",
}


def _rolling_allan_dev(series: np.ndarray, tau: int, window: int) -> np.ndarray:
    """
    Compute rolling Allan deviation at lag tau over a rolling window.

    Allan variance at averaging time tau is:
        AVAR(tau) = 1 / (2*(N-1)) * sum_{i=1}^{N-1} (y[i] - y[i-1])^2
    where y[i] are non-overlapping means of the series over blocks of size tau.

    For rolling efficiency we use sliding_window_view with the full window.
    Returns an array of length len(series), NaN for the first (window-1) entries.
    """
    n = len(series)
    out = np.full(n, np.nan)

    if n < window:
        return out

    # We need enough blocks of size tau within each rolling window.
    # Number of complete tau-blocks inside one window of size `window`:
    n_blocks = window // tau
    if n_blocks < 2:
        return out

    # Build windows: shape (n - window + 1, window)
    wins = sliding_window_view(series, window_shape=window)  # (n-window+1, window)

    for wi in range(wins.shape[0]):
        w = wins[wi]
        # split into n_blocks non-overlapping blocks of size tau
        blocks = w[: n_blocks * tau].reshape(n_blocks, tau)
        block_means = blocks.mean(axis=1)
        diffs = np.diff(block_means)
        avar = 0.5 * np.mean(diffs ** 2)
        out[window - 1 + wi] = np.sqrt(max(avar, 0.0))

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    TAU = 5       # averaging time in bars
    WIN = 80      # rolling window length
    SLOPE_WIN = 20  # window for z-scoring the adev series

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    # Parkinson range: ln(H/L), guard against bad data
    with np.errstate(divide="ignore", invalid="ignore"):
        hl_ratio = np.where((low > 0) & (high > 0) & (high >= low), high / low, np.nan)
        r = np.log(hl_ratio)

    # Replace inf / -inf with nan
    r = np.where(np.isfinite(r), r, np.nan)

    # 1) Rolling Allan deviation of r_t
    adev = _rolling_allan_dev(r, tau=TAU, window=WIN)

    # 2) Rolling std of r_t (same window) for normalisation
    r_series = pd.Series(r)
    rolling_std = r_series.rolling(WIN, min_periods=WIN).std().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rolling_std > 0, adev / rolling_std, np.nan)

    # 3) Z-score of adev over SLOPE_WIN to capture trend / acceleration
    adev_series = pd.Series(adev)
    adev_roll_mean = adev_series.rolling(SLOPE_WIN, min_periods=SLOPE_WIN).mean().to_numpy()
    adev_roll_std = adev_series.rolling(SLOPE_WIN, min_periods=SLOPE_WIN).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(adev_roll_std > 0, (adev - adev_roll_mean) / adev_roll_std, np.nan)

    df["ext_allan_range_adev"] = adev
    df["ext_allan_range_ratio"] = ratio
    df["ext_allan_range_slope"] = slope

    return df
