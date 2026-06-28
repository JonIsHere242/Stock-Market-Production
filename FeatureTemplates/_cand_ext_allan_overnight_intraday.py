"""
Allan deviation split: overnight vs intraday returns.

Allan deviation measures the frequency-domain instability of a time series at a
given averaging time tau. Here we split the daily return into two components:
  - overnight: ln(Open_t / Close_{t-1})   -- gap / news-driven
  - intraday:  ln(Close_t / Open_t)        -- session / order-flow driven

We compute the Allan deviation (tau=5, rolling 80-bar window) for each component
independently, and their ratio.  This is orthogonal to the parent xdom_allan_variance
feature (which operated on total daily returns) because it decomposes WHERE in the
day the random-walk-like instability originates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_allan_overnight_intraday",
    "description": (
        "Per-ticker Allan deviation (tau=5, rolling 80-bar window) computed "
        "separately on overnight log-returns ln(Open/prevClose) and intraday "
        "log-returns ln(Close/Open), plus their ratio. Captures which component "
        "(gap vs session) carries random-walk-like instability. Orthogonal to "
        "xdom_allan_variance which uses total daily returns. "
        "Per-ticker proxy -- cross-sectional ranking not performed here."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext_allan_overnight_intraday_ovn",    # Allan dev of overnight returns
        "ext_allan_overnight_intraday_intra",  # Allan dev of intraday returns
        "ext_allan_overnight_intraday_ratio",  # overnight Allan dev / intraday Allan dev
    ],
    "tags": ["volatility", "allan_variance", "overnight", "intraday", "return_decomposition"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner xdom_allan_variance (spec: ext_allan_overnight_intraday)",
}

# Allan deviation at tau=k is computed as:
#   ADEV(tau) = sqrt( 0.5 * mean( (y_{i+k} - y_{i})^2 ) )
# where y_i are the *phase* values (cumulative sum of the rate series).
# In a rolling window of length W we compute on the sub-series of length W.
# tau=5 means we average over 5-bar intervals.

TAU = 5
WIN = 80   # rolling window in bars


def _allan_dev_rolling(series: pd.Series, tau: int, window: int) -> pd.Series:
    """
    Rolling Allan deviation of a rate series at averaging time tau,
    using a rolling window of `window` bars.

    At each position t (enough history), takes the last `window` values of
    the rate series, cumulates them into a phase series, then computes:
        ADEV = sqrt( 0.5 * mean( (phase[tau:] - phase[:-tau])^2 ) )

    Returns a Series aligned to the input index, NaN where window is incomplete.
    """
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)

    # Minimum bars needed: window + 1 for the first diff
    # Actually we need `window` bars for phase, then tau differences
    # Phase has length `window`, differences have length window - tau
    min_bars = window + tau  # conservative; ensures window has tau pairs

    for t in range(min_bars - 1, n):
        # Extract last `window` rate values
        start = t - window + 1
        rates = arr[start : t + 1]  # shape (window,)

        # Build phase = cumulative sum (like Allan variance convention)
        phase = np.cumsum(rates)  # length window

        # Differences at lag tau
        diffs = phase[tau:] - phase[:-tau]  # length window - tau
        if len(diffs) < 2:
            continue

        adev_sq = 0.5 * np.mean(diffs ** 2)
        out[t] = np.sqrt(max(adev_sq, 0.0))

    return pd.Series(out, index=series.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Overnight log-return: ln(Open_t / Close_{t-1})
    prev_close = df["Close"].shift(1)
    # Guard against zero/negative prices
    safe_open = df["Open"].where(df["Open"] > 0, np.nan)
    safe_prev_close = prev_close.where(prev_close > 0, np.nan)
    safe_close = df["Close"].where(df["Close"] > 0, np.nan)

    r_overnight = np.log(safe_open / safe_prev_close)   # NaN on first bar
    r_intraday  = np.log(safe_close / safe_open)        # NaN where Open==0

    adev_ovn   = _allan_dev_rolling(r_overnight, TAU, WIN)
    adev_intra = _allan_dev_rolling(r_intraday,  TAU, WIN)

    # Ratio: overnight / intraday  (guard zero intraday)
    safe_intra = adev_intra.where(adev_intra > 0, np.nan)
    ratio = adev_ovn / safe_intra

    # Replace inf/-inf with NaN
    adev_ovn   = adev_ovn.replace([np.inf, -np.inf], np.nan)
    adev_intra = adev_intra.replace([np.inf, -np.inf], np.nan)
    ratio      = ratio.replace([np.inf, -np.inf], np.nan)

    df["ext_allan_overnight_intraday_ovn"]   = adev_ovn
    df["ext_allan_overnight_intraday_intra"] = adev_intra
    df["ext_allan_overnight_intraday_ratio"] = ratio

    return df
