"""
osap_high52 — 52-week high proximity feature block.

Based on: George, T. J. & Hwang, C.-Y. (2004). "The 52-Week High and Momentum
Investing." Journal of Finance, 59(5), 2145-2176. Also replicated and catalogued
in Chen & Zimmermann (2022) Open Source Asset Pricing (OSAP) anomaly library
(signal ID: high52).

The 52-week high ratio = Close_t / max(High over prior 252 trading days).
George & Hwang find that stocks trading near their 52-week high tend to
outperform, as investors anchor on the high and are slow to revise upward,
creating underreaction momentum.

Per-ticker implementation (no cross-section needed — the ratio itself is the
predictive signal; XS ranking is an inference overlay done downstream).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_high52",
    "description": (
        "52-week high proximity (George & Hwang 2004 / OSAP high52). "
        "osap_high52_ratio = Close / rolling-252-bar max(High) — the anchoring "
        "momentum signal. osap_high52_dist_pct = signed distance from 52-week "
        "high as a percentage (0 = at high, negative = below). "
        "osap_high52_ratio_chg4w = 4-week change in the ratio, capturing "
        "whether the stock is approaching or retreating from its 52-week high. "
        "All computed per-ticker from OHLCV only; no cross-section required."
    ),
    "requires": ["High", "Close"],
    "produces": [
        "osap_high52_ratio",       # Close / 252-day rolling max(High); [0,1] anchor signal
        "osap_high52_dist_pct",    # (Close - max52) / max52 * 100; <= 0
        "osap_high52_ratio_chg4w", # change in ratio over trailing 21 trading days (~4 weeks)
    ],
    "tags": ["momentum", "anchoring", "52week", "osap", "george_hwang"],
    "version": "1.0.0",
    "author": "George & Hwang (2004) J. Finance 59(5); OSAP replication Chen & Zimmermann (2022)",
}

# 252 trading days ≈ 1 calendar year
_WINDOW_252 = 252
# 21 trading days ≈ 1 calendar month / 4 weeks
_WINDOW_21 = 21


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 52-week high proximity features for a single ticker time series.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.
        Must contain columns: High, Close.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    high = df["High"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    n = len(df)

    ratio = np.full(n, np.nan, dtype=np.float64)
    dist_pct = np.full(n, np.nan, dtype=np.float64)

    # Rolling 252-bar max of High (inclusive of current bar — no lookahead,
    # because max(High[t-251..t]) uses only current and past bars).
    # We need at least _WINDOW_252 bars for a valid estimate; rows before
    # that are left as NaN (expected leading-NaN behaviour).
    for i in range(_WINDOW_252 - 1, n):
        window_max = np.max(high[i - _WINDOW_252 + 1 : i + 1])
        if window_max > 0.0 and not np.isnan(window_max):
            r = close[i] / window_max
            ratio[i] = r
            dist_pct[i] = (close[i] - window_max) / window_max * 100.0
        # else leave NaN (zero-price or all-NaN window)

    df["osap_high52_ratio"] = ratio
    df["osap_high52_dist_pct"] = dist_pct

    # 4-week momentum in the ratio: ratio[t] - ratio[t-21]
    ratio_series = pd.Series(ratio, index=df.index)
    df["osap_high52_ratio_chg4w"] = ratio_series - ratio_series.shift(_WINDOW_21)

    return df
