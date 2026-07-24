"""
Feature block: ff0703i_ohlc_pos_strength_ewma_acceleration

VEIN: ohlc_pos

Per-bar net closing strength NS = (2*Close - High - Low) / (High - Low),
range [-1, +1] (NaN when the bar has zero range). We track two EWMAs of NS
(fast span=5, slow span=40, adjust=False) and derive:

  - ff0703i_ewma_ns_mom   : fast_ewma - slow_ewma  (closing-strength momentum:
                            are recent closes stronger than the slow baseline?)
  - ff0703i_ewma_ns_slow  : the slow_ewma itself (smoothed baseline level)
  - ff0703i_ewma_ns_accel : slow_ewma_t - slow_ewma_{t-5} (5-day slope /
                            acceleration of the slow closing-strength trend)

This is an EWMA momentum/acceleration construction on where price closes
within its own daily range -- distinct from a windowed mean or an OLS trend
line, and intended to capture regime shifts in intrabar closing strength.
Fully causal: EWMA (adjust=False) only uses current & past bars, and the
5-bar slope uses a simple positive-lag diff (no lookahead).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "ff0703i_ohlc_pos_strength_ewma_acceleration",
    "description": (
        "EWMA (fast span=5 vs slow span=40) of per-bar net closing strength "
        "NS=(2*Close-High-Low)/(High-Low): momentum = fast-slow, plus the "
        "slow EWMA level, plus its 5-bar slope (acceleration). Faithful "
        "per-ticker OHLC-only implementation of the specified method."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff0703i_ewma_ns_mom",
        "ff0703i_ewma_ns_slow",
        "ff0703i_ewma_ns_accel",
    ],
    "tags": ["ohlc_pos", "ewma", "momentum", "acceleration"],
    "version": "1.0",
    "author": "feature-factory (auto-generated, faithful OHLC-only implementation)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_mom = "ff0703i_ewma_ns_mom"
    col_slow = "ff0703i_ewma_ns_slow"
    col_accel = "ff0703i_ewma_ns_accel"

    df[col_mom] = np.nan
    df[col_slow] = np.nan
    df[col_accel] = np.nan

    n = len(df)
    if n == 0:
        return df

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    rng = high - low
    rng_safe = rng.where(rng > 0, np.nan)

    ns = (2.0 * close - high - low) / rng_safe

    fast_ewma = ns.ewm(span=5, adjust=False, min_periods=1).mean()
    slow_ewma = ns.ewm(span=40, adjust=False, min_periods=1).mean()

    df[col_mom] = fast_ewma - slow_ewma
    df[col_slow] = slow_ewma
    df[col_accel] = slow_ewma - slow_ewma.shift(5)

    return df
