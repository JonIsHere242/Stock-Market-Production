"""
Range-position reversal feature block.

Signal: where the close falls within its recent High-Low range predicts mean-reversion.
Closing near the top of a range -> expect pullback (negative expected return);
closing near the bottom -> expect bounce (positive expected return).

This is orthogonal to return-sum reversal (osap_streversal), which uses
cumulative past returns. Here the axis is RANGE POSITION (Williams %R family),
not net return level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_reversal_range_position",
    "description": (
        "Range-position reversal: measures where the close sits within its recent "
        "20-day High-Low price range (Williams %R / Stochastic-K concept). "
        "Extremeness is computed as the signed distance from the midpoint (0.5), "
        "negated so that a value near 1.0 (top of range) yields a NEGATIVE signal "
        "(expect pullback) and a value near 0.0 (bottom) yields a POSITIVE signal "
        "(expect bounce). "
        "Produces three columns: "
        "(1) ext_reversal_range_position_21d -- 21-day window signed reversal signal; "
        "(2) ext_reversal_range_position_5d  -- 5-day window, captures shorter-term extremes; "
        "(3) ext_reversal_range_position_diff -- 21d minus 5d, measures whether short-term "
        "position is more extreme than the longer-term context (momentum within range). "
        "Per-ticker proxy; cross-sectional ranking is an overlay applied elsewhere. "
        "Signal is orthogonal to return-sum reversal (osap_streversal family)."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ext_reversal_range_position_21d",
        "ext_reversal_range_position_5d",
        "ext_reversal_range_position_diff",
    ],
    "tags": ["reversal", "mean_reversion", "range_position", "williams_r", "stochastic"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner osap_streversal. "
        "Method: Williams %R / Stochastic-K range-position concept applied as "
        "a signed mean-reversion signal."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- 21-day window (uses rolling 20 past bars so window=21 gives 20 prior + current) ---
    # min/max over a 21-bar window (inclusive of current bar -- no lookahead)
    roll20_min = close.rolling(21, min_periods=10).min()
    roll20_max = close.rolling(21, min_periods=10).max()

    range20 = roll20_max - roll20_min
    # Guard divide-by-zero: flat price -> NaN
    range20_safe = range20.where(range20 > 0, np.nan)

    # Range position [0, 1]: 0 = bottom of range, 1 = top
    pos_21d = (close - roll20_min) / range20_safe

    # Signed reversal signal: negate (closed high -> negative -> expect pullback)
    # Centre at 0.5: (pos - 0.5) gives [-0.5, +0.5]; negate for reversal direction
    signal_21d = -(pos_21d - 0.5)

    # --- 5-day window ---
    roll5_min = close.rolling(5, min_periods=3).min()
    roll5_max = close.rolling(5, min_periods=3).max()

    range5 = roll5_max - roll5_min
    range5_safe = range5.where(range5 > 0, np.nan)

    pos_5d = (close - roll5_min) / range5_safe
    signal_5d = -(pos_5d - 0.5)

    # --- Diff: 21d signal minus 5d signal ---
    # Positive -> short-term more oversold than longer-term context (stronger bounce signal)
    # Negative -> short-term more overbought (stronger reversal signal)
    signal_diff = signal_21d - signal_5d

    df["ext_reversal_range_position_21d"] = signal_21d
    df["ext_reversal_range_position_5d"] = signal_5d
    df["ext_reversal_range_position_diff"] = signal_diff

    return df
