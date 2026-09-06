"""
_cand_gap0903_rce_inside_outside.py - Inside and outside bar structure.

The oldest compression primitive in the price-pattern literature: an INSIDE bar
is a session whose entire range is contained by the prior session (High <= prev
High AND Low >= prev Low) - the market refused to take out either side, so the
range coiled. An OUTSIDE bar (High > prev High AND Low < prev Low) is the
opposite: both sides were taken out, an engulfing expansion. Neither construct
exists anywhere in the production panel, and unlike an ATR level they are
SCALE-FREE - they say something about the tape's structure rather than its
amplitude, so they survive across price and volatility regimes.

Columns
  rce_inside_freq_21   fraction of the last 21 bars that were inside bars.
                       High = a persistently coiling, range-contracting name.
  rce_outside_freq_21  fraction of the last 21 bars that were outside bars.
                       High = a whippy, two-sided, expansion-prone tape.
  rce_io_balance_63    (inside count - outside count) / (inside + outside) over
                       the trailing 63, in [-1, 1]. +1 all coil, -1 all
                       expansion. NaN when the denominator is zero (a 63-bar
                       stretch with neither structure, e.g. a pure trend or a
                       halted name) - never a divide by zero.
  rce_inside_streak    consecutive inside bars ending at t, capped at 10.
                       Two or three stacked inside bars is the textbook
                       pre-breakout coil.

Causality: classification uses only bar t and bar t-1 (a POSITIVE shift(1));
frequencies are trailing rolling means; the streak is accumulated FORWARD from
the series start with a cumsum/groupby-cumcount, never counted backward from
the last bar.

Edge cases: the first bar of the series has no predecessor and is classified as
neither inside nor outside. Zero-range bars (High == Low, halted or illiquid)
are handled naturally - such a bar is inside whenever the prior bar brackets it,
which is the economically correct reading, and no division touches a raw range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "_cand_gap0903_rce_inside_outside",
    "description": (
        "Inside/outside bar structure: 21-bar inside and outside bar frequencies, a "
        "63-bar inside-minus-outside balance in [-1,1] and the consecutive inside-bar "
        "streak. Scale-free compression vs expansion structure."
    ),
    "requires": ["High", "Low"],
    "produces": [
        "rce_inside_freq_21",
        "rce_outside_freq_21",
        "rce_io_balance_63",
        "rce_inside_streak",
    ],
    "tags": ["range", "compression", "candle", "breakout", "experimental"],
    "version": "1.0",
    "author": "feature-factory gap0903 (range compression and expansion theme)",
}

_PRODUCES = METADATA["produces"]
_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for col in _PRODUCES:
        df[col] = np.nan

    if len(df) == 0:
        return df

    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # first bar has no predecessor -> neither inside nor outside
    inside = ((high <= prev_high) & (low >= prev_low)).fillna(False)
    outside = ((high > prev_high) & (low < prev_low)).fillna(False)

    inside_f = inside.astype("float64")
    outside_f = outside.astype("float64")

    df["rce_inside_freq_21"] = inside_f.rolling(21, min_periods=21).mean().values
    df["rce_outside_freq_21"] = outside_f.rolling(21, min_periods=21).mean().values

    ins_63 = inside_f.rolling(63, min_periods=63).sum()
    out_63 = outside_f.rolling(63, min_periods=63).sum()
    denom = ins_63 + out_63
    denom = denom.where(denom > _EPS)          # 0 events in the window -> NaN
    balance = (ins_63 - out_63) / denom
    balance = balance.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    df["rce_io_balance_63"] = balance.values

    # consecutive inside bars, accumulated forward from the series start
    reset_grp = (~inside).cumsum()
    streak = inside.groupby(reset_grp).cumsum().astype("float64")
    df["rce_inside_streak"] = streak.clip(upper=10.0).values

    return df
