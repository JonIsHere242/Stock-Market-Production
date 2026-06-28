"""
ish_gap_dynamics.py — Opening-gap frequency, fill rate, magnitude and persistence.

A "gap" is an open that prints away from the prior close. Three behavioural facts
make gaps informative beyond raw momentum:

  1. FREQUENCY  — how often a name gaps (up/down) over a window proxies the rate at
     which closed-market information arrives in this stock.
  2. FILL RATE  — fraction of gaps that are "closed" the same session (price trades
     back through the prior close intraday). High fill rate => gaps are noise that
     mean-reverts; low fill rate => gaps stick and tend to be informative breaks.
  3. MAGNITUDE  — the standardized size of today's gap (z-score vs its own recent
     distribution) flags abnormal overnight repricing.

Gap fill detection uses the day's High/Low range:
  - an UP gap (Open > prevClose) is "filled" if Low <= prevClose that day.
  - a DOWN gap (Open < prevClose) is "filled" if High >= prevClose that day.

All ratios/fractions; clipped; leading rolling NaN expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "ish_gap_dynamics",
    "description": (
        "Opening-gap behaviour: up/down gap frequency, same-day gap-fill rate, "
        "gap-magnitude z-score and signed-gap persistence over 20/60d."
    ),
    "requires":    ["Open", "High", "Low", "Close"],
    "produces":    [
        "ish_gap_pct",
        "ish_gap_z_60",
        "ish_gap_up_freq_20",
        "ish_gap_dn_freq_20",
        "ish_gap_freq_60",
        "ish_gap_fill_rate_20",
        "ish_gap_fill_rate_60",
        "ish_gap_persist_20",
    ],
    "tags":    ["gap", "overnight", "mean_reversion"],
    "version": "1.0",
    "author": "feature-gen",
}

_EPS = 1e-9
# a bar only counts as a "gap" if it opens at least this far from prior close
_GAP_THRESHOLD = 0.005  # 0.5%


def compute(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["Open"].astype("float64")
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    close = df["Close"].astype("float64")
    prev_close = close.shift(1)

    # --- raw signed gap ----------------------------------------------------
    gap = open_ / prev_close.replace(0, np.nan) - 1.0
    gap = gap.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    df["ish_gap_pct"] = gap.values

    # --- gap magnitude z-score vs own 60d distribution ---------------------
    gap_mean_60 = gap.rolling(60, min_periods=30).mean()
    gap_std_60 = gap.rolling(60, min_periods=30).std()
    gap_z = (gap - gap_mean_60) / (gap_std_60 + _EPS)
    df["ish_gap_z_60"] = gap_z.replace([np.inf, -np.inf], np.nan).clip(-10, 10).values

    # --- classify each bar -------------------------------------------------
    is_up_gap = (gap > _GAP_THRESHOLD).astype("float64")
    is_dn_gap = (gap < -_GAP_THRESHOLD).astype("float64")
    is_any_gap = ((is_up_gap + is_dn_gap) > 0).astype("float64")
    # do not let the first row (NaN prev_close) count as a non-gap falsely
    is_up_gap[prev_close.isna()] = np.nan
    is_dn_gap[prev_close.isna()] = np.nan
    is_any_gap[prev_close.isna()] = np.nan

    df["ish_gap_up_freq_20"] = is_up_gap.rolling(20, min_periods=10).mean().values
    df["ish_gap_dn_freq_20"] = is_dn_gap.rolling(20, min_periods=10).mean().values
    df["ish_gap_freq_60"] = is_any_gap.rolling(60, min_periods=30).mean().values

    # --- gap fill detection ------------------------------------------------
    # filled if the day's range trades back through the prior close
    up_filled = (low <= prev_close)
    dn_filled = (high >= prev_close)
    filled = pd.Series(np.where(is_up_gap > 0, up_filled,
                       np.where(is_dn_gap > 0, dn_filled, np.nan)),
                       index=df.index, dtype="float64")
    # fill rate = (# filled gaps) / (# gaps) over the window
    filled_count_20 = filled.rolling(20, min_periods=1).sum()
    gap_count_20 = is_any_gap.rolling(20, min_periods=1).sum()
    fill_20 = filled_count_20 / (gap_count_20 + _EPS)
    # mask windows with too few gaps to be meaningful
    fill_20 = fill_20.where(gap_count_20 >= 3)
    df["ish_gap_fill_rate_20"] = fill_20.clip(0, 1).values

    filled_count_60 = filled.rolling(60, min_periods=1).sum()
    gap_count_60 = is_any_gap.rolling(60, min_periods=1).sum()
    fill_60 = filled_count_60 / (gap_count_60 + _EPS)
    fill_60 = fill_60.where(gap_count_60 >= 5)
    df["ish_gap_fill_rate_60"] = fill_60.clip(0, 1).values

    # --- gap persistence: net signed gap drift over 20d --------------------
    # positive => a string of up-gaps (sticky bullish repricing), neg => down-gaps
    gap_sign = np.sign(gap)
    df["ish_gap_persist_20"] = gap_sign.rolling(20, min_periods=10).mean().clip(-1, 1).values

    return df
