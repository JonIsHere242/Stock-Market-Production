"""
Earnings streak length (osap_numearnincrease)
Loh & Warachka 2012, via OpenSourceAP (Chen-Zimmermann).

Per-ticker proxy: counts consecutive quarterly periods of increasing
PIT net income (TTM), capped at 8, using SEC fundamentals as_of().
The original uses ibq (income before extraordinary items, quarterly);
we use net_income_ttm as the closest available PIT fundamental.
Cross-sectional rank is not reproduced here -- the raw streak length
is used directly as a per-ticker signal.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_numearnincrease",
    "description": (
        "Earnings streak length: number of consecutive 4-quarter periods of "
        "increasing net income (TTM), capped at 8. Per-ticker proxy for the "
        "cross-sectional ibq-based streak from Loh & Warachka 2012 "
        "(OpenSourceAP / Chen-Zimmermann). Positive predicted sign: longer "
        "streaks predict higher future returns. Also emits a 1-year change "
        "in streak length (momentum of the streak) and a rolling 252-day "
        "z-score of the streak level."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_numearnincrease_streak",
        "osap_numearnincrease_delta",
        "osap_numearnincrease_zscore",
    ],
    "tags": ["fundamentals", "earnings", "streak", "accounting", "osap"],
    "version": "1.0",
    "author": "Loh & Warachka (2012), OpenSourceAP Chen-Zimmermann; block by Claude",
}


# ---------------------------------------------------------------------------
# Helper: vectorised streak counter
# ---------------------------------------------------------------------------
def _consecutive_increase_streak(series: pd.Series, max_streak: int = 8) -> pd.Series:
    """
    For each position i, count the number of consecutive prior values that
    were strictly increasing up to max_streak.  Fully vectorised via
    cumsum-based trick.

    A "period" is defined by the quarterly snapshots already embedded in
    `series` (fund_net_income_ttm changes only at filing dates).
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    streak = np.zeros(n, dtype=float)

    for i in range(1, n):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            streak[i] = np.nan
        elif vals[i] > vals[i - 1]:
            streak[i] = min(streak[i - 1] + 1.0, float(max_streak))
        else:
            streak[i] = 0.0

    streak[0] = np.nan
    return pd.Series(streak, index=series.index)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT net income (TTM) -- backward-merged on filed_date
    df = _fundamentals.as_of(df, fields=["net_income_ttm"])

    ni = df["fund_net_income_ttm"].copy()

    # The fundamentals column changes only at quarterly filing dates;
    # forward-fill to daily so we work on a daily-indexed series, then
    # detect when the value actually changes (i.e., a new quarterly reading
    # arrived).  We compute the streak on those change points, then broadcast
    # back to daily.

    # Build the streak on the (already daily, ffilled by as_of) series.
    # Because the value is constant between filings, consecutive daily
    # comparisons within the same quarter are always "equal" and contribute 0.
    # We only want to step the counter at filing boundaries.  We detect
    # quarterly filing events as rows where fund_net_income_ttm differs from
    # the previous row (or is the first non-NaN).

    changed = ni.ne(ni.shift(1)) & ni.notna()

    # Snapshot series: NaN on non-change days, actual value on change days
    snapshot = ni.where(changed)

    # Streak computed only on snapshots
    snapshot_streak = _consecutive_increase_streak(snapshot.dropna(), max_streak=8)

    # Reindex back to full daily index, then forward-fill
    streak_daily = snapshot_streak.reindex(df.index).ffill()

    # On days before any filing, streak is NaN
    # Also NaN where fundamentals never arrived
    streak_daily = streak_daily.where(ni.notna())

    df["osap_numearnincrease_streak"] = streak_daily

    # Delta: change in streak over ~252 trading days (one year)
    df["osap_numearnincrease_delta"] = streak_daily - streak_daily.shift(252)

    # Z-score of streak level over a 252-day rolling window
    roll = streak_daily.rolling(252, min_periods=63)
    roll_mean = roll.mean()
    roll_std = roll.std(ddof=1)
    zscore = (streak_daily - roll_mean) / roll_std.replace(0, np.nan)
    df["osap_numearnincrease_zscore"] = zscore

    # Drop scratch fundamental column
    df.drop(columns=["fund_net_income_ttm"], inplace=True)

    return df
