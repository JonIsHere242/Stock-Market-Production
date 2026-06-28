"""
evt_turn_of_month.py — Turn-of-month and intra-month position event-time features.

THEME: calendar / event-time structure derived purely from the Date column.

Well-documented market microstructure anomaly: the "turn-of-month" (TOM) effect —
returns cluster in the last trading day of a month plus the first few of the next,
driven by index-fund / pension cash-flow reinvestment and window dressing. This
block also encodes WHERE in the month a bar sits (half-of-month, week-of-month,
day-of-month position) which is orthogonal to plain price momentum/volatility.

IMPORTANT — trading-day awareness:
  We do NOT have a market-calendar dependency, but the df is itself the realized
  trading calendar for this ticker (one row per trading day, ascending, no gaps).
  So "first/last trading day of the month" is computed by grouping the df's own
  Date column by (year, month) and ranking rows within each month. This is the
  true trading-day position, not the naive calendar-day position, and it is fully
  deterministic / lookahead-safe (each row only depends on its own month label).

All produced columns are bounded numeric flags / normalized positions in [0, 1]
(or small integer counts), never NaN for any row that has a valid Date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "evt_turn_of_month",
    "description": (
        "Turn-of-month window flags plus intra-month position (half-of-month, "
        "week-of-month, normalized trading-day position) computed from the df's "
        "own realized trading calendar."
    ),
    "requires":    ["Date"],
    "produces":    [
        "evt_tom_last1",          # 1 on the LAST trading day of the month
        "evt_tom_window",         # 1 on last-1 .. first-3 trading days of the turn (TOM window)
        "evt_first3_tdom",        # 1 on the first 3 trading days of the month
        "evt_tdom_pos_norm",      # normalized trading-day-of-month position in [0,1]
        "evt_tdom_from_end_norm", # normalized trading days remaining in month, [0,1]
        "evt_week_of_month",      # week-of-month index 1..6 (calendar weeks)
        "evt_half_of_month",      # 0 = first half of month, 1 = second half
    ],
    "tags":    ["calendar", "event_time", "seasonality"],
    "version": "1.0",
    "author":  "feature-gen",
}

_TOM_LAST_N  = 1   # last N trading days of a month count as the "turn"
_TOM_FIRST_N = 3   # first N trading days of next month count as the "turn"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])

    # Month label per row (period) → groups rows belonging to the same calendar month.
    ym = dates.dt.to_period("M")

    # Trading-day-of-month index (1-based) within each month, using the df's own
    # realized trading days. cumcount is order-preserving since df is ascending.
    tdom = ym.groupby(ym).cumcount() + 1                      # 1,2,3,... per month
    tdom = pd.Series(tdom.values, index=df.index)

    # Count of trading days in the month each row belongs to (broadcast back).
    month_size = ym.map(ym.value_counts())
    month_size = pd.Series(np.asarray(month_size, dtype=float), index=df.index)

    # Trading days remaining until month end (0 on the last trading day).
    tdom_from_end = month_size - tdom                         # 0-based from end

    # --- Turn-of-month flags -------------------------------------------------
    # Last trading day of the month.
    df["evt_tom_last1"] = (tdom_from_end < _TOM_LAST_N).astype(float)

    # TOM window: last _TOM_LAST_N trading days OR first _TOM_FIRST_N trading days.
    in_last  = tdom_from_end < _TOM_LAST_N
    in_first = tdom <= _TOM_FIRST_N
    df["evt_tom_window"] = (in_last | in_first).astype(float)

    # First 3 trading days of the month (January-style cash-inflow window).
    df["evt_first3_tdom"] = (tdom <= _TOM_FIRST_N).astype(float)

    # --- Normalized intra-month positions in [0,1] ---------------------------
    # Position from start: (tdom-1)/(size-1); single-day months → 0.
    denom = (month_size - 1.0).replace(0.0, np.nan)
    pos_norm = (tdom - 1.0) / denom
    df["evt_tdom_pos_norm"] = pos_norm.fillna(0.0).clip(0.0, 1.0)

    # Position from end: trading days remaining / (size-1).
    from_end_norm = tdom_from_end / denom
    df["evt_tdom_from_end_norm"] = from_end_norm.fillna(0.0).clip(0.0, 1.0)

    # --- Calendar-based intra-month bucketing --------------------------------
    # Week-of-month (1..6): ceil(calendar_day_of_month / 7).
    dom = dates.dt.day
    df["evt_week_of_month"] = np.ceil(dom / 7.0).astype(float)

    # Half-of-month: 0 for days 1..15, 1 for days 16..end.
    df["evt_half_of_month"] = (dom > 15).astype(float)

    return df
