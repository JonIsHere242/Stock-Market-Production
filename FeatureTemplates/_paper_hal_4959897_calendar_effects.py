"""
_paper_hal_4959897_calendar_effects.py  --  Calendar / seasonal effect features.

Inspired by HAL-4959897: "Trading volumes in stock markets: Forecasts, Trading
Strategies, Market Impact, Equity Premium, and Monetary Policy", which studies
intraday volume forecasts, VWAP market impact, and the equity risk premium on
FOMC announcement days and across the cross-section.

HONEST PROXY: the paper's core empirical findings rest on well-documented
seasonal structure in volume and equity returns — day-of-week, turn-of-month,
and month-of-year effects. All columns here are derived purely from the calendar
Date of each row. No future information is used: the turn-of-month and
quarter-end flags rely solely on calendar day-of-month thresholds, so they are
bit-identical when the frame is truncated at any past date.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_hal_4959897_calendar_effects",
    "description": (
        "Seasonal / calendar effect features: cyclical day-of-week and month "
        "encodings, normalised day-of-month, turn-of-month flag, and "
        "quarter-end flag — all derived from the row's own Date."
    ),
    "requires":    ["Date"],
    "produces":    [
        "cal_dow_sin",
        "cal_dow_cos",
        "cal_month_sin",
        "cal_month_cos",
        "cal_day_of_month",
        "cal_turn_of_month",
        "cal_is_quarter_end",
    ],
    "tags":        ["calendar", "seasonal", "volume", "market_regime"],
    "version":     "1.0",
    "author":      "paper proxy: HAL-4959897 calendar/seasonal effects",
}

# ---------------------------------------------------------------------------
# Helpers (module-level constants)
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * np.pi

# Turn-of-month: last ~2 calendar days of the month (day >= 26) or first ~3
# calendar days of the next month (day <= 3). Pure calendar threshold — no
# look-ahead possible.
_TOM_HIGH = 26   # day-of-month >= this => end of month window
_TOM_LOW  = 3    # day-of-month <= this => start of month window

# Quarter-end: month in {3, 6, 9, 12} AND day >= 25 (last ~6 calendar days).
# Pure calendar threshold — bit-identical under any frame truncation.
_QE_MONTHS = frozenset({3, 6, 9, 12})
_QE_DAY    = 25


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seven calendar-derived feature columns to df.

    All arithmetic is performed on integer arrays extracted from
    pd.to_datetime(df['Date']).  No rolling window is needed — every
    value depends only on the calendar date of that row.

    Column definitions
    ------------------
    cal_dow_sin / cal_dow_cos
        Cyclical (sin/cos) encoding of weekday (Monday=0 … Friday=4) over a
        period-5 cycle.  Preserves the circular distance between weekdays so
        that Monday and Friday are neighbours across the week boundary.

    cal_month_sin / cal_month_cos
        Cyclical encoding of calendar month (1…12) over a period-12 cycle.
        Captures the circular distance between months (December and January
        are neighbours).

    cal_day_of_month
        Day-of-month normalised to [1/31, 1.0] by dividing by 31.  A simple
        linear proxy for position within the month.

    cal_turn_of_month
        1.0 if the row falls in the turn-of-month window defined purely by
        calendar day-of-month:  day >= 26  OR  day <= 3.  Else 0.0.
        Bit-identical under any frame truncation because only the row's own
        Date is used.

    cal_is_quarter_end
        1.0 if month in {3, 6, 9, 12} AND day >= 25.  Else 0.0.
        Same causality guarantee as cal_turn_of_month.
    """

    dates = pd.to_datetime(df["Date"])

    # Extract integer date components as numpy arrays (fast, no object overhead)
    dow   = dates.dt.dayofweek.to_numpy(dtype=np.float64)   # 0=Mon … 4=Fri
    month = dates.dt.month.to_numpy(dtype=np.float64)        # 1..12
    dom   = dates.dt.day.to_numpy(dtype=np.float64)          # 1..31

    # -- Cyclical encodings ---------------------------------------------------
    df["cal_dow_sin"] = np.sin(_TWO_PI * dow / 5.0)
    df["cal_dow_cos"] = np.cos(_TWO_PI * dow / 5.0)

    df["cal_month_sin"] = np.sin(_TWO_PI * (month - 1.0) / 12.0)
    df["cal_month_cos"] = np.cos(_TWO_PI * (month - 1.0) / 12.0)

    # -- Day-of-month normalised ----------------------------------------------
    df["cal_day_of_month"] = dom / 31.0

    # -- Turn-of-month flag (calendar-threshold, zero look-ahead) -------------
    tom = ((dom >= _TOM_HIGH) | (dom <= _TOM_LOW)).astype(np.float64)
    df["cal_turn_of_month"] = tom

    # -- Quarter-end flag (calendar-threshold, zero look-ahead) ---------------
    month_int = dates.dt.month.to_numpy(dtype=np.int32)
    qe_month  = np.isin(month_int, list(_QE_MONTHS))
    qe        = (qe_month & (dom >= _QE_DAY)).astype(np.float64)
    df["cal_is_quarter_end"] = qe

    return df
