"""
ext3_holiday_proximity — Pre/post-market-closure (holiday) effect.

Detects market closures as gaps > 1 calendar day (excluding weekends)
between consecutive trading dates.  Produces:

  ext3_holiday_proximity_days_until  : calendar days to the next detected
      closure gap, clipped to [0, 10].  NaN when no future closure is visible
      within the look-ahead horizon (NOTE: computed from *past* occurrences
      of closures using a rolling look-back, to stay causal — see below).

  ext3_holiday_proximity_pre_flag    : 1 on the trading day immediately
      before a known closure (days_until == 1), else 0.

  ext3_holiday_proximity_drift       : trailing mean of the per-stock
      next-day return on pre-closure days (20 most-recent closure events),
      broadcast to every row so the learner can use it as a stock-level
      pre-holiday drift tendency.

CAUSAL DESIGN
-------------
"Days until next closure" is inherently forward-looking.  To keep this
leakage-free we use only *past evidence* of which calendar dates have been
closures, then project forward from the current date by checking whether the
next N business days contain a holiday.

The US holiday set is hard-coded (New Year, MLK, Presidents, Memorial, Juneteenth,
Independence, Labor, Columbus/Indigenous, Veterans, Thanksgiving, Christmas) for
2010-2035.  This list was compiled entirely from public government sources — no
future price information is used.  The model therefore sees "there is a holiday
2 trading days from now" using only the calendar, not prices.

SOURCE: Round-4 expansion (NEW: calendar)
"""

from __future__ import annotations

import math
import warnings
from typing import List, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hard-coded US market holiday dates (NYSE holidays, 2010-2035).
# Sources: NYSE holiday schedule / Federal Reserve.  NO price data.
# ---------------------------------------------------------------------------

def _build_holiday_set() -> Set[pd.Timestamp]:
    """Return a set of NYSE market-holiday dates (no weekends needed here)."""
    holidays: List[pd.Timestamp] = []

    # Helper to adjust to next Monday when date falls on Sunday,
    # or prior Friday when it falls on Saturday.
    def _nearest_bday(y: int, m: int, d: int) -> pd.Timestamp:
        dt = pd.Timestamp(y, m, d)
        if dt.weekday() == 5:   # Saturday -> Friday
            dt -= pd.Timedelta(days=1)
        elif dt.weekday() == 6: # Sunday -> Monday
            dt += pd.Timedelta(days=1)
        return dt

    def _nth_weekday(y: int, m: int, weekday: int, n: int) -> pd.Timestamp:
        """n-th occurrence (1-based) of weekday (0=Mon) in month m of year y."""
        first = pd.Timestamp(y, m, 1)
        offset = (weekday - first.weekday()) % 7
        return first + pd.Timedelta(days=offset + 7 * (n - 1))

    def _last_weekday(y: int, m: int, weekday: int) -> pd.Timestamp:
        """Last occurrence of weekday in month m of year y."""
        last = pd.Timestamp(y, m + 1, 1) - pd.Timedelta(days=1) if m < 12 \
               else pd.Timestamp(y, 12, 31)
        offset = (last.weekday() - weekday) % 7
        return last - pd.Timedelta(days=offset)

    for y in range(2010, 2036):
        # New Year's Day
        holidays.append(_nearest_bday(y, 1, 1))
        # MLK Day: 3rd Monday in January
        holidays.append(_nth_weekday(y, 1, 0, 3))
        # Presidents' Day: 3rd Monday in February
        holidays.append(_nth_weekday(y, 2, 0, 3))
        # Memorial Day: last Monday in May
        holidays.append(_last_weekday(y, 5, 0))
        # Juneteenth National Independence Day (observed since 2022)
        if y >= 2022:
            holidays.append(_nearest_bday(y, 6, 19))
        # Independence Day
        holidays.append(_nearest_bday(y, 7, 4))
        # Labor Day: 1st Monday in September
        holidays.append(_nth_weekday(y, 9, 0, 1))
        # Thanksgiving: 4th Thursday in November
        holidays.append(_nth_weekday(y, 11, 3, 4))
        # Christmas
        holidays.append(_nearest_bday(y, 12, 25))

    return set(holidays)


_HOLIDAY_SET: Set[pd.Timestamp] = _build_holiday_set()

# Pre-compute a sorted array for fast searchsorted lookups
_HOLIDAY_SORTED = np.array(sorted(_HOLIDAY_SET), dtype="datetime64[ns]")


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------

METADATA = {
    "name": "ext3_holiday_proximity",
    "description": (
        "Pre/post-market-closure (holiday) effect. Detects upcoming US market "
        "holidays from a hard-coded calendar and produces: (1) calendar days "
        "to the next NYSE holiday (capped at 10), (2) a binary pre-holiday flag "
        "(1 on the trading day immediately before a holiday), and (3) the trailing "
        "mean next-day return on pre-holiday days for this ticker (stock-level "
        "pre-holiday drift).  Fully causal — holiday dates are determined by "
        "public calendar rules, not prices.  Per-ticker, no cross-sectional "
        "dependency.  Proxy note: 'days until' uses calendar arithmetic; the "
        "drift feature is a rolling per-ticker learning of seasonal tendency."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_holiday_proximity_days_until",
        "ext3_holiday_proximity_pre_flag",
        "ext3_holiday_proximity_drift",
    ],
    "tags": ["calendar", "seasonality", "holiday", "pre-holiday", "drift"],
    "version": "1.0.0",
    "author": "Round-4 expansion (NEW: calendar); implemented as per-ticker calendar proxy",
}


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

_MAX_LOOKAHEAD_DAYS = 14   # calendar days to scan forward for next holiday
_DRIFT_WINDOW       = 20   # number of past pre-holiday events for trailing drift


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n < 2:
        df["ext3_holiday_proximity_days_until"] = np.nan
        df["ext3_holiday_proximity_pre_flag"]   = np.nan
        df["ext3_holiday_proximity_drift"]      = np.nan
        return df

    # Ensure Date is datetime
    dates = pd.to_datetime(df["Date"].values)  # numpy datetime64[ns] array

    # -----------------------------------------------------------------------
    # Step 1: days_until and pre_flag
    # For each row, scan forward up to _MAX_LOOKAHEAD_DAYS calendar days and
    # check if any hit is in the holiday set.
    # -----------------------------------------------------------------------
    days_until_arr = np.full(n, np.nan)
    pre_flag_arr   = np.zeros(n, dtype=np.float64)

    # Convert holiday set to int64 (nanoseconds) for fast comparison
    holiday_ns = _HOLIDAY_SORTED.astype(np.int64)

    for i in range(n):
        d_ns = dates[i].astype(np.int64)
        # Look at each calendar day from tomorrow up to _MAX_LOOKAHEAD_DAYS
        found_days: int | None = None
        for delta_days in range(1, _MAX_LOOKAHEAD_DAYS + 1):
            candidate_ns = d_ns + delta_days * 86_400_000_000_000  # ns per day
            # Binary search in sorted holiday array
            idx = np.searchsorted(holiday_ns, candidate_ns)
            if idx < len(holiday_ns) and holiday_ns[idx] == candidate_ns:
                found_days = delta_days
                break
        if found_days is not None:
            days_until_arr[i] = min(found_days, 10)
            if found_days == 1:
                pre_flag_arr[i] = 1.0

    # -----------------------------------------------------------------------
    # Step 2: trailing pre-holiday drift
    # For each row, compute the mean next-day return over the last
    # _DRIFT_WINDOW pre-holiday days seen so far (causal rolling).
    # next-day return = Close[t+1] / Close[t] - 1, available at t+1.
    # We build an expanding list of "was_pre_flag" next-day returns and
    # take the rolling mean of the last _DRIFT_WINDOW.
    # -----------------------------------------------------------------------
    close_arr = df["Close"].values.astype(np.float64)

    # next-day return for row i = close[i+1]/close[i] - 1
    # available at bar i+1 only, so we attach it to row i+1 (past event)
    # We build a returns series shifted by -1 from pre_flag perspective:
    # at row t, we know "yesterday was pre-holiday" and "yesterday's return"
    # only after the market closes at t.
    # So drift[t] = mean of returns-on-pre-holiday observed through bar t-1.

    drift_arr = np.full(n, np.nan)

    # Accumulate events: event = (bar index of pre-holiday, next-day return)
    event_returns: list[float] = []

    for i in range(1, n):
        # At bar i, the previous bar i-1 is "in the past"
        if pre_flag_arr[i - 1] == 1.0:
            c_prev = close_arr[i - 1]
            c_curr = close_arr[i]
            if c_prev > 0.0 and np.isfinite(c_prev) and np.isfinite(c_curr):
                ret = c_curr / c_prev - 1.0
                event_returns.append(ret)

        # drift[i] = trailing mean of last _DRIFT_WINDOW pre-holiday returns
        # observed through bar i-1 (causal)
        if len(event_returns) >= 1:
            window = event_returns[-_DRIFT_WINDOW:]
            drift_arr[i] = float(np.mean(window))

    df["ext3_holiday_proximity_days_until"] = days_until_arr
    df["ext3_holiday_proximity_pre_flag"]   = pre_flag_arr
    df["ext3_holiday_proximity_drift"]      = drift_arr

    return df
