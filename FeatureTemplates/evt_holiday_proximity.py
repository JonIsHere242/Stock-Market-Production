"""
evt_holiday_proximity.py — Proximity to US market holidays (event-time features).

THEME: calendar / event-time derived purely from the Date column.

The "holiday effect" is a long-documented anomaly: equity returns are abnormally
positive on the trading day(s) immediately BEFORE a market closure, and behave
differently on the first session AFTER. This block encodes, for every bar, its
proximity to the set of US equity-market holidays — distinct from turn-of-month
and weekday seasonality.

We compute the US market holiday dates DETERMINISTICALLY (no external data) from
simple calendar rules covering the years spanned by the data (2023–2027). Holidays
included: New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day,
Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas. Observed-date
shifting (Sat→Fri, Sun→Mon) is applied for the fixed-date holidays.

Proximity is then measured against the df's OWN realized trading calendar: the
pre-/post-holiday flags fire on the trading bar that is adjacent to a holiday in
calendar time. Signed-distance is in calendar days, clipped and normalized. All
deterministic and lookahead-safe (each row depends only on the fixed holiday set
and its own Date / immediate neighbors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "evt_holiday_proximity",
    "description": (
        "Proximity of each trading bar to US equity-market holidays: pre-holiday "
        "and post-holiday flags plus a normalized signed calendar-day distance to "
        "the nearest holiday."
    ),
    "requires":    ["Date"],
    "produces":    [
        "evt_pre_holiday",          # 1 if the NEXT trading bar is on/after a holiday gap (last session before closure)
        "evt_post_holiday",         # 1 if the PREVIOUS trading bar was before a holiday gap (first session after closure)
        "evt_days_to_holiday",      # calendar days to the nearest UPCOMING holiday, normalized [0,1] (capped 15d)
        "evt_days_since_holiday",   # calendar days since the nearest PAST holiday, normalized [0,1] (capped 15d)
        "evt_holiday_adjacent",     # 1 if within +/-3 calendar days of any holiday
    ],
    "tags":    ["calendar", "event_time", "seasonality", "holiday"],
    "version": "1.0",
    "author":  "feature-gen",
}

_CAP_DAYS = 15.0   # cap/normalization horizon for signed distances


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """nth (1-based) `weekday` (Mon=0..Sun=6) of month/year. n<0 counts from end."""
    if n > 0:
        first = pd.Timestamp(year=year, month=month, day=1)
        offset = (weekday - first.dayofweek) % 7
        return first + pd.Timedelta(days=offset + 7 * (n - 1))
    # last occurrence (n == -1)
    last = (pd.Timestamp(year=year, month=month, day=1)
            + pd.offsets.MonthEnd(0))
    offset = (last.dayofweek - weekday) % 7
    return last - pd.Timedelta(days=offset)


def _observed(d: pd.Timestamp) -> pd.Timestamp:
    """Apply US observed-date shift: Sat→prior Fri, Sun→following Mon."""
    if d.dayofweek == 5:      # Saturday
        return d - pd.Timedelta(days=1)
    if d.dayofweek == 6:      # Sunday
        return d + pd.Timedelta(days=1)
    return d


def _easter(year: int) -> pd.Timestamp:
    """Anonymous Gregorian (Meeus/Jones/Butcher) algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _holidays_for(years) -> "pd.DatetimeIndex":
    out = []
    for y in years:
        # Fixed-date holidays (with observed shifting).
        out.append(_observed(pd.Timestamp(y, 1, 1)))    # New Year's Day
        out.append(_observed(pd.Timestamp(y, 6, 19)))   # Juneteenth
        out.append(_observed(pd.Timestamp(y, 7, 4)))    # Independence Day
        out.append(_observed(pd.Timestamp(y, 12, 25)))  # Christmas
        # Floating-weekday holidays.
        out.append(_nth_weekday(y, 1, 0, 3))            # MLK Day: 3rd Mon Jan
        out.append(_nth_weekday(y, 2, 0, 3))            # Presidents' Day: 3rd Mon Feb
        out.append(_nth_weekday(y, 5, 0, -1))           # Memorial Day: last Mon May
        out.append(_nth_weekday(y, 9, 0, 1))            # Labor Day: 1st Mon Sep
        out.append(_nth_weekday(y, 11, 3, 4))           # Thanksgiving: 4th Thu Nov
        out.append(_easter(y) - pd.Timedelta(days=2))   # Good Friday
    return pd.DatetimeIndex(sorted(set(out)))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)

    years = range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2)
    hol = _holidays_for(years)
    hol_vals = hol.values.astype("datetime64[D]")
    d_vals = dates.values.astype("datetime64[D]")

    # Signed calendar-day distance from each bar to every holiday.
    # diff[i, j] = holiday_j - date_i  (in days)
    diff = (hol_vals[None, :] - d_vals[:, None]) / np.timedelta64(1, "D")

    # Nearest upcoming holiday (diff >= 0): min of non-negative diffs.
    up = np.where(diff >= 0, diff, np.inf)
    days_to = up.min(axis=1)
    days_to = np.where(np.isfinite(days_to), days_to, _CAP_DAYS)

    # Nearest past holiday (diff <= 0): min |diff| of non-positive diffs.
    past = np.where(diff <= 0, -diff, np.inf)
    days_since = past.min(axis=1)
    days_since = np.where(np.isfinite(days_since), days_since, _CAP_DAYS)

    df["evt_days_to_holiday"]    = np.clip(days_to, 0, _CAP_DAYS) / _CAP_DAYS
    df["evt_days_since_holiday"] = np.clip(days_since, 0, _CAP_DAYS) / _CAP_DAYS

    # Within +/-3 calendar days of any holiday.
    nearest = np.minimum(days_to, days_since)
    df["evt_holiday_adjacent"] = (nearest <= 3).astype(float)

    # --- Pre / post-holiday using the df's own trading gaps ------------------
    # A "holiday gap" is a place where consecutive trading bars straddle one or
    # more holidays (the closure). Pre-holiday = last bar before such a gap;
    # post-holiday = first bar after it.
    next_date = dates.shift(-1)
    prev_date = dates.shift(1)

    pre = np.zeros(len(dates), dtype=float)
    post = np.zeros(len(dates), dtype=float)
    # Build the membership set as integer day-counts so type-mismatch can't sneak
    # in (datetime64[D] viewed as int64 == calendar days since epoch).
    hol_int = set(hol_vals.astype("int64").tolist())

    def _gap_has_holiday(a, b) -> bool:
        # any holiday strictly between calendar dates a (exclusive) and b (exclusive)?
        if pd.isna(a) or pd.isna(b):
            return False
        a_i = int(np.datetime64(pd.Timestamp(a), "D").astype("int64"))
        b_i = int(np.datetime64(pd.Timestamp(b), "D").astype("int64"))
        # scan the calendar days strictly inside (a, b)
        for cur in range(a_i + 1, b_i):
            if cur in hol_int:
                return True
        return False

    for i in range(len(dates)):
        if _gap_has_holiday(dates.iloc[i], next_date.iloc[i] if i < len(dates) - 1 else pd.NaT):
            pre[i] = 1.0
        if _gap_has_holiday(prev_date.iloc[i] if i > 0 else pd.NaT, dates.iloc[i]):
            post[i] = 1.0

    df["evt_pre_holiday"]  = pre
    df["evt_post_holiday"] = post

    return df
