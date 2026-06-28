"""
_cand_osap_momseasonshort.py

Return seasonality (last year) — Heston & Sadka (2008).

Signal: average monthly return for this stock in the SAME calendar month
        over the prior year(s).  Per Heston-Sadka, stocks that performed
        well in January tend to do well the following January, etc.

This is a pure per-ticker time-series calculation from OHLCV:
  1. Compute each day's 1-month forward-looking... wait, per-ticker
     we compute PAST same-month returns (no leakage) and average them.

Implementation:
  - For each trading day t we identify its calendar month M.
  - We collect all past observations whose calendar month == M,
    using the last-day-of-month Close-to-Close return for that month.
  - osap_momseasonshort_1y  : same-month return 1 year ago (single obs).
  - osap_momseasonshort_avg : average same-month return over up to 5
                               prior years (the Heston-Sadka mean estimator).
  - osap_momseasonshort_trend: OLS slope of same-month returns over
                               years (is the seasonal effect strengthening?).

All values are assigned to the LAST trading day of each month so the
signal is available for trading on the first day of the next month.
Within-month days carry the signal forward (ffill) so the framework
sees a dense series.  This is leakage-free: only past same-month
returns enter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momseasonshort",
    "description": (
        "Return seasonality (same calendar month, prior year) per Heston & Sadka (2008). "
        "For each stock/day, computes: (1) the same-month return exactly 1 year ago, "
        "(2) the average same-month return over up to 5 prior years, and "
        "(3) the OLS trend slope of those same-month returns (is the pattern strengthening?). "
        "Signal is assigned to the last trading day of each month then forward-filled "
        "within the month — leakage-free, pure per-ticker OHLCV proxy of the "
        "cross-sectional Heston-Sadka seasonality factor."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momseasonshort_1y",
        "osap_momseasonshort_avg",
        "osap_momseasonshort_trend",
    ],
    "tags": ["momentum", "seasonality", "return", "price"],
    "version": "1.0",
    "author": "Heston & Sadka (2008) via OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 0. Initialise output columns with NaN
    # ------------------------------------------------------------------
    df = df.copy()
    df["osap_momseasonshort_1y"] = np.nan
    df["osap_momseasonshort_avg"] = np.nan
    df["osap_momseasonshort_trend"] = np.nan

    if len(df) < 2:
        return df

    # Ensure Date is datetime for month/year arithmetic
    dates = pd.to_datetime(df["Date"])
    close = df["Close"].values
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Build a month-end return series
    #    month_end[i] = (Close[last_day_of_month] / Close[last_day_of_prev_month]) - 1
    #    stored at the last-trading-day index of that month.
    # ------------------------------------------------------------------
    year_arr = dates.year.values
    month_arr = dates.month.values

    # Find the index of the last trading day in each (year, month) group
    # We iterate through sorted rows (df is ascending by Date per spec).
    # Build a dict: (year, month) -> last_row_index
    ym_last_idx: dict = {}
    for i in range(n):
        ym = (int(year_arr[i]), int(month_arr[i]))
        ym_last_idx[ym] = i  # overwrite → keeps the last index per group

    # Build sorted list of (year, month) keys for sequential processing
    ym_keys = sorted(ym_last_idx.keys())

    # month_end_ret[(year, month)] = monthly close-to-close return
    # previous month-end close → current month-end close
    month_end_ret: dict = {}
    prev_close = None
    for ym in ym_keys:
        idx = ym_last_idx[ym]
        cur_close = close[idx]
        if prev_close is not None and prev_close > 0:
            month_end_ret[ym] = cur_close / prev_close - 1.0
        prev_close = cur_close

    # ------------------------------------------------------------------
    # 2. For each month-end, look back at same-month returns in prior years
    #    (up to 5 years back, minimum 1 year back)
    # ------------------------------------------------------------------
    MAX_YEARS = 5

    # We'll accumulate values at the last-trading-day row index
    val_1y = {}
    val_avg = {}
    val_trend = {}

    for ym in ym_keys:
        yr, mo = ym
        # Gather same-month returns from prior years (yr-1 down to yr-MAX_YEARS)
        past_rets = []
        for lag in range(1, MAX_YEARS + 1):
            past_ym = (yr - lag, mo)
            if past_ym in month_end_ret:
                past_rets.append((lag, month_end_ret[past_ym]))

        if not past_rets:
            continue

        idx = ym_last_idx[ym]

        # 1-year-ago value (lag==1)
        for lag, ret in past_rets:
            if lag == 1:
                val_1y[idx] = ret
                break

        # Average across all available prior-year same months
        rets_only = [r for _, r in past_rets]
        val_avg[idx] = float(np.mean(rets_only))

        # OLS trend slope (x = lag in years: 1,2,3...; y = return)
        if len(past_rets) >= 2:
            lags = np.array([l for l, _ in past_rets], dtype=float)
            rets_arr = np.array([r for _, r in past_rets], dtype=float)
            x = lags - lags.mean()
            ss = np.dot(x, x)
            if ss > 0:
                slope = np.dot(x, rets_arr - rets_arr.mean()) / ss
                val_trend[idx] = float(slope)

    # ------------------------------------------------------------------
    # 3. Write to df at month-end rows, then forward-fill within month
    # ------------------------------------------------------------------
    for idx, v in val_1y.items():
        df.at[df.index[idx], "osap_momseasonshort_1y"] = v
    for idx, v in val_avg.items():
        df.at[df.index[idx], "osap_momseasonshort_avg"] = v
    for idx, v in val_trend.items():
        df.at[df.index[idx], "osap_momseasonshort_trend"] = v

    # Forward-fill so every trading day carries the most recent signal
    df["osap_momseasonshort_1y"] = df["osap_momseasonshort_1y"].ffill()
    df["osap_momseasonshort_avg"] = df["osap_momseasonshort_avg"].ffill()
    df["osap_momseasonshort_trend"] = df["osap_momseasonshort_trend"].ffill()

    # Guard: replace any inf that might have slipped through
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
