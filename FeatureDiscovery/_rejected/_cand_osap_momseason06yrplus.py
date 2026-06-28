"""
Return Seasonality: Years 6–10 (Heston & Sadka 2008)

Per-ticker proxy of the OpenSourceAP signal `momseason06yrplus`.
Cross-sectional ranking is not feasible per-ticker; instead we compute
the average same-calendar-month return over the 5 yearly lags spanning
years 6–10 prior (approx. months -72, -84, -96, -108, -120 relative
to the current observation). A positive value means the stock
historically rose in this month during that older window, predicting a
continuation per Heston & Sadka (2008).
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_momseason06yrplus",
    "description": (
        "Average same-calendar-month monthly return over the 5 yearly lags "
        "spanning years 6–10 prior to each observation (Heston & Sadka 2008). "
        "Stocks that outperformed in the same month across years 6–10 ago tend "
        "to continue (predicted sign +1, long high). "
        "Per-ticker OHLCV proxy: cross-sectional ranking is not applied; the "
        "raw average prior-year-same-month return captures the economic signal "
        "faithfully at the individual stock level."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momseason06yrplus_avg",   # avg same-month return over years 6–10
        "osap_momseason06yrplus_std",   # volatility of those 5 same-month returns
        "osap_momseason06yrplus_cnt",   # number of non-NaN observations (data quality)
    ],
    "tags": ["seasonality", "momentum", "return", "calendar"],
    "version": "1.0",
    "author": "Heston and Sadka 2008 via OpenSourceAP (Chen-Zimmermann)",
}

# Approximate number of trading days per year
_TDAYS_PER_YEAR = 252
# Lags in calendar years (6 through 10)
_YEAR_LAGS = [6, 7, 8, 9, 10]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # We need monthly returns keyed by (year, month).                     #
    # Strategy:                                                            #
    #   1. Compute the month-end close for each calendar month using the  #
    #      last available close of that month (no lookahead: within a     #
    #      given month, we look at rows already present).                 #
    #   2. For each row, find its (year, month); look up the monthly      #
    #      return for (year-lag, month) for lags 6–10.                    #
    #   3. Average the up-to-5 valid values; propagate back to each       #
    #      trading day in the current month.                              #
    # ------------------------------------------------------------------ #

    n = len(df)
    result_avg = np.full(n, np.nan)
    result_std = np.full(n, np.nan)
    result_cnt = np.full(n, np.nan)

    if n < 2:
        df["osap_momseason06yrplus_avg"] = result_avg
        df["osap_momseason06yrplus_std"] = result_std
        df["osap_momseason06yrplus_cnt"] = result_cnt
        return df

    # Build a DatetimeIndex-aware series of Close
    dates = pd.to_datetime(df["Date"])
    close = df["Close"].values.astype(float)

    # ---- Step 1: build month-end close series ----
    # month_end_close[(year, month)] = last close price in that month
    # We do this by grouping the array by (year, month) and taking the
    # last element (which is correct because df is sorted ascending by Date).

    years = dates.dt.year.values
    months = dates.dt.month.values

    # Build a dict: (yr, mo) -> last close (ascending order ensures last = month-end)
    month_end_close: dict[tuple[int, int], float] = {}
    for i in range(n):
        ym = (int(years[i]), int(months[i]))
        # Overwrite each time; last overwrite = last day of month
        c = close[i]
        if not np.isnan(c):
            month_end_close[ym] = c

    # ---- Step 2: compute monthly return for each (yr, mo) ----
    # monthly_ret[(yr, mo)] = (close[yr,mo] / close[prev_month]) - 1
    # We need prev month's close too; derive it from month_end_close.
    def _prev_ym(yr: int, mo: int):
        if mo == 1:
            return (yr - 1, 12)
        return (yr, mo - 1)

    monthly_ret: dict[tuple[int, int], float] = {}
    for ym, c_end in month_end_close.items():
        prev_ym = _prev_ym(ym[0], ym[1])
        c_prev = month_end_close.get(prev_ym, np.nan)
        if c_prev and c_prev > 0 and not np.isnan(c_prev):
            monthly_ret[ym] = c_end / c_prev - 1.0
        else:
            monthly_ret[ym] = np.nan

    # ---- Step 3: for each row, compute average same-month return over lags 6-10 ----
    # We assign the same value to ALL rows within the same (year, month).
    # Cache per (year, month) to avoid recomputation.
    cache: dict[tuple[int, int], tuple[float, float, float]] = {}

    def _season_signal(yr: int, mo: int):
        key = (yr, mo)
        if key in cache:
            return cache[key]
        vals = []
        for lag in _YEAR_LAGS:
            past_ym = (yr - lag, mo)
            r = monthly_ret.get(past_ym, np.nan)
            if not np.isnan(r):
                vals.append(r)
        if len(vals) == 0:
            out = (np.nan, np.nan, 0.0)
        elif len(vals) == 1:
            out = (vals[0], np.nan, 1.0)
        else:
            arr = np.array(vals, dtype=float)
            out = (float(np.mean(arr)), float(np.std(arr, ddof=1)), float(len(arr)))
        cache[key] = out
        return out

    for i in range(n):
        yr = int(years[i])
        mo = int(months[i])
        avg, std, cnt = _season_signal(yr, mo)
        result_avg[i] = avg
        result_std[i] = std
        result_cnt[i] = cnt

    df["osap_momseason06yrplus_avg"] = result_avg
    df["osap_momseason06yrplus_std"] = result_std
    df["osap_momseason06yrplus_cnt"] = result_cnt
    return df
