"""
Momentum without the Seasonal Part (Heston & Sadka 2008)
=========================================================
Average monthly return over the prior 12 months, excluding any month
whose calendar month matches the current calendar month.

Per-ticker proxy: at each daily bar we look back ~252 trading days (~12
calendar months), resample into monthly returns, drop the month(s) whose
calendar-month number equals the current bar's calendar month, and
average the remaining monthly returns.

Spec ID : osap_mom12moffseason
Source  : OpenSourceAP (Chen-Zimmermann); Heston and Sadka (2008)
Sign    : +1  (long high off-season momentum)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_mom12moffseason",
    "description": (
        "Per-ticker proxy for Heston & Sadka (2008) 'momentum without the seasonal part'. "
        "At each daily bar, looks back ~252 trading days (~12 calendar months), "
        "computes month-end returns for each calendar month in the window, "
        "excludes months whose calendar month equals the current bar's calendar month "
        "(stripping the seasonal/repetitive component), and averages the remaining "
        "monthly returns. A slope variant (change over 63-day window) and a "
        "normalised z-score variant are also produced. Inherently cross-sectional "
        "in the paper; this is a faithful per-ticker proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_mom12moffseason_avg",    # mean of off-season monthly returns
        "osap_mom12moffseason_slope",  # 63-day rolling change in the signal
        "osap_mom12moffseason_z",      # 252-day rolling z-score of the signal
    ],
    "tags": ["momentum", "seasonality", "monthly", "heston_sadka"],
    "version": "1.0",
    "author": "Heston and Sadka (2008); OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute off-season momentum signal per ticker."""

    # --------------------------------------------------------------------------
    # Guard: need at least a few rows
    # --------------------------------------------------------------------------
    n = len(df)
    if n < 30:
        df["osap_mom12moffseason_avg"] = np.nan
        df["osap_mom12moffseason_slope"] = np.nan
        df["osap_mom12moffseason_z"] = np.nan
        return df

    close = df["Close"].values.astype(np.float64)
    dates = pd.to_datetime(df["Date"])
    date_arr = dates.values  # numpy datetime64

    # --------------------------------------------------------------------------
    # Step 1: build month-end return series from the full per-ticker price history
    # We need month-end closes to form monthly returns.
    # --------------------------------------------------------------------------
    tmp = pd.DataFrame({"Date": dates, "Close": close})
    tmp = tmp.set_index("Date")

    # resample to month-end using last available close
    monthly = tmp["Close"].resample("ME").last().dropna()

    if len(monthly) < 3:
        df["osap_mom12moffseason_avg"] = np.nan
        df["osap_mom12moffseason_slope"] = np.nan
        df["osap_mom12moffseason_z"] = np.nan
        return df

    # monthly simple returns: r_t = Close_t / Close_{t-1} - 1
    monthly_ret = monthly.pct_change()  # index = month-end dates, value = that month's return
    # month number for each observation
    monthly_month = monthly_ret.index.month  # 1..12

    # --------------------------------------------------------------------------
    # Step 2: for each daily bar, compute off-season momentum
    # We look back ~252 trading days = ~12 calendar months.
    # We identify which month-end obs fall inside [bar_date - 252td, bar_date),
    # exclude those whose calendar month == current calendar month, then average.
    # --------------------------------------------------------------------------
    # To vectorise efficiently we compute per calendar-day via a rolling scheme:
    # For each daily bar we:
    #   a) find the bar's calendar month M
    #   b) sum all monthly returns in the trailing ~12-month window, excluding
    #      those in month M
    # We work in a numpy-friendly way by broadcasting.

    # Convert monthly_ret index to int64 (ns) for searchsorted
    monthly_dates_ns = monthly_ret.index.astype(np.int64)
    monthly_ret_vals = monthly_ret.values.astype(np.float64)  # may contain NaN at start
    monthly_month_vals = monthly_month.values.astype(np.int32)

    # ~252 trading days ≈ 365 calendar days; use 366 days lookback to be safe
    lookback_ns = np.int64(366 * 24 * 3600 * int(1e9))

    date_ns = dates.values.astype(np.int64)
    bar_months = dates.dt.month.values.astype(np.int32)

    result = np.full(n, np.nan)

    for i in range(n):
        d_ns = date_ns[i]
        m = bar_months[i]

        # window: (d_ns - lookback_ns, d_ns]
        lo = d_ns - lookback_ns
        # find indices of monthly_ret inside window
        left = np.searchsorted(monthly_dates_ns, lo, side="right")
        right = np.searchsorted(monthly_dates_ns, d_ns, side="right")

        if right <= left:
            continue

        window_ret = monthly_ret_vals[left:right]
        window_month = monthly_month_vals[left:right]

        # exclude months matching current calendar month and NaN entries
        mask = (window_month != m)
        sel = window_ret[mask]
        # drop NaN
        valid = sel[~np.isnan(sel)]
        if len(valid) < 2:
            continue

        result[i] = float(np.mean(valid))

    # --------------------------------------------------------------------------
    # The inner loop over n rows could be slow for n~700.
    # At 700 rows with binary-search per row this is ~O(n log M) where M~84
    # monthly obs — well under 100 ms.
    # --------------------------------------------------------------------------

    df["osap_mom12moffseason_avg"] = result

    # --------------------------------------------------------------------------
    # Slope: 63-day rolling change (momentum of momentum)
    # --------------------------------------------------------------------------
    avg_series = pd.Series(result, index=df.index)
    df["osap_mom12moffseason_slope"] = avg_series.diff(63)

    # --------------------------------------------------------------------------
    # Z-score: 252-day rolling standardisation
    # --------------------------------------------------------------------------
    roll = avg_series.rolling(252, min_periods=60)
    roll_mean = roll.mean()
    roll_std = roll.std()
    denom = roll_std.where(roll_std > 0, other=np.nan)
    df["osap_mom12moffseason_z"] = (avg_series - roll_mean) / denom

    return df
