"""
Off-season reversal: average monthly return in the SAME calendar month
over the preceding 11-15 years (Heston & Sadka 2008).

Cross-sectional note: the original paper ranks stocks by this average
return cross-sectionally. Here we implement a faithful PER-TICKER proxy:
for each month-end bar we compute the equally-weighted mean of the stock's
own return in that calendar month across years t-11 through t-15 (inclusive,
5 matching observations max). Predicted sign is -1 (high past return →
reversal going forward). We also emit the slope (trend in those 5 obs)
and the number of valid observations used.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_momoffseason11yrplus",
    "description": (
        "Per-ticker proxy for Heston-Sadka (2008) off-season seasonal reversal. "
        "For each month-end bar, computes the mean monthly return in the SAME "
        "calendar month over the 11 to 15 years immediately preceding (up to 5 "
        "matching observations). Also emits a linear slope across those 5 prior "
        "same-month returns and the count of valid obs. "
        "Predicted sign -1: high average past same-month return → future reversal. "
        "Per-ticker proxy — original factor is cross-sectional rank."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momoffseason11yrplus_mean",
        "osap_momoffseason11yrplus_slope",
        "osap_momoffseason11yrplus_nobs",
    ],
    "tags": ["momentum", "seasonality", "reversal", "monthly", "long-horizon"],
    "version": "1.0",
    "author": "Heston and Sadka 2008 (via OpenSourceAP Chen-Zimmermann); per-ticker proxy",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 0. Work on a copy of the index to keep original df intact           #
    # ------------------------------------------------------------------ #
    close = df["Close"].values
    dates = pd.to_datetime(df["Date"])
    n = len(df)

    # Initialise output arrays
    mean_arr = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)
    nobs_arr = np.full(n, np.nan)

    if n < 2:
        df["osap_momoffseason11yrplus_mean"] = mean_arr
        df["osap_momoffseason11yrplus_slope"] = slope_arr
        df["osap_momoffseason11yrplus_nobs"] = nobs_arr
        return df

    # ------------------------------------------------------------------ #
    # 1. Build a month-end return series                                   #
    #    monthly_ret[m] = last_close_of_month / last_close_of_prior_month #
    # ------------------------------------------------------------------ #
    # Tag each row with year-month
    ym = dates.dt.to_period("M")

    # Identify the last row index for each year-month
    # We iterate over the raw array; vectorised via groupby-last
    temp = pd.DataFrame({
        "idx": np.arange(n),
        "ym": ym,
        "close": close,
    })
    # Last row per month
    monthly = (
        temp.groupby("ym", sort=True)
        .agg(last_idx=("idx", "last"), last_close=("close", "last"))
        .reset_index()
    )
    monthly["ym_ts"] = monthly["ym"].dt.to_timestamp("M")  # end-of-month date
    monthly["month"] = monthly["ym"].dt.month
    monthly["year"] = monthly["ym"].dt.year

    # Monthly return: current month close / previous month close - 1
    monthly["prev_close"] = monthly["last_close"].shift(1)
    monthly["mret"] = np.where(
        monthly["prev_close"] > 0,
        monthly["last_close"] / monthly["prev_close"] - 1.0,
        np.nan,
    )

    # Build a lookup: (year, month) -> mret
    # Using dict for O(1) lookup inside the loop
    mret_lookup: dict[tuple[int, int], float] = {}
    for row in monthly.itertuples(index=False):
        if not np.isnan(row.mret):
            mret_lookup[(row.year, row.month)] = row.mret

    # ------------------------------------------------------------------ #
    # 2. For each month-end row, gather same-month returns at lags 11-15y #
    # ------------------------------------------------------------------ #
    # We only need to compute once per calendar-month period, then
    # broadcast back to all daily rows within that month.
    month_results: dict[tuple[int, int], tuple[float, float, float]] = {}

    for row in monthly.itertuples(index=False):
        yr = row.year
        mo = row.month
        if (yr, mo) in month_results:
            continue

        # Collect returns for same calendar month, years t-11 to t-15
        vals = []
        for lag in range(11, 16):  # 11,12,13,14,15
            past_yr = yr - lag
            ret = mret_lookup.get((past_yr, mo), np.nan)
            if not np.isnan(ret):
                vals.append((lag, ret))

        if len(vals) == 0:
            month_results[(yr, mo)] = (np.nan, np.nan, 0.0)
            continue

        rets = np.array([v[1] for v in vals])
        lags = np.array([v[0] for v in vals], dtype=float)
        nobs = float(len(vals))
        mean_val = float(np.mean(rets))

        # Slope: regress ret on lag (most-recent lag=11 to oldest=15)
        # Use simple OLS: slope = cov(lag, ret) / var(lag)
        if nobs >= 2:
            lag_mean = np.mean(lags)
            lag_var = np.sum((lags - lag_mean) ** 2)
            if lag_var > 0:
                slope_val = float(np.sum((lags - lag_mean) * (rets - mean_val)) / lag_var)
            else:
                slope_val = np.nan
        else:
            slope_val = np.nan

        month_results[(yr, mo)] = (mean_val, slope_val, nobs)

    # ------------------------------------------------------------------ #
    # 3. Map month-period results back to every daily row                  #
    # ------------------------------------------------------------------ #
    row_ym_month = dates.dt.month.values
    row_ym_year = dates.dt.year.values

    for i in range(n):
        key = (int(row_ym_year[i]), int(row_ym_month[i]))
        res = month_results.get(key)
        if res is not None:
            mean_arr[i] = res[0]
            slope_arr[i] = res[1]
            nobs_arr[i] = res[2]

    # ------------------------------------------------------------------ #
    # 4. Assign to df                                                      #
    # ------------------------------------------------------------------ #
    df["osap_momoffseason11yrplus_mean"] = mean_arr
    df["osap_momoffseason11yrplus_slope"] = slope_arr
    df["osap_momoffseason11yrplus_nobs"] = nobs_arr

    return df
