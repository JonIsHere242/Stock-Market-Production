"""
Feature block: osap_momseason16yrplus
Return seasonality: average same-month return over preceding 16-20 years.

Heston & Sadka (2008) show that stocks with high average returns in the same
calendar month over long horizons continue to outperform — a persistent
seasonal pattern in cross-sectional returns. The cross-sectional ranking
requires a panel; this block implements the per-ticker proxy: for each bar,
compute the equal-weight average of the monthly returns observed 16, 17, 18,
19, and 20 years prior in the SAME calendar month, using the stock's own
price history. The economic content (same-month seasonal autocorrelation) is
faithfully captured per-ticker; the cross-sectional rank is an inference
overlay applied downstream.

Per-ticker proxy: uses only this ticker's OHLCV history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momseason16yrplus",
    "description": (
        "Per-ticker proxy for Heston-Sadka (2008) long-horizon return seasonality. "
        "For each date, computes the average monthly return in the same calendar month "
        "over the 16, 17, 18, 19, and 20 years prior. A high value indicates persistent "
        "seasonal tailwinds for this ticker in the current month. "
        "Cross-sectional ranking not applied (per-ticker proxy)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momseason16yrplus_avg",   # mean same-month ret across lags 16-20yr
        "osap_momseason16yrplus_n",     # number of non-NaN lag observations (0-5)
        "osap_momseason16yrplus_slope", # OLS slope of the 5 lagged seasonal rets (trend in seasonal strength)
    ],
    "tags": ["seasonality", "momentum", "long-horizon", "heston-sadka", "monthly"],
    "version": "1.0",
    "author": "Heston and Sadka 2008 (via OpenSourceAP / Chen-Zimmermann); per-ticker proxy implementation",
}

# approximate trading days per year
_DAYS_PER_YEAR = 252


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a minimal copy; keep original df intact throughout.
    dates = pd.to_datetime(df["Date"])
    close = df["Close"].values
    n = len(df)

    # Output arrays — initialise to NaN
    avg_arr = np.full(n, np.nan, dtype=np.float64)
    n_arr = np.zeros(n, dtype=np.float64)
    slope_arr = np.full(n, np.nan, dtype=np.float64)

    # Build a month-end close series so we can look up same-month returns.
    # For each bar we need the *monthly* return for the same calendar month in
    # years -16 .. -20.  We derive monthly returns from the daily series by
    # taking the last available close in each (year, month) bucket.

    # Step 1: build a (year, month) -> last-day-index map (no lookahead; we
    # will only look BACKWARDS from the current row's date).
    # We iterate over the date array once to assign each row to its (yr, mo).
    years = dates.dt.year.values
    months = dates.dt.month.values

    # For each (yr, mo) store the index of the last bar in that month (we'll
    # compute monthly returns as the change from the last bar of the prior
    # month to the last bar of this month).
    # We build this as a dict: (yr, mo) -> last_close_in_month
    # We accumulate as we go, so at row i we have a fully backward-looking map.

    # Pre-compute the per-(yr,mo) last-close using groupby (vectorised).
    # This uses all rows — we need to be careful: the map is used only to look
    # up values MORE THAN 16 YEARS in the past, so the lookahead risk is
    # negligible for the PRODUCTION bars (bars within the last 16 years can't
    # contribute lags, so they output NaN anyway).  The map is statically built
    # from the full per-stock series, which is safe because the seasonal lag
    # values are always ≥16 years old at every output row.

    # Group by (year, month), take the last close.
    ym_idx = years * 100 + months  # unique key per (yr, mo)
    tmp = pd.DataFrame({"ym": ym_idx, "close": close})
    mo_last = tmp.groupby("ym")["close"].last()  # Series indexed by ym key

    # Monthly return for a (yr, mo) cell = close[yr,mo] / close[prior_month] - 1
    # Prior month key: we need the preceding (yr, mo) pair.
    ym_keys = mo_last.index.values  # sorted because groupby is sorted
    mo_close = mo_last.values
    # Compute month-over-month returns for all (yr,mo) pairs
    mo_ret = np.full(len(ym_keys), np.nan, dtype=np.float64)
    for i in range(1, len(ym_keys)):
        yr_i = ym_keys[i] // 100
        mo_i = ym_keys[i] % 100
        yr_p = ym_keys[i - 1] // 100
        mo_p = ym_keys[i - 1] % 100
        # Only use if previous key is exactly the prior calendar month
        if mo_i == 1:
            expected_prev_yr, expected_prev_mo = yr_i - 1, 12
        else:
            expected_prev_yr, expected_prev_mo = yr_i, mo_i - 1
        if yr_p == expected_prev_yr and mo_p == expected_prev_mo:
            prev_c = mo_close[i - 1]
            curr_c = mo_close[i]
            if prev_c != 0 and not np.isnan(prev_c):
                mo_ret[i] = curr_c / prev_c - 1.0

    # Build lookup: ym_key -> monthly_return
    mo_ret_dict: dict[int, float] = {}
    for k, v in zip(ym_keys, mo_ret):
        if not np.isnan(v):
            mo_ret_dict[int(k)] = float(v)

    # x-values for slope: lag years 16..20 mapped to 0..4
    x_lags = np.arange(5, dtype=np.float64)  # 0=16yr, 1=17yr, ..., 4=20yr
    x_mean = x_lags.mean()
    x_denom = float(np.sum((x_lags - x_mean) ** 2))  # = 10.0

    # Step 2: for each row compute the seasonal average over lags 16-20yr
    for i in range(n):
        yr_i = int(years[i])
        mo_i = int(months[i])
        lag_rets = np.empty(5, dtype=np.float64)
        lag_rets[:] = np.nan
        for j, lag in enumerate((16, 17, 18, 19, 20)):
            target_yr = yr_i - lag
            key = target_yr * 100 + mo_i
            if key in mo_ret_dict:
                lag_rets[j] = mo_ret_dict[key]

        valid_mask = ~np.isnan(lag_rets)
        n_valid = int(valid_mask.sum())
        n_arr[i] = float(n_valid)

        if n_valid == 0:
            continue

        avg_arr[i] = float(lag_rets[valid_mask].mean())

        # Slope only meaningful when ≥2 obs
        if n_valid >= 2:
            valid_x = x_lags[valid_mask]
            valid_y = lag_rets[valid_mask]
            vx_mean = valid_x.mean()
            vy_mean = valid_y.mean()
            num = float(np.sum((valid_x - vx_mean) * (valid_y - vy_mean)))
            den = float(np.sum((valid_x - vx_mean) ** 2))
            if den != 0:
                slope_arr[i] = num / den

    df["osap_momseason16yrplus_avg"] = avg_arr
    df["osap_momseason16yrplus_n"] = n_arr
    df["osap_momseason16yrplus_slope"] = slope_arr

    return df
