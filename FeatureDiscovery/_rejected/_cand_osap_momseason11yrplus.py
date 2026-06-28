"""
Return Seasonality Years 11-15 (Heston & Sadka 2008)
Per-ticker proxy: for each bar, compute the average same-calendar-month return
over the 11th through 15th prior years.

Heston & Sadka (2008) document a cross-sectional seasonal momentum effect where
stocks that performed well in the same month in prior years tend to outperform.
This block captures the same signal in a fully per-ticker, lookahead-free way.

Produces:
  osap_momseason11yrplus_avg  - mean of same-month monthly returns in years [t-15 .. t-11]
  osap_momseason11yrplus_std  - std of those same-month returns (signal consistency)
  osap_momseason11yrplus_pos  - fraction of those years with positive same-month return
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momseason11yrplus",
    "description": (
        "Return seasonality years 11-15 (Heston & Sadka 2008, via OpenSourceAP / Chen-Zimmermann). "
        "For each date, computes the average monthly return in the SAME calendar month over the "
        "11th through 15th prior years (i.e. lags of ~132–180 months). "
        "Proxy is fully per-ticker from OHLCV; no cross-sectional ranking needed. "
        "Predicted sign: +1 (long high average seasonal return)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momseason11yrplus_avg",
        "osap_momseason11yrplus_std",
        "osap_momseason11yrplus_pos",
    ],
    "tags": ["seasonality", "momentum", "return", "long-horizon", "heston-sadka"],
    "version": "1.0",
    "author": "Heston & Sadka (2008); OpenSourceAP / Chen-Zimmermann",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute same-month return seasonality over years 11-15.

    Strategy:
      1. Resample Close to month-end prices.
      2. Compute month-over-month log returns on the monthly series.
      3. For each calendar month M in the monthly series at position i,
         look back at months with the SAME calendar month M, at lags
         11 to 15 years (132 to 180 months). Average those 5 returns.
      4. Align the resulting monthly signal back to daily bars (forward-fill
         within each month — only from month-start so no lookahead).
    """
    if df.empty or "Close" not in df.columns:
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    # Work on a copy of the date-indexed close series
    dates = pd.to_datetime(df["Date"])
    close = pd.Series(df["Close"].values, index=dates, dtype=float)

    # ------------------------------------------------------------------ #
    # Build a month-end price series (last trading day of each month)
    # ------------------------------------------------------------------ #
    monthly_close = close.resample("ME").last()
    monthly_close = monthly_close.dropna()

    if len(monthly_close) < 133:  # need at least 11 years + 1 month
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    # Month-over-month log returns
    monthly_ret = np.log(monthly_close / monthly_close.shift(1))

    # Calendar month for each bar in the monthly series
    cal_month = monthly_ret.index.month  # 1..12

    n = len(monthly_ret)
    ret_vals = monthly_ret.values
    cal_month_vals = cal_month.values

    # Lags in months: 11 years = 132, 15 years = 180
    lags = np.arange(132, 181)  # 132,133,...,180 -> 49 months; same-month = every 12

    # For each position i, collect same-calendar-month returns at lags [132..180]
    avg_arr = np.full(n, np.nan)
    std_arr = np.full(n, np.nan)
    pos_arr = np.full(n, np.nan)

    for i in range(n):
        cm = cal_month_vals[i]
        # Indices in [132..180] steps back that share the same calendar month
        same_month_rets = []
        for lag in lags:
            j = i - lag
            if j < 0:
                continue
            if cal_month_vals[j] == cm:
                v = ret_vals[j]
                if np.isfinite(v):
                    same_month_rets.append(v)
        if len(same_month_rets) >= 1:
            arr = np.array(same_month_rets)
            avg_arr[i] = arr.mean()
            std_arr[i] = arr.std(ddof=0) if len(arr) > 1 else np.nan
            pos_arr[i] = (arr > 0).mean()

    # ------------------------------------------------------------------ #
    # Build monthly DataFrame and merge back to daily
    # ------------------------------------------------------------------ #
    monthly_df = pd.DataFrame(
        {
            "Date": monthly_close.index,
            "osap_momseason11yrplus_avg": avg_arr,
            "osap_momseason11yrplus_std": std_arr,
            "osap_momseason11yrplus_pos": pos_arr,
        }
    )
    # monthly_df dates are month-end; we want to broadcast to daily bars
    # Use merge_asof so each daily bar gets the LAST available monthly signal
    # (backward direction = no lookahead: the month-end value becomes available
    #  only once that month has closed).
    daily_df = pd.DataFrame({"Date": dates})
    daily_df = daily_df.sort_values("Date")
    monthly_df = monthly_df.sort_values("Date")

    merged = pd.merge_asof(
        daily_df,
        monthly_df,
        on="Date",
        direction="backward",
    )

    # Restore original row order using the original index
    merged.index = daily_df.index
    merged = merged.reindex(df.index)

    df["osap_momseason11yrplus_avg"] = merged["osap_momseason11yrplus_avg"].values
    df["osap_momseason11yrplus_std"] = merged["osap_momseason11yrplus_std"].values
    df["osap_momseason11yrplus_pos"] = merged["osap_momseason11yrplus_pos"].values

    return df
