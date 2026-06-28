"""
Return seasonality years 2–5 (Heston & Sadka 2008).

For each bar, look back at the same calendar month in years t-2, t-3, t-4, t-5
(i.e., approximately 24, 36, 48, 60 months ago) and average the monthly return
observed in that same month.  Stocks that historically rose in the current
calendar month tend to continue rising in that month (seasonal alpha).

Per-ticker implementation: fully faithful to the definition using monthly
price sampling.  This is NOT cross-sectional -- the signal is computed entirely
from each ticker's own price history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momseason",
    "description": (
        "Return seasonality years 2-5 (Heston & Sadka 2008 / Chen-Zimmermann OpenSourceAP). "
        "For each daily bar, computes the average same-calendar-month return at lags "
        "24, 36, 48, 60 months (years t-2 to t-5).  Monthly return = "
        "last-trading-day close in month M / last-trading-day close in month M-1 - 1. "
        "Produces: level (mean of 4 same-month past returns), std across those 4 lags, "
        "and a 3-month EMA of the level for a smoothed dynamic variant. "
        "Per-ticker proxy; sign = +1 (high seasonality predicts positive future return)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momseason_avg",   # mean same-month return over lags 2-5y
        "osap_momseason_std",   # cross-lag std (uncertainty)
        "osap_momseason_ema3",  # 3-month EMA of avg for smoothed signal
    ],
    "tags": ["seasonality", "momentum", "monthly", "heston_sadka", "price"],
    "version": "1.0",
    "author": "Heston & Sadka (2008) via Chen-Zimmermann OpenSourceAP; block by Claude",
}

# Lags in months for years 2, 3, 4, 5
_LAGS = [24, 36, 48, 60]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Build a monthly close series (last trading day of each month)
    # ------------------------------------------------------------------ #
    close = df[["Date", "Close"]].copy()
    close["Date"] = pd.to_datetime(close["Date"])
    close = close.set_index("Date")["Close"]

    # Resample to month-end using last available trading day
    monthly = close.resample("ME").last()   # DatetimeIndex at month-end

    if len(monthly) < max(_LAGS) + 2:
        # Not enough history — emit NaNs
        df["osap_momseason_avg"] = np.nan
        df["osap_momseason_std"] = np.nan
        df["osap_momseason_ema3"] = np.nan
        return df

    # Monthly log returns (arithmetic return)
    monthly_ret = monthly.pct_change()  # ret[t] = close[t]/close[t-1] - 1

    # Month number (1-12) for each monthly bar
    month_num = monthly.index.month

    # ------------------------------------------------------------------ #
    # 2. For each monthly bar t, compute mean return across the 4 lags
    #    if and only if the lagged bar falls in the same calendar month.
    #    (Due to irregular month lengths this should always hold for
    #    integer-month lags, but we guard anyway.)
    # ------------------------------------------------------------------ #
    n = len(monthly)
    avg_arr = np.full(n, np.nan)
    std_arr = np.full(n, np.nan)

    # We need index positions; build integer index
    for i in range(n):
        cur_month = month_num[i]
        lag_rets = []
        for lag in _LAGS:
            j = i - lag
            if j < 0:
                continue
            # Confirm same calendar month (should always be true for exact month lags)
            if month_num[j] == cur_month:
                ret_val = monthly_ret.iloc[j]
                if np.isfinite(ret_val):
                    lag_rets.append(ret_val)
        if len(lag_rets) >= 1:
            avg_arr[i] = float(np.mean(lag_rets))
        if len(lag_rets) >= 2:
            std_arr[i] = float(np.std(lag_rets, ddof=1))

    monthly_avg = pd.Series(avg_arr, index=monthly.index)
    monthly_std = pd.Series(std_arr, index=monthly.index)

    # 3-month EMA of the seasonality average (span=3 months)
    monthly_ema3 = monthly_avg.ewm(span=3, min_periods=1, adjust=False).mean()

    # ------------------------------------------------------------------ #
    # 3. Forward-fill monthly signal back to daily bars (no lookahead:
    #    the monthly value for month M is only assigned after month M's
    #    last trading day has passed, i.e., starting the first day of M+1).
    #    We shift by 1 month so each daily bar in month M+1 sees the
    #    signal computed through month M.
    # ------------------------------------------------------------------ #
    # Shift the monthly series by 1 period (= 1 month) to avoid lookahead
    monthly_avg_lag = monthly_avg.shift(1)
    monthly_std_lag = monthly_std.shift(1)
    monthly_ema3_lag = monthly_ema3.shift(1)

    # Reindex to daily using the original df dates, forward-filling
    daily_index = pd.to_datetime(df["Date"])

    def _daily_fill(monthly_series: pd.Series) -> np.ndarray:
        # merge_asof: each daily date gets the most recent past month-end value
        daily_df = pd.DataFrame({"Date": daily_index})
        monthly_df = pd.DataFrame(
            {"Date": monthly_series.index, "val": monthly_series.values}
        ).dropna(subset=["Date"])
        merged = pd.merge_asof(
            daily_df.sort_values("Date"),
            monthly_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # restore original row order
        merged = merged.set_index(daily_df.sort_values("Date").index)
        # align back to original df index order
        return merged["val"].reindex(df.index).values

    df["osap_momseason_avg"] = _daily_fill(monthly_avg_lag)
    df["osap_momseason_std"] = _daily_fill(monthly_std_lag)
    df["osap_momseason_ema3"] = _daily_fill(monthly_ema3_lag)

    return df
