"""
Turn-of-month effect feature block.
Spec: ext3_turn_of_month (Round-4 expansion, NEW: calendar)
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext3_turn_of_month",
    "description": (
        "Turn-of-month (TOM) calendar effect. Flags TOM days (last trading day "
        "of the month + first 3 trading days of the next month). Computes a "
        "causal rolling 252-day mean log-return on TOM days vs non-TOM days for "
        "each ticker, and scales by whether today is itself a TOM day. "
        "Per-ticker proxy; no cross-sectional data required. "
        "Produces: (1) ext3_turn_of_month_ind -- binary TOM-day indicator, "
        "(2) ext3_turn_of_month_premium -- rolling TOM-day return premium "
        "(TOM mean minus non-TOM mean log-return, 252-day window), "
        "(3) ext3_turn_of_month_signal -- indicator * premium (active bet size)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_turn_of_month_ind",
        "ext3_turn_of_month_premium",
        "ext3_turn_of_month_signal",
    ],
    "tags": ["calendar", "seasonality", "turn_of_month", "return_premium"],
    "version": "1.0.0",
    "author": "Round-4 expansion (NEW: calendar) / spec ext3_turn_of_month",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute turn-of-month features on a per-ticker OHLCV frame (ascending Date).
    """
    if len(df) < 5:
        df["ext3_turn_of_month_ind"] = np.nan
        df["ext3_turn_of_month_premium"] = np.nan
        df["ext3_turn_of_month_signal"] = np.nan
        return df

    # ---- 1. Build a DatetimeIndex-aligned series from Date column ----------
    dates = pd.to_datetime(df["Date"])

    # ---- 2. Log returns (needed for premium computation) -------------------
    close = df["Close"].values.astype(np.float64)
    # shift(1) forward: log_ret[t] = log(Close[t] / Close[t-1])
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(
            (prev_close > 0) & (close > 0),
            np.log(close / prev_close),
            np.nan,
        )

    # ---- 3. Determine TOM-day indicator ------------------------------------
    # For each date, we need to know:
    #   (a) Is it the last trading day of its month?
    #   (b) Is it among the first 3 trading days of its month?
    #
    # We do this purely from the Date column (no lookahead).
    # "Last trading day of month" = next trading day is in a different month.
    # "First 3 trading days of month" = it's the 1st, 2nd, or 3rd trading day
    #   within its calendar month.

    year_month = dates.dt.to_period("M")  # Period per row
    n = len(df)

    # Rank of each row within its year-month (1-indexed = 1st trading day, etc.)
    # Computed via cumcount grouped by year_month -- purely causal rank within month
    ym_str = year_month.astype(str)  # string for groupby
    within_month_rank = (
        pd.Series(ym_str, index=df.index)
        .groupby(pd.Series(ym_str, index=df.index))
        .cumcount()
        + 1
    )  # 1, 2, 3, ... per month

    # First 3 trading days of the month
    first3 = (within_month_rank <= 3).values.astype(np.float64)  # shape (n,)

    # Last trading day of the month:
    # current month != next row's month (and we are not the last row with no successor)
    cur_ym = ym_str.values
    next_ym = np.empty(n, dtype=object)
    next_ym[:-1] = cur_ym[1:]
    next_ym[-1] = ""  # last row: no successor known -> not flagged as last

    last_td = (cur_ym != next_ym).astype(np.float64)
    # Last row has no next row: we cannot tell if it's the last trading day of
    # the month without lookahead, so set to 0 (conservative, causal)
    last_td[-1] = 0.0

    # TOM indicator: last trading day OR first 3 trading days
    tom_ind = np.clip(last_td + first3, 0.0, 1.0)

    # ---- 4. Rolling TOM premium (causal, 252-day window) ------------------
    # For each day t, using data from [t-251 .. t]:
    #   tom_mean  = mean of log_ret[i] where tom_ind[i]==1, i in [t-251..t]
    #   non_mean  = mean of log_ret[i] where tom_ind[i]==0, i in [t-251..t]
    #   premium   = tom_mean - non_mean
    #
    # We implement this with two pandas rolling sums + rolling counts.

    WINDOW = 252

    lr_series = pd.Series(log_ret, index=df.index)
    tom_series = pd.Series(tom_ind, index=df.index)

    # TOM contributions
    tom_ret = lr_series * tom_series  # nan where log_ret is nan
    tom_count = tom_series.where(lr_series.notna(), other=np.nan)  # 1 or nan

    # Non-TOM contributions
    non_tom_mask = (1.0 - tom_series)
    non_tom_ret = lr_series * non_tom_mask
    non_tom_count = non_tom_mask.where(lr_series.notna(), other=np.nan)

    # Rolling sums (min_periods ensures we need at least some data)
    tom_sum = tom_ret.rolling(window=WINDOW, min_periods=20).sum()
    tom_cnt = tom_count.rolling(window=WINDOW, min_periods=20).sum()

    non_sum = non_tom_ret.rolling(window=WINDOW, min_periods=20).sum()
    non_cnt = non_tom_count.rolling(window=WINDOW, min_periods=20).sum()

    # Mean returns; guard zero counts
    with np.errstate(divide="ignore", invalid="ignore"):
        tom_mean = np.where(tom_cnt > 0, tom_sum.values / tom_cnt.values, np.nan)
        non_mean = np.where(non_cnt > 0, non_sum.values / non_cnt.values, np.nan)

    premium = tom_mean - non_mean  # may be nan if not enough data

    # ---- 5. Signal = indicator * premium -----------------------------------
    signal = tom_ind * np.where(np.isfinite(premium), premium, np.nan)

    # ---- 6. Assign columns to df ------------------------------------------
    df["ext3_turn_of_month_ind"] = tom_ind
    df["ext3_turn_of_month_premium"] = premium
    df["ext3_turn_of_month_signal"] = signal

    return df
