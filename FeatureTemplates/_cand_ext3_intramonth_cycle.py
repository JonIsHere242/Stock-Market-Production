"""
Intramonth Position Cyclicality
---------------------------------
Captures the cyclical position of each trading day within its calendar month
as sin/cos coordinates, plus an interaction between the stock's trailing
return mean and a first-half vs second-half-of-month indicator.

All signals are derived purely from Date (+ Close for returns); no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_intramonth_cycle",
    "description": (
        "Encodes the trading-day position within the calendar month as two "
        "cyclical coordinates: sin(2*pi*dayidx/mlen) and cos(2*pi*dayidx/mlen), "
        "where dayidx = 0-based index of the trading day within the month and "
        "mlen = total number of trading days in that month (forward-safe: estimated "
        "from days already seen up to the current row using a 13-month rolling count). "
        "Also computes the interaction of a 21-day trailing mean daily return with the "
        "first-half-vs-second-half-of-month binary indicator, capturing turn-of-month "
        "momentum asymmetry. Pure calendar + OHLCV proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_intramonth_cycle_sin",
        "ext3_intramonth_cycle_cos",
        "ext3_intramonth_cycle_ret_x_half",
    ],
    "tags": ["calendar", "cyclical", "intramonth", "seasonality", "momentum"],
    "version": "1.0.0",
    "author": "Round-4 expansion (NEW: calendar)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 0. Ensure Date is datetime; work on a copy of the Date column only  #
    # ------------------------------------------------------------------ #
    dates = pd.to_datetime(df["Date"])

    # ------------------------------------------------------------------ #
    # 1. Trading-day index within month (0-based, causal)                 #
    #    dayidx[t] = number of rows with the SAME year-month that appear  #
    #                on or before row t (up to and including row t).      #
    #    We compute a cumcount-within-group; because df is ascending by   #
    #    Date, cumcount is purely historical -- no lookahead.             #
    # ------------------------------------------------------------------ #
    ym = dates.dt.to_period("M")  # year-month label per row

    # cumcount within (Ticker, year-month) -- 0-based
    dayidx = ym.groupby(ym).cumcount()  # pd.Series aligned to df index

    # ------------------------------------------------------------------ #
    # 2. Month length estimate (number of trading days in that month)     #
    #    We cannot use the *full* count of a month's trading days without #
    #    peeking into the future for the current month.                   #
    #    Causal proxy: use the PREVIOUS month's observed trading-day      #
    #    count. For the very first month we fall back to 21 (typical).   #
    # ------------------------------------------------------------------ #
    # Count trading days per year-month from the data we have so far.
    # Because each stock's df is one ticker, group by ym period.
    month_size = ym.groupby(ym).transform("count")  # full-month count per row

    # Build a lagged lookup: for each row, find the count of the PREVIOUS month.
    # Steps:
    #   a) unique months in chronological order
    #   b) map each month to its full count
    #   c) shift by 1 to get previous month count
    unique_months = pd.Series(ym.unique()).sort_values()
    counts_series = ym.groupby(ym).count()  # index=Period, values=count

    # Align unique_months to counts_series
    prev_month_count = pd.Series(
        counts_series.values, index=counts_series.index
    ).shift(1)  # previous month's count; first month gets NaN

    # Map each row's ym to the previous month count
    prev_month_map = prev_month_count.reindex(ym)
    prev_month_map.index = df.index  # realign to df index

    # For the first month (NaN), use 21 as the fallback
    mlen = prev_month_map.fillna(21).astype(float)

    # Guard: mlen must be >= 1
    mlen = mlen.where(mlen >= 1, 21)

    # ------------------------------------------------------------------ #
    # 3. Cyclical coordinates                                              #
    # ------------------------------------------------------------------ #
    angle = 2.0 * np.pi * dayidx.values.astype(float) / mlen.values

    df["ext3_intramonth_cycle_sin"] = np.sin(angle)
    df["ext3_intramonth_cycle_cos"] = np.cos(angle)

    # ------------------------------------------------------------------ #
    # 4. First-half vs second-half indicator (1 = first half, 0 = second) #
    #    First half: dayidx < mlen / 2                                    #
    # ------------------------------------------------------------------ #
    half_mask = (dayidx.values < mlen.values / 2.0).astype(float)

    # ------------------------------------------------------------------ #
    # 5. Trailing 21-day mean daily return                                 #
    # ------------------------------------------------------------------ #
    close = df["Close"].replace(0, np.nan)
    daily_ret = close.pct_change()  # NaN at first row; no lookahead

    trailing_mean_ret = daily_ret.rolling(window=21, min_periods=5).mean()

    # ------------------------------------------------------------------ #
    # 6. Interaction: trailing_mean_ret * half_indicator                  #
    # ------------------------------------------------------------------ #
    df["ext3_intramonth_cycle_ret_x_half"] = trailing_mean_ret * half_mask

    return df
