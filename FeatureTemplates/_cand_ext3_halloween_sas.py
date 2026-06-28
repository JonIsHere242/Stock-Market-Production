"""
Halloween / Sell-in-May seasonality feature block.

Computes per-ticker trailing-3yr mean monthly return for the Nov-Apr
('winter') half-year minus the May-Oct ('summer') half-year, plus an
interaction term: the current half-year indicator times that spread.

All computations are purely causal (no lookahead): the trailing window
uses only past observations available at bar t.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext3_halloween_sas",
    "description": (
        "Halloween / Sell-in-May seasonality. "
        "For each bar computes: (1) trailing-3yr causal per-stock spread = "
        "mean monthly return in Nov-Apr window minus mean monthly return in "
        "May-Oct window, using only past data; (2) current_winter_indicator = "
        "1 if month in {11,12,1,2,3,4} else -1; (3) interaction = "
        "indicator * spread (positive when winter and spread>0 or summer and "
        "spread<0). Per-ticker proxy -- cross-sectional ranking not applied."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_halloween_sas_spread",
        "ext3_halloween_sas_winter",
        "ext3_halloween_sas_interaction",
    ],
    "tags": ["calendar", "seasonality", "halloween", "sell_in_may"],
    "version": "1.0",
    "author": "Round-4 expansion (NEW: calendar); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # Compute daily log returns (causal: ret[t] uses Close[t] / Close[t-1])
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # log return at each bar (NaN at index 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.where(close[:-1] != 0,
                           np.log(close[1:] / close[:-1]),
                           np.nan)
    log_ret = np.concatenate([[np.nan], log_ret])  # align to df index

    # Extract month for each bar
    dates = pd.to_datetime(df["Date"])
    months = dates.dt.month.to_numpy()

    # Winter months = Nov, Dec, Jan, Feb, Mar, Apr
    WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

    # We need per-bar trailing-3yr monthly return aggregates.
    # Strategy: build a monthly return series (resample), then for each
    # bar compute the trailing 36 monthly obs before that bar's
    # year-month, and split by season.
    #
    # Steps:
    # 1. Attach log_ret & month to a frame indexed by date.
    # 2. Compute calendar-month return (sum of daily log rets).
    # 3. For each bar, find trailing <=36 complete months before it,
    #    split into winter/summer, compute means, then take spread.
    #
    # For vectorisation we pre-build the monthly table, then use
    # searchsorted to find which monthly rows precede each bar.

    tmp = pd.DataFrame({
        "Date": dates,
        "log_ret": log_ret,
        "month": months,
    })
    tmp = tmp.set_index("Date")

    # Monthly return = sum of daily log rets within that calendar month
    monthly = (
        tmp["log_ret"]
        .resample("ME")          # month-end; one row per calendar month
        .sum(min_count=1)        # NaN if all NaN
        .reset_index()
    )
    monthly.columns = ["month_end", "m_ret"]
    monthly["month_num"] = monthly["month_end"].dt.month
    monthly["is_winter"] = monthly["month_num"].isin(WINTER_MONTHS)
    # month_end timestamps (for searchsorted)
    me_arr = monthly["month_end"].to_numpy(dtype="datetime64[ns]")
    m_ret_arr = monthly["m_ret"].to_numpy(dtype=np.float64)
    is_winter_arr = monthly["is_winter"].to_numpy()

    # For each bar t, we want: trailing 36 *complete* months ending BEFORE
    # the first day of bar t's calendar month, i.e. months whose month_end
    # < start-of-current-month.
    bar_dates_ns = dates.to_numpy(dtype="datetime64[ns]")

    # Start of each bar's calendar month
    bar_month_start = (dates - pd.offsets.MonthBegin(1)).to_numpy(dtype="datetime64[ns]")
    # Correct for bars already at month start
    already_start = (dates.day == 1).to_numpy()
    bar_month_start = np.where(already_start, bar_dates_ns, bar_month_start)

    # For bars on the 1st, month_start == bar_date; months with month_end <
    # bar_date are fully prior months. For other bars, same logic works.
    # Actually safest: month_end < bar's month-start means it's a PRIOR month.
    # We want months where month_end < floor_month_start(bar).
    # floor_month_start(bar) = year-month-01 00:00 of the bar's month.
    floor_starts = pd.to_datetime(
        {"year": dates.dt.year, "month": dates.dt.month, "day": 1}
    ).to_numpy(dtype="datetime64[ns]")

    spread_arr = np.full(n, np.nan)
    winter_ind_arr = np.empty(n, dtype=np.float64)
    for i, month_num in enumerate(months):
        winter_ind_arr[i] = 1.0 if month_num in WINTER_MONTHS else -1.0

    # Use searchsorted to find boundary in monthly table
    # me_arr is sorted ascending by construction
    boundaries = np.searchsorted(me_arr, floor_starts, side="left")
    # boundaries[i] = number of monthly rows with month_end < floor_starts[i]

    LOOKBACK = 36  # trailing months

    for i in range(n):
        end_idx = boundaries[i]   # exclusive; these are prior months
        if end_idx < 2:           # need at least some data
            continue
        start_idx = max(0, end_idx - LOOKBACK)
        w_rets = m_ret_arr[start_idx:end_idx][is_winter_arr[start_idx:end_idx]]
        s_rets = m_ret_arr[start_idx:end_idx][~is_winter_arr[start_idx:end_idx]]
        if len(w_rets) < 3 or len(s_rets) < 3:
            continue
        w_mean = np.nanmean(w_rets)
        s_mean = np.nanmean(s_rets)
        if np.isnan(w_mean) or np.isnan(s_mean):
            continue
        spread_arr[i] = w_mean - s_mean

    # Guard against inf
    spread_arr = np.where(np.isinf(spread_arr), np.nan, spread_arr)
    interaction = winter_ind_arr * spread_arr
    interaction = np.where(np.isinf(interaction), np.nan, interaction)

    df["ext3_halloween_sas_spread"] = spread_arr
    df["ext3_halloween_sas_winter"] = winter_ind_arr
    df["ext3_halloween_sas_interaction"] = interaction

    return df
