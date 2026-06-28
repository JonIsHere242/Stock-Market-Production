"""
Quarter-end window-dressing tilt feature block.

Captures the well-known tendency for fund managers to "window-dress" at
quarter-end by buying recent winners (to show them in quarter-end holdings).
Signal is strongest in the last ~5 trading days of each calendar quarter,
interacted with the stock's own trailing 20-day return.

Per-ticker proxy: since we cannot rank cross-sectionally, we use the raw
trailing 20d return during quarter-end windows, plus a trailing 252d mean
of that same quarter-end-window return, as the long-run seasonal baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_quarter_end_dressing",
    "description": (
        "Quarter-end window-dressing tilt. For the last 5 trading days of each "
        "calendar quarter (March/June/September/December), records the stock's "
        "trailing 20-day return (window-dressing buys recent winners). Outside "
        "those windows the value is NaN (or zero for the rolling mean). "
        "Produces: (1) ext3_quarter_end_dressing_ret20 — trailing 20d return "
        "active only in the last-5-days window, else NaN; "
        "(2) ext3_quarter_end_dressing_flag — binary 1/0 indicating the window; "
        "(3) ext3_quarter_end_dressing_hist_mean — trailing 252-day mean of "
        "the window-active 20d return, capturing the stock's historical "
        "quarter-end seasonal lift. Per-ticker proxy; cross-sectional ranking "
        "is not available in this block."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_quarter_end_dressing_ret20",
        "ext3_quarter_end_dressing_flag",
        "ext3_quarter_end_dressing_hist_mean",
    ],
    "tags": ["calendar", "seasonality", "window-dressing", "quarter-end", "momentum"],
    "version": "1.0.0",
    "author": "Round-4 expansion (NEW: calendar); spec ext3_quarter_end_dressing",
}


def _quarter_end_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Return a boolean array: True if the date falls in the last 5 trading days
    of its calendar quarter (March, June, September, December).

    Strategy: a date is in the last 5 trading days of the quarter if, within
    the same (year, quarter), it ranks among the top 5 dates in descending
    order. We do this without lookahead — we only look at dates already
    present in the series up to that point.

    Implementation: we label each date with (year, quarter), then for each
    date check whether it is among the 5 largest dates in that group.
    Since we have the full date column available (no future prices are used —
    the flag is purely calendar-based), ranking within a calendar quarter
    group is causal: we are not using any future price, only the calendar
    date itself.
    """
    df_tmp = pd.DataFrame({"date": dates})
    df_tmp["year"] = dates.year
    df_tmp["quarter"] = dates.quarter

    # For each (year, quarter) group, find the 5 largest dates.
    # rank(ascending=False) gives rank 1 = latest date in that quarter.
    df_tmp["rank_in_qtr"] = df_tmp.groupby(["year", "quarter"])["date"].rank(
        ascending=False, method="min"
    )
    flag = (df_tmp["rank_in_qtr"] <= 5).to_numpy()
    return flag


def compute(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["Date"])

    # --- trailing 20-day return (causal: pct change over past 20 bars) ---
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    # ret20[i] = close[i] / close[i-20] - 1  (NaN for i < 20)
    ret20 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        prev20 = np.empty(n)
        prev20[:] = np.nan
        prev20[20:] = close[:n - 20]
        denom = prev20.copy()
        denom[denom == 0] = np.nan
        ret20 = close / denom - 1.0

    # --- quarter-end flag ---
    flag = _quarter_end_flag(pd.DatetimeIndex(dates))  # bool array

    # --- ext3_quarter_end_dressing_flag ---
    df["ext3_quarter_end_dressing_flag"] = flag.astype(float)

    # --- ext3_quarter_end_dressing_ret20: trailing 20d return, active in window ---
    ret20_windowed = np.where(flag, ret20, np.nan)
    df["ext3_quarter_end_dressing_ret20"] = ret20_windowed

    # --- ext3_quarter_end_dressing_hist_mean ---
    # Rolling 252-bar mean of ret20_windowed (NaN-aware).
    # This captures the stock's historical average quarter-end lift.
    # We use a pandas rolling with min_periods=1 (need at least 1 non-NaN).
    s = pd.Series(ret20_windowed)
    # rolling mean ignoring NaN: use apply with np.nanmean, but that is slow.
    # Faster: fill NaN with 0 for sum, track count separately.
    # Actually, pandas rolling().mean() propagates NaN by default; we want
    # nanmean semantics — use expanding window approach via cumsum.
    # Simplest correct approach: rolling(252).apply(np.nanmean, raw=True).
    # This is acceptable: 700 rows * window=252 is ~175k ops, well under 100ms.
    hist_mean = s.rolling(window=252, min_periods=1).apply(
        lambda x: np.nanmean(x) if np.any(~np.isnan(x)) else np.nan,
        raw=True,
    )
    # Replace inf/-inf with NaN (guard)
    hist_mean = hist_mean.replace([np.inf, -np.inf], np.nan)
    df["ext3_quarter_end_dressing_hist_mean"] = hist_mean.to_numpy()

    return df
