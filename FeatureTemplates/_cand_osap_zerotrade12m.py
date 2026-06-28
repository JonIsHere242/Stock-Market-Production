"""
Zero-trade illiquidity measure (Liu 2006), per-ticker implementation.

Per the OpenSourceAP / Chen-Zimmermann catalog:
  For each month, count days with zero volume (zero-trade days), then add
  (sum-of-monthly-turnover / 48e5), scaled by 21 / trading_days_in_month.
  Take the 12-month average of this monthly statistic.

  Original cross-sectional use: stocks ranked by zerotrade (high = illiquid)
  predict positive future returns (illiquidity premium).

  Per-ticker proxy: we faithfully replicate the monthly construction using
  OHLCV Volume and PIT shares_outstanding from SEC fundamentals.  When
  fundamentals are unavailable (ETFs, foreign stocks) we degrade to a
  volume-only proxy (zero-day fraction * 21).  We also emit a 3-month
  slope to capture dynamic changes in illiquidity.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# -- optional: PIT fundamentals for shares outstanding --
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_zerotrade12m",
    "description": (
        "Liu (2006) zero-trade illiquidity measure. Each calendar month: "
        "count zero-volume days + (sum_of_daily_turnover / 48e5), scaled by "
        "21 / trading_days_in_month.  12-month rolling average of that "
        "monthly statistic is the main signal (high = illiquid = premium). "
        "PIT shares_outstanding used for turnover when available; degrades "
        "to zero-day fraction * 21 otherwise.  Per-ticker proxy for what is "
        "inherently a cross-sectional rank in the original paper "
        "(OpenSourceAP / Chen-Zimmermann; Liu 2006)."
    ),
    "requires": ["Volume", "Close"],
    "produces": [
        "osap_zerotrade12m_lvl",    # 12-month avg zero-trade statistic (main)
        "osap_zerotrade12m_3m",     # 3-month avg (shorter-horizon variant)
        "osap_zerotrade12m_slope",  # slope: 12m avg minus 3m avg (trend)
    ],
    "tags": ["liquidity", "zero-trade", "illiquidity", "liu2006", "monthly"],
    "version": "1.0",
    "author": "Liu 2006 / OpenSourceAP (Chen-Zimmermann); implemented as per-ticker proxy",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Liu (2006) zero-trade illiquidity and slope variant."""

    # ------------------------------------------------------------------ #
    # 1.  Pull PIT shares_outstanding if available                         #
    # ------------------------------------------------------------------ #
    try:
        df = _fundamentals.as_of(df, fields=["shares_outstanding"])
        has_shrout = True
    except Exception:
        has_shrout = False

    # ------------------------------------------------------------------ #
    # 2.  Daily inputs                                                     #
    # ------------------------------------------------------------------ #
    vol = df["Volume"].copy()
    is_zero = (vol == 0) | vol.isna()  # bool: no-trade day

    if has_shrout:
        shrout = df["fund_shares_outstanding"].copy()
        # shares_outstanding in units (from SEC); turnover = vol / shrout
        # Guard against zero / NaN shrout
        turnover = np.where(
            (shrout.isna()) | (shrout <= 0),
            np.nan,
            vol.values / shrout.values,
        )
        turnover = pd.Series(turnover, index=df.index)
    else:
        turnover = pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------ #
    # 3.  Month-level aggregation                                          #
    # ------------------------------------------------------------------ #
    # Use a YearMonth period key to group by calendar month
    dates = pd.to_datetime(df["Date"])
    ym = dates.dt.to_period("M")

    # Build a small frame for groupby
    tmp = pd.DataFrame(
        {
            "ym": ym.values,
            "is_zero": is_zero.values.astype(float),
            "turnover": turnover.values if has_shrout else np.nan,
            "row_idx": np.arange(len(df)),
        },
        index=df.index,
    )

    # Aggregate per month: zero_days count, sum_turnover, trading_days count
    grp = tmp.groupby("ym", sort=True)
    monthly_zero = grp["is_zero"].sum()           # days with zero volume
    monthly_turn = grp["turnover"].sum()          # sum of daily turnover
    monthly_ndays = grp["is_zero"].count()        # total trading days
    monthly_last_idx = grp["row_idx"].last()      # last row index of each month

    # Liu (2006) formula:
    #   zerotrade_m = (zero_days_m + sum_turn_m / 48e5) * (21 / ndays_m)
    # When turnover is missing fall back to zero-days-only proxy scaled to 21
    sum_turn_safe = monthly_turn.where(monthly_turn.notna(), 0.0)

    ndays_safe = monthly_ndays.replace(0, np.nan)
    monthly_zt = (monthly_zero + sum_turn_safe / 48e5) * (21.0 / ndays_safe)

    # ------------------------------------------------------------------ #
    # 4.  Rolling 12-month and 3-month averages of monthly statistic       #
    # ------------------------------------------------------------------ #
    # Need at least 3 months for 3m, 6 for 12m (allow partial)
    zt_12m = monthly_zt.rolling(12, min_periods=6).mean()
    zt_3m = monthly_zt.rolling(3, min_periods=2).mean()
    zt_slope = zt_12m - zt_3m  # positive = illiquidity rising (trend up)

    # ------------------------------------------------------------------ #
    # 5.  Map monthly values back to daily rows (carry last month forward) #
    # ------------------------------------------------------------------ #
    # Build a series indexed by last row index of each month
    last_rows = monthly_last_idx.values  # array of integer positions

    # Create arrays filled with NaN, then assign month-end values
    n = len(df)
    arr_12m = np.full(n, np.nan)
    arr_3m = np.full(n, np.nan)
    arr_slope = np.full(n, np.nan)

    zt12_vals = zt_12m.values
    zt3_vals = zt_3m.values
    zts_vals = zt_slope.values

    for i, row_pos in enumerate(last_rows):
        arr_12m[row_pos] = zt12_vals[i]
        arr_3m[row_pos] = zt3_vals[i]
        arr_slope[row_pos] = zts_vals[i]

    # Forward-fill within each future day (carry month-end to next month end)
    s12 = pd.Series(arr_12m, index=df.index).ffill()
    s3 = pd.Series(arr_3m, index=df.index).ffill()
    ss = pd.Series(arr_slope, index=df.index).ffill()

    # ------------------------------------------------------------------ #
    # 6.  Assign to df (replace inf with nan)                              #
    # ------------------------------------------------------------------ #
    df["osap_zerotrade12m_lvl"] = s12.replace([np.inf, -np.inf], np.nan).values
    df["osap_zerotrade12m_3m"] = s3.replace([np.inf, -np.inf], np.nan).values
    df["osap_zerotrade12m_slope"] = ss.replace([np.inf, -np.inf], np.nan).values

    # Drop scratch fundamentals column if added
    if has_shrout and "fund_shares_outstanding" in df.columns:
        df = df.drop(columns=["fund_shares_outstanding"])

    return df
