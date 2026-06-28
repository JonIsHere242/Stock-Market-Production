"""
Off-Season Momentum Reversal (Years 6-10) — Heston & Sadka (2008)
SPEC ID: osap_momoffseason06yrplus

Per-ticker proxy: for each trading day, look back 6–10 calendar years
and compute the average monthly return in months that do NOT share the
same calendar month as the current observation (off-season months).

Cross-sectional note: the original factor is ranked cross-sectionally,
but the raw signal (average off-season return) is computed per-ticker
from price history, so this proxy is faithful. The reversal sign (-1)
is preserved: a high rolling off-season return predicts lower future return.
"""
from __future__ import annotations

import math
import warnings
from typing import List

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momoffseason06yrplus",
    "description": (
        "Per-ticker proxy for Heston & Sadka (2008) off-season reversal (years 6-10). "
        "For each date, computes the average monthly log-return over months 6-10 calendar "
        "years prior that do NOT share the same calendar month as the current date "
        "(i.e., off-season months). Predicted sign: -1 (reversal). "
        "Inherently cross-sectional ranking is not replicated; raw per-ticker signal is used."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_momoffseason06yrplus_avg",   # average off-season monthly return, years 6-10
        "osap_momoffseason06yrplus_cnt",   # count of months used (data-quality / coverage)
        "osap_momoffseason06yrplus_std",   # std-dev of those monthly returns (signal dispersion)
    ],
    "tags": ["momentum", "seasonality", "reversal", "price", "long-horizon"],
    "version": "1.0",
    "author": "Heston and Sadka (2008); OpenSourceAP (Chen-Zimmermann); proxy impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute off-season momentum (years 6-10) per ticker."""

    # Initialise output columns with NaN
    df = df.copy()
    df["osap_momoffseason06yrplus_avg"] = np.nan
    df["osap_momoffseason06yrplus_cnt"] = np.nan
    df["osap_momoffseason06yrplus_std"] = np.nan

    if len(df) < 2:
        return df

    # Work on a copy with DatetimeIndex for resampling
    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.set_index("Date").sort_index()

    if work["Close"].isna().all():
        return df

    # --- Step 1: resample to month-end, compute monthly log-returns ---
    # Use last Close of each calendar month
    monthly_close = work["Close"].resample("ME").last()
    # Drop leading / trailing NaN entries from the monthly series
    monthly_close = monthly_close.dropna()

    if len(monthly_close) < 2:
        return df

    # Monthly log returns: r_t = log(P_t / P_{t-1})
    # Shape: index = month-end dates, values = log return for that month
    monthly_ret = np.log(monthly_close / monthly_close.shift(1))
    # First entry is NaN (no prior close); that is fine

    # Extract the calendar month of each monthly observation
    monthly_month = monthly_close.index.month  # 1..12

    # --- Step 2: for each trading day in df, compute the signal ---
    # We need the calendar month of the current day and the 6-10yr window.
    # To avoid row-by-row Python loops over all ~700 rows, we compute the
    # signal once per calendar month (it changes monthly, not daily), then
    # broadcast back to trading days.

    # Map each trading day date → year-month key
    dates_dt = pd.to_datetime(df["Date"])
    ym_series = dates_dt.dt.to_period("M")

    # Unique year-month periods to evaluate
    unique_yms = ym_series.unique()

    # Build a lookup: period -> (avg, cnt, std)
    signal_map: dict = {}

    for ym in unique_yms:
        current_month = ym.month  # calendar month (1-12)
        # The current month-end date (approximate; used to define the window boundary)
        # Window: from 10 years prior to 6 years prior (inclusive)
        # i.e., rows where month-end date is in [ym - 10yr, ym - 6yr)
        # We define year offsets strictly: 6yr = 72 months, 10yr = 120 months
        # So the lookback window is monthly obs from 120 to 72 months before current ym.

        current_ts = ym.to_timestamp("M")  # end of current month

        # Window boundaries (exclusive on near end, inclusive on far end)
        # far boundary: 10 years (120 months) back
        # near boundary: 6 years (72 months) back  (we exclude [0, 72) months)
        far_boundary = current_ts - pd.DateOffset(years=10)
        near_boundary = current_ts - pd.DateOffset(years=6)

        # Select monthly return observations strictly within the window:
        # far_boundary <= month_end_date < near_boundary
        mask_window = (monthly_ret.index >= far_boundary) & (monthly_ret.index < near_boundary)
        window_ret = monthly_ret[mask_window]

        if window_ret.empty:
            signal_map[ym] = (np.nan, 0, np.nan)
            continue

        # Off-season: exclude months that share the same calendar month as current
        off_season_mask = window_ret.index.month != current_month
        off_season_ret = window_ret[off_season_mask].dropna()

        if off_season_ret.empty:
            signal_map[ym] = (np.nan, 0, np.nan)
            continue

        avg_ret = float(off_season_ret.mean())
        cnt = int(off_season_ret.count())
        std_ret = float(off_season_ret.std(ddof=1)) if cnt > 1 else np.nan

        signal_map[ym] = (avg_ret, cnt, std_ret)

    # --- Step 3: broadcast back to trading days ---
    avg_vals = ym_series.map(lambda p: signal_map.get(p, (np.nan, 0, np.nan))[0])
    cnt_vals = ym_series.map(lambda p: signal_map.get(p, (np.nan, 0, np.nan))[1])
    std_vals = ym_series.map(lambda p: signal_map.get(p, (np.nan, 0, np.nan))[2])

    df["osap_momoffseason06yrplus_avg"] = avg_vals.values
    df["osap_momoffseason06yrplus_cnt"] = cnt_vals.values
    df["osap_momoffseason06yrplus_std"] = std_vals.values

    # Sanitise: replace inf/-inf with NaN (guard)
    for col in ["osap_momoffseason06yrplus_avg", "osap_momoffseason06yrplus_std"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
