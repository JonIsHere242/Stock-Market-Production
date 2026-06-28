"""
osap_volsd — Volume Variance (Chordia, Subra, Anshuman 2001 / Chen-Zimmermann OpenSourceAP)

Rolling standard deviation of monthly trading volume over the past 36 months
(min 24 months required). Predicted sign: -1 (high vol-variance → lower returns).

Per-ticker proxy: monthly volume is approximated by summing daily volumes within
each calendar month. Cross-sectional NYSE-only restriction cannot be enforced
per-ticker; the block produces the raw time-series signal for all tickers and
relies on the downstream pipeline to apply universe screens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_volsd",
    "description": (
        "Rolling standard deviation of monthly trading volume over the trailing 36 months "
        "(min 24 months). Approximates the Chordia-Subra-Anshuman (2001) Volume Variance "
        "anomaly from Chen-Zimmermann OpenSourceAP. Predicted sign: -1 (high variance → "
        "lower future returns). Per-ticker proxy: NYSE-only restriction cannot be applied "
        "inside this block; daily volumes are aggregated to calendar-month totals before "
        "computing the rolling std. Also produces a normalised variant (std / mean) and a "
        "12-month slope of the rolling std to capture trend in liquidity variability."
    ),
    "requires": ["Volume"],
    "produces": [
        "osap_volsd_36m",        # rolling 36-month std of monthly volume (raw)
        "osap_volsd_cv_36m",     # coefficient of variation (std/mean) — size-neutral
        "osap_volsd_slope_12m",  # OLS slope of the last 12 monthly-std observations
    ],
    "tags": ["liquidity", "volume", "volatility", "anomaly", "osap"],
    "version": "1.0",
    "author": "Chordia, Subra, Anshuman (2001) / Chen-Zimmermann OpenSourceAP (osap_volsd spec)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-variance features, one ticker at a time (ascending Date)."""

    # ------------------------------------------------------------------ #
    # 0. Guard: need enough rows and the Volume column
    # ------------------------------------------------------------------ #
    if df.shape[0] < 2 or "Volume" not in df.columns:
        df["osap_volsd_36m"] = np.nan
        df["osap_volsd_cv_36m"] = np.nan
        df["osap_volsd_slope_12m"] = np.nan
        return df

    # ------------------------------------------------------------------ #
    # 1. Build a daily series with a proper DatetimeIndex for resampling
    # ------------------------------------------------------------------ #
    dates = pd.to_datetime(df["Date"])
    vol = df["Volume"].values.astype(np.float64)

    daily = pd.Series(vol, index=dates, name="vol")

    # Replace non-positive or NaN volume with NaN so months with bad data
    # degrade gracefully rather than distorting the std.
    daily = daily.where(daily > 0, np.nan)

    # ------------------------------------------------------------------ #
    # 2. Resample to calendar months (sum of daily volume per month)
    # ------------------------------------------------------------------ #
    monthly_vol = daily.resample("MS").sum(min_count=1)  # NaN if all days missing

    # ------------------------------------------------------------------ #
    # 3. Rolling 36-month std (min 24 months)
    # ------------------------------------------------------------------ #
    WINDOW = 36
    MIN_OBS = 24

    roll_std = monthly_vol.rolling(window=WINDOW, min_periods=MIN_OBS).std()
    roll_mean = monthly_vol.rolling(window=WINDOW, min_periods=MIN_OBS).mean()

    # Coefficient of variation — normalises by scale so tickers with different
    # average volumes are comparable (closer to the cross-sectional signal).
    roll_cv = roll_std / roll_mean.replace(0, np.nan)

    # ------------------------------------------------------------------ #
    # 4. 12-month OLS slope of the rolling std series
    #    Captures whether liquidity variability is trending up or down.
    # ------------------------------------------------------------------ #
    SLOPE_WIN = 12
    MIN_SLOPE = 8

    std_arr = roll_std.values
    n_months = len(std_arr)

    slope_vals = np.full(n_months, np.nan)

    if n_months >= MIN_SLOPE:
        x_base = np.arange(SLOPE_WIN, dtype=np.float64)
        x_base_c = x_base - x_base.mean()
        ss_xx = (x_base_c ** 2).sum()

        for i in range(SLOPE_WIN - 1, n_months):
            window_data = std_arr[i - SLOPE_WIN + 1: i + 1]
            valid = ~np.isnan(window_data)
            if valid.sum() < MIN_SLOPE:
                continue
            y = window_data[valid]
            x = x_base_c[valid]
            ss_xy = (x * (y - y.mean())).sum()
            ss_xx_v = (x ** 2).sum()
            if ss_xx_v == 0:
                continue
            slope_vals[i] = ss_xy / ss_xx_v

    roll_slope = pd.Series(slope_vals, index=monthly_vol.index)

    # ------------------------------------------------------------------ #
    # 5. Map monthly values back to daily rows (backward-fill within month)
    # ------------------------------------------------------------------ #
    # Each daily row gets the monthly value computed from the month that
    # ENDS before or at that day (merge_asof direction="backward" on the
    # month-start index).  This is lookahead-safe: a month's std is only
    # available after the month closes, so we shift the monthly index
    # forward by one month before merging.
    #
    # Concretely: the std computed from months [t-36, t-1] is first
    # observable on the first trading day of month t+1 (after month t
    # closes). We implement this by using "forward=1" shift on the monthly
    # index so daily rows in month t see the std from months [t-37, t-2].
    # This is conservative (one month stale) but strictly leakage-free.

    # Shift monthly index forward by 1 period (make available next month)
    available_idx = monthly_vol.index + pd.DateOffset(months=1)

    monthly_std_avail = pd.Series(
        roll_std.values, index=available_idx, name="osap_volsd_36m"
    )
    monthly_cv_avail = pd.Series(
        roll_cv.values, index=available_idx, name="osap_volsd_cv_36m"
    )
    monthly_slope_avail = pd.Series(
        roll_slope.values, index=available_idx, name="osap_volsd_slope_12m"
    )

    # Convert to DataFrames for merge_asof
    df_std = monthly_std_avail.reset_index().rename(columns={"index": "Date"})
    df_cv = monthly_cv_avail.reset_index().rename(columns={"index": "Date"})
    df_slope = monthly_slope_avail.reset_index().rename(columns={"index": "Date"})

    # Build a helper frame with daily dates
    daily_dates = pd.DataFrame({"Date": dates.values})

    out_std = pd.merge_asof(
        daily_dates.sort_values("Date"),
        df_std.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    out_cv = pd.merge_asof(
        daily_dates.sort_values("Date"),
        df_cv.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    out_slope = pd.merge_asof(
        daily_dates.sort_values("Date"),
        df_slope.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Re-align to original df index order (df is already ascending, daily_dates too)
    df["osap_volsd_36m"] = out_std["osap_volsd_36m"].values
    df["osap_volsd_cv_36m"] = out_cv["osap_volsd_cv_36m"].values
    df["osap_volsd_slope_12m"] = out_slope["osap_volsd_slope_12m"].values

    # Ensure no inf values leak through
    for col in ["osap_volsd_36m", "osap_volsd_cv_36m", "osap_volsd_slope_12m"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
