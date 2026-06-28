"""
Feature block: osap_voltrend
Rolling volume trend coefficient (Haugen & Baker 1996), adapted from OpenSourceAP (Chen-Zimmermann).

Per the spec: rolling OLS coefficient from regressing monthly trading volume on a linear time
trend over 60-month windows (min 30 months), scaled by the 60-month average monthly volume.
Predicted sign: -1 (stocks with rising volume trend tend to underperform).

Implementation note: the spec is inherently monthly-frequency and cross-sectional in the original
paper; here we aggregate daily Volume to monthly totals per ticker, compute the rolling 60-month
OLS slope divided by the 60-month average monthly volume, then forward-fill back to daily frequency
(using the most recently completed month-end estimate at each daily row). This is lookahead-safe
because month-end values are only available after the month closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_voltrend",
    "description": (
        "Rolling volume trend: OLS slope from regressing monthly volume on a linear time index "
        "over a 60-month window (min 30 months), scaled by 60-month mean monthly volume. "
        "Predicted sign: -1 (rising volume trend predicts underperformance). "
        "Spec is inherently monthly & cross-sectional (Haugen & Baker 1996); implemented as a "
        "per-ticker rolling monthly OLS proxy, forward-filled to daily frequency from month-end."
    ),
    "requires": ["Volume"],
    "produces": ["osap_voltrend_coef", "osap_voltrend_zscore"],
    "tags": ["volume", "trend", "osap", "haugen_baker"],
    "version": "1.0",
    "author": "OpenSourceAP / Chen-Zimmermann; original: Haugen and Baker (1996)",
}


def _rolling_ols_slope_scaled(monthly_vol: np.ndarray) -> float:
    """OLS slope of volume ~ time_index, scaled by mean volume. Returns NaN if insufficient data."""
    n = len(monthly_vol)
    if n < 30:
        return np.nan
    # Mask out NaN rows
    mask = np.isfinite(monthly_vol)
    y = monthly_vol[mask]
    if y.sum() == 0:
        return np.nan
    k = mask.sum()
    if k < 30:
        return np.nan
    # Build time index for non-NaN positions (use their original positions within window)
    x = np.where(mask)[0].astype(float)
    # OLS: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    n_k = float(k)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n_k * sxx - sx * sx
    if denom == 0.0:
        return np.nan
    slope = (n_k * sxy - sx * sy) / denom
    mean_vol = sy / n_k
    if mean_vol == 0.0 or not np.isfinite(mean_vol):
        return np.nan
    return slope / mean_vol


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] < 2:
        df["osap_voltrend_coef"] = np.nan
        df["osap_voltrend_zscore"] = np.nan
        return df

    # Ensure Date is datetime for resampling
    dates = pd.to_datetime(df["Date"])
    vol = df["Volume"].values.astype(float)

    # Build a dated Series of daily volume
    daily_series = pd.Series(vol, index=dates, name="vol")

    # Aggregate to monthly totals; label at month-end (right boundary = period end)
    # Use ME (month-end) resample anchor
    try:
        monthly = daily_series.resample("ME").sum()
    except Exception:
        # Fallback for older pandas
        monthly = daily_series.resample("M").sum()

    # Replace 0-volume months with NaN (data gap / halt)
    monthly = monthly.replace(0.0, np.nan)

    n_months = len(monthly)
    if n_months < 30:
        df["osap_voltrend_coef"] = np.nan
        df["osap_voltrend_zscore"] = np.nan
        return df

    monthly_vals = monthly.values
    monthly_idx = monthly.index  # DatetimeIndex of month-end dates

    WINDOW = 60
    MIN_PERIODS = 30

    # Compute rolling OLS slope for each month-end
    coefs = np.full(n_months, np.nan)
    for i in range(n_months):
        start = max(0, i - WINDOW + 1)
        window_data = monthly_vals[start: i + 1]
        if len(window_data) >= MIN_PERIODS:
            coefs[i] = _rolling_ols_slope_scaled(window_data)

    coef_series = pd.Series(coefs, index=monthly_idx, name="osap_voltrend_coef")

    # Rolling z-score of coef over trailing 60 months (standardise the signal)
    roll_mean = coef_series.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    roll_std = coef_series.rolling(WINDOW, min_periods=MIN_PERIODS).std()
    zscore_series = (coef_series - roll_mean) / roll_std.replace(0.0, np.nan)
    zscore_series.name = "osap_voltrend_zscore"

    # Forward-fill monthly values back to daily:
    # For each daily row, use the most recently completed month-end estimate.
    # We merge_asof on Date (direction=backward) so future month-ends never leak.
    monthly_df = pd.DataFrame({
        "Date": monthly_idx,
        "osap_voltrend_coef": coef_series.values,
        "osap_voltrend_zscore": zscore_series.values,
    })
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])

    daily_dates_df = pd.DataFrame({"Date": dates})
    daily_dates_df["Date"] = pd.to_datetime(daily_dates_df["Date"])

    merged = pd.merge_asof(
        daily_dates_df.sort_values("Date"),
        monthly_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original row order
    merged.index = daily_dates_df.sort_values("Date").index
    merged = merged.reindex(df.index)

    df["osap_voltrend_coef"] = merged["osap_voltrend_coef"].values
    df["osap_voltrend_zscore"] = merged["osap_voltrend_zscore"].values

    return df
