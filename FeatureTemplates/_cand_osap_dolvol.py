"""
osap_dolvol — Past trading dollar volume (Brennan, Chordia & Subramanyam 1998).

Predicted sign: -1 (cross-sectional, long low-volume; illiquidity premium).

Per-ticker proxy: the original signal is cross-sectional (ranks stocks by 2-month
lagged dollar volume). We implement the raw per-ticker log dollar volume with a
2-month lag, plus a dynamic slope variant and a rolling z-score — capturing the
same economic signal so a downstream cross-sectional ranker can use it correctly.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_dolvol",
    "description": (
        "Log of two-month lagged average daily dollar volume (Close × Volume), "
        "following Brennan, Chordia & Subramanyam (1998). Predicted sign is -1 "
        "(high dollar-vol stocks earn lower future returns; illiquidity premium). "
        "The original signal is cross-sectional; this block provides the per-ticker "
        "raw log level, a 6-month rolling z-score for within-ticker normalization, "
        "and the change vs 4-month-lagged level to capture momentum in liquidity. "
        "Cross-sectional ranking should be applied downstream."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_dolvol_log",   # log of 2-month lagged avg daily dollar volume
        "osap_dolvol_z",     # 126-day rolling z-score of the log level
        "osap_dolvol_chg",   # log level minus 4-month-lagged log level (liquidity trend)
    ],
    "tags": ["volume", "liquidity", "dollar_volume", "osap", "brennan1998"],
    "version": "1.0",
    "author": "Brennan, Chordia & Subramanyam (1998); OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker dollar-volume features.

    Two-month window = 42 trading days (approx 21 days/month).
    The 42-day rolling mean of (Close * Volume) is then lagged by 42 days
    so that the value available on date t used only data up to t-42.
    """
    # Daily dollar volume
    dv = df["Close"] * df["Volume"]

    # Replace zero or negative dollar volume with NaN to avoid log(0)
    dv = dv.where(dv > 0, other=np.nan)

    # 42-day rolling mean of daily dollar volume (one trading month = 21 days,
    # so two months ≈ 42 days)
    WINDOW_2M = 42
    WINDOW_4M = 84   # 4-month lag for the change variant
    ZWIN     = 126   # 6-month rolling window for z-score

    rolling_2m = dv.rolling(window=WINDOW_2M, min_periods=21)
    avg_dv_2m = rolling_2m.mean()

    # Lag by 42 days so today's value is based on data ending 42 bars ago
    avg_dv_2m_lagged = avg_dv_2m.shift(WINDOW_2M)

    # Log of lagged dollar volume
    log_dv = np.log(avg_dv_2m_lagged)  # np.log(NaN) = NaN, np.log(>0) is fine

    # Rolling z-score of log level (126-day window)
    roll_mean = log_dv.rolling(window=ZWIN, min_periods=42).mean()
    roll_std  = log_dv.rolling(window=ZWIN, min_periods=42).std(ddof=1)
    dv_z = (log_dv - roll_mean) / roll_std.replace(0, np.nan)

    # Change in log level: 2-month lagged minus 4-month lagged
    # (rising = stock becoming more liquid recently)
    avg_dv_4m_lagged = avg_dv_2m.shift(WINDOW_4M)
    log_dv_4m = np.log(avg_dv_4m_lagged)
    dv_chg = log_dv - log_dv_4m

    # Guard: replace inf/-inf with NaN
    df["osap_dolvol_log"]  = log_dv.replace([np.inf, -np.inf], np.nan)
    df["osap_dolvol_z"]    = dv_z.replace([np.inf, -np.inf], np.nan)
    df["osap_dolvol_chg"]  = dv_chg.replace([np.inf, -np.inf], np.nan)

    return df
