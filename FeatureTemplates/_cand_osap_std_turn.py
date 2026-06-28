"""
osap_std_turn — Standard Deviation of Share Turnover
Source: OpenSourceAP (Chen-Zimmermann); Chordia, Subrahmanyam & Anshuman (2001)

Economic signal: stocks with HIGH variability in daily/monthly turnover earn LOWER
future returns (predicted sign: -1). The intuition is that erratic trading activity
proxies for investor disagreement, attention-driven speculation, or liquidity risk.

Per-ticker proxy: The canonical measure uses monthly volume / shares_outstanding over
a 12-month window. Here we use DAILY turnover (Volume / shares_outstanding) pulled from
PIT fundamentals (fund_shares_outstanding) when available, with a fallback to
Volume / rolling_median_volume as a unit-free turnover proxy when fundamentals are
missing. We then compute the rolling 252-day std of daily turnover and a 21-day
short-term variant. A trend column (short/long ratio) captures whether turnover
variability is rising or falling.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (optional — gracefully degrades if unavailable)
# ---------------------------------------------------------------------------
try:
    _s2 = _ilu.spec_from_file_location(
        "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
    )
    _fundamentals = _ilu.module_from_spec(_s2)
    _s2.loader.exec_module(_fundamentals)
    _HAS_FUND = True
except Exception:
    _HAS_FUND = False

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_std_turn",
    "description": (
        "Standard deviation of share turnover (Chordia, Subrahmanyam & Anshuman 2001). "
        "High turnover volatility predicts lower future returns (predicted sign: -1). "
        "Per-ticker proxy: daily turnover = Volume / shares_outstanding (from PIT "
        "fundamentals when available; falls back to Volume / rolling-median-volume when "
        "fundamentals are absent). Outputs: 252-day rolling std of daily turnover "
        "(long-window), 21-day rolling std (short-window), and their ratio as a trend "
        "signal. Cross-sectional ranking is not performed — this is a within-ticker "
        "time-series measure; cross-sectional ranking is applied externally."
    ),
    "requires": ["Volume", "Close"],
    "produces": [
        "osap_std_turn_252d",   # long-window std of daily turnover (primary signal)
        "osap_std_turn_21d",    # short-window std (recent variability)
        "osap_std_turn_ratio",  # short/long ratio: >1 = variability rising
    ],
    "tags": ["volume", "turnover", "liquidity", "osap", "trading_activity"],
    "version": "1.0",
    "author": "OpenSourceAP (Chen-Zimmermann); Chordia, Subrahmanyam & Anshuman (2001)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute osap_std_turn features for a single ticker (ascending Date).

    Steps:
      1. Fetch shares_outstanding from PIT fundamentals if available.
      2. Compute daily turnover = Volume / shares_outstanding (or fallback proxy).
      3. Rolling std over 252d and 21d windows.
      4. Ratio of short to long std as a trend indicator.
    """
    # --- Step 1: shares outstanding (PIT-safe) ---
    shr_out = None
    if _HAS_FUND:
        try:
            tmp = _fundamentals.as_of(df, fields=["shares_outstanding"])
            col = "fund_shares_outstanding"
            if col in tmp.columns:
                shr_out = tmp[col].copy()
                # Drop the scratch column (not in produces)
                # (we work on a local copy; df is never mutated to add fund_ cols)
        except Exception:
            shr_out = None

    vol = df["Volume"].copy().astype(float)

    # --- Step 2: daily turnover ---
    if shr_out is not None:
        # PIT shares outstanding in millions (typical Compustat units); scale to shares
        # companyfacts gives actual share count (not in millions) but check magnitude:
        # if median > 1e6 treat as raw shares; otherwise assume millions
        median_shr = shr_out.median()
        if pd.notna(median_shr) and median_shr > 0:
            # Normalize units: if median shares < 1e4, likely in millions
            if median_shr < 1e4:
                shr_out_adj = shr_out * 1_000_000.0
            else:
                shr_out_adj = shr_out
            # Guard division
            shr_safe = shr_out_adj.where(shr_out_adj > 0, other=np.nan)
            turnover = vol / shr_safe
        else:
            # Fall back to proxy
            turnover = _turnover_proxy(vol)
    else:
        # No fundamentals — use volume-normalized proxy
        turnover = _turnover_proxy(vol)

    # Replace inf/-inf with NaN (guard)
    turnover = turnover.replace([np.inf, -np.inf], np.nan)

    # --- Step 3: rolling standard deviations ---
    # min_periods: require at least half the window to emit a value
    std_252 = turnover.rolling(window=252, min_periods=126).std()
    std_21  = turnover.rolling(window=21,  min_periods=10).std()

    # --- Step 4: ratio (short / long) ---
    std_long_safe = std_252.where(std_252 > 0, other=np.nan)
    ratio = std_21 / std_long_safe
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_std_turn_252d"] = std_252
    df["osap_std_turn_21d"]  = std_21
    df["osap_std_turn_ratio"] = ratio

    return df


def _turnover_proxy(vol: pd.Series) -> pd.Series:
    """
    Fallback when shares_outstanding is unavailable.
    Proxy daily turnover as vol / rolling_median(vol, 252d).
    This is unit-free and captures relative trading intensity over time,
    preserving the variability signal even without a share count denominator.
    """
    roll_med = vol.rolling(window=252, min_periods=63).median()
    roll_med_safe = roll_med.where(roll_med > 0, other=np.nan)
    return vol / roll_med_safe
