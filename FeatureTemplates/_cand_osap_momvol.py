"""
Momentum in High Volume Stocks (per-ticker proxy)
Lee & Swaminathan (2000) / OpenSourceAP (Chen-Zimmermann)

Cross-sectional original: sort stocks into 10 momentum ports and 3 volume-turnover
ports; keep top volume port, assign signal = momentum port rank.

Per-ticker proxy strategy:
  - osap_momvol_mom6m   : 6-month (126-day) price return (the momentum signal)
  - osap_momvol_vol_rel : ratio of recent 6m avg daily volume to trailing 12m avg
                          daily volume -- a self-relative "high volume" indicator
                          (cross-sectional rank of turnover replaced by a within-ticker
                           z-score vs its own history, since shares outstanding are
                           unavailable daily)
  - osap_momvol_signal  : product of the two; positive when both momentum and relative
                          volume are elevated (approximates being in top volume port
                          AND high momentum port)

Requires >= 2 years of data per the original screen; leading rows will be NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_momvol",
    "description": (
        "Per-ticker proxy for Lee & Swaminathan (2000) 'Momentum in High Volume Stocks'. "
        "Original method: independent cross-sectional sort into 10 momentum deciles and "
        "3 volume-turnover terciles; keep top turnover tercile, signal = momentum decile. "
        "Per-ticker implementation: 6-month return (osap_momvol_mom6m) as the momentum "
        "signal; ratio of 6-month rolling avg volume to 12-month rolling avg volume "
        "(osap_momvol_vol_rel) as a within-ticker high-volume proxy (replaces XS turnover "
        "rank); osap_momvol_signal = mom6m * vol_rel, capturing the joint high-momentum / "
        "high-volume regime. Requires ~504 trading days to compute cleanly; leading rows NaN."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_momvol_mom6m",
        "osap_momvol_vol_rel",
        "osap_momvol_signal",
    ],
    "tags": ["momentum", "volume", "cross-sectional-proxy", "lee-swaminathan"],
    "version": "1.0",
    "author": "Lee and Swaminathan 2000 / OpenSourceAP (Chen-Zimmermann); per-ticker proxy implementation",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)
    n = len(df)

    # ------------------------------------------------------------------ #
    # 1. 6-month momentum: return over past 126 trading days (skip t-1   #
    #    to avoid 1-month reversal, consistent with L&S 2000)            #
    # ------------------------------------------------------------------ #
    # We use a 1-day skip (skip most recent day) to align with standard
    # intermediate-horizon momentum construction.
    window_mom = 126   # ~6 months
    skip = 1           # skip 1 day (avoids 1-month reversal)

    mom6m = np.full(n, np.nan)
    start_mom = window_mom + skip  # need at least this many bars
    for i in range(start_mom, n):
        price_now = close[i - skip]          # price 1 day ago
        price_past = close[i - skip - window_mom]  # price 127 days ago
        if price_past > 0:
            mom6m[i] = (price_now / price_past) - 1.0

    # ------------------------------------------------------------------ #
    # 2. Relative volume: 6m avg volume / 12m avg volume                 #
    #    > 1 means current activity elevated vs its own history          #
    # ------------------------------------------------------------------ #
    window_short = 126   # 6 months
    window_long  = 252   # 12 months (minimum for 2-year screen alignment)

    vol_short = pd.Series(volume).rolling(window_short, min_periods=max(1, window_short // 2)).mean().to_numpy()
    vol_long  = pd.Series(volume).rolling(window_long,  min_periods=max(1, window_long  // 2)).mean().to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_rel = np.where(vol_long > 0, vol_short / vol_long, np.nan)
    vol_rel = np.where(np.isinf(vol_rel), np.nan, vol_rel)

    # ------------------------------------------------------------------ #
    # 3. Combined signal: mom6m * vol_rel                                #
    #    High when both momentum AND relative volume are elevated.        #
    # ------------------------------------------------------------------ #
    signal = np.where(
        np.isnan(mom6m) | np.isnan(vol_rel),
        np.nan,
        mom6m * vol_rel,
    )

    df["osap_momvol_mom6m"]  = mom6m
    df["osap_momvol_vol_rel"] = vol_rel
    df["osap_momvol_signal"]  = signal

    return df
