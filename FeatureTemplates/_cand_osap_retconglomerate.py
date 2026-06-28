"""
osap_retconglomerate — Conglomerate of return-predictive signals (per-ticker proxy).

The OSAP "RetConglomerate" anomaly aggregates multiple return-predictive signals
(medium-term momentum, short-term reversal, long-run reversal, and a volume-trend
interaction) into a single composite score. The original cross-sectional construction
ranks each component across stocks and averages the ranks; here we implement faithful
per-ticker time-series proxies for each component, then form an equal-weight composite.

Components (all computed from OHLCV, no lookahead):
  1. mom_11_1   : 11-month cumulative return skipping the last month (classic momentum)
  2. st_rev     : 1-month reversal (negative sign — reversal predicts next month negatively)
  3. lr_rev     : 3-year to 1-year reversal (long-run reversal / contrarian)
  4. vol_trend  : volume trend interaction — expanding volume with price trend amplifies signal

The composite (osap_retconglomerate_score) is the equal-weight Z-score average of the
four components, each internally Z-scored over a trailing 252-day window to make them
comparable. A slope variant captures the rate-of-change of the composite.

Per-ticker proxy note: the original ranks components cross-sectionally each period;
here we rank each component through time using rolling Z-scores (same economic
direction, different normalisation). Signal direction should be preserved.

Source: Open Source Asset Pricing (OSAP) project, Chen & Zimmermann (2022),
"Open Source Cross-Sectional Asset Pricing", Critical Finance Review.
Authors: Andrew Y. Chen, Tom Zimmermann.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "osap_retconglomerate",
    "description": (
        "Conglomerate of four return-predictive signals (medium-term momentum, "
        "short-term reversal, long-run reversal, volume-trend interaction), "
        "each Z-scored over trailing 252 days and equal-weighted into a composite. "
        "Per-ticker time-series proxy for the OSAP cross-sectional RetConglomerate anomaly "
        "(Chen & Zimmermann 2022). Produces composite score and its 21-day slope."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_retconglomerate_score",
        "osap_retconglomerate_slope",
        "osap_retconglomerate_mom",
    ],
    "tags": ["momentum", "reversal", "composite", "osap", "return_predictors"],
    "version": "1.0.0",
    "author": "Chen & Zimmermann (2022) 'Open Source Cross-Sectional Asset Pricing', Critical Finance Review. Per-ticker proxy implementation.",
}


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score; returns NaN where std == 0 or window not yet filled."""
    roll = s.rolling(window, min_periods=max(window // 2, 20))
    mu = roll.mean()
    sd = roll.std(ddof=1)
    sd = sd.replace(0, np.nan)
    return (s - mu) / sd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].copy()
    volume = df["Volume"].copy()

    # -----------------------------------------------------------------------
    # 1. Medium-term momentum (mom_11_1): cumulative return from t-252 to t-21
    #    (approx 11 months skipping last 1 month on daily bars)
    # -----------------------------------------------------------------------
    ret_252 = close / close.shift(252) - 1.0   # 252-day raw return
    ret_21  = close / close.shift(21)  - 1.0   # last-month return (to skip)
    # mom_11_1 = compound return from -252 to -21
    # = (1 + ret_252) / (1 + ret_21) - 1  (lookahead-free: both use past prices)
    denom_mom = (1.0 + ret_21).replace(0, np.nan)
    mom_11_1  = (1.0 + ret_252) / denom_mom - 1.0

    # -----------------------------------------------------------------------
    # 2. Short-term reversal (st_rev): negative 1-month return (reversal)
    # -----------------------------------------------------------------------
    st_rev = -(ret_21)  # negated so positive = expected up next period

    # -----------------------------------------------------------------------
    # 3. Long-run reversal (lr_rev): negative of 3yr-to-1yr cumulative return
    #    (contrarian; stocks that rose a lot 1-3 years ago tend to revert)
    # -----------------------------------------------------------------------
    ret_756 = close / close.shift(756) - 1.0   # ~3 years
    ret_252_skip = ret_252                      # 1-year already computed above
    denom_lr = (1.0 + ret_252_skip).replace(0, np.nan)
    lr_raw   = (1.0 + ret_756) / denom_lr - 1.0  # excess of 3yr over 1yr
    lr_rev   = -lr_raw  # reversal = negative sign

    # -----------------------------------------------------------------------
    # 4. Volume-trend interaction: trending price amplified by rising volume
    #    = mom_11_1 * sign(volume trend over 63d)
    #    Volume trend: current 21d avg volume vs 63d avg volume
    # -----------------------------------------------------------------------
    vol_21  = volume.rolling(21,  min_periods=10).mean()
    vol_63  = volume.rolling(63,  min_periods=30).mean()
    vol_ratio = (vol_21 / vol_63.replace(0, np.nan)) - 1.0  # >0 = rising volume
    vol_trend_interaction = mom_11_1 * vol_ratio  # amplified momentum

    # -----------------------------------------------------------------------
    # Z-score each component over 252-day rolling window, then average
    # -----------------------------------------------------------------------
    z_mom  = _rolling_zscore(mom_11_1,             252)
    z_str  = _rolling_zscore(st_rev,               252)
    z_lr   = _rolling_zscore(lr_rev,               252)
    z_vt   = _rolling_zscore(vol_trend_interaction, 252)

    # Equal-weight composite (nanmean so partial availability is ok)
    composite_arr = np.nanmean(
        np.stack([z_mom.values, z_str.values, z_lr.values, z_vt.values], axis=1),
        axis=1,
    )
    # Mark as NaN where ALL four components are NaN (true missing)
    all_nan_mask = (
        np.isnan(z_mom.values)
        & np.isnan(z_str.values)
        & np.isnan(z_lr.values)
        & np.isnan(z_vt.values)
    )
    composite_arr[all_nan_mask] = np.nan

    composite = pd.Series(composite_arr, index=df.index)

    # -----------------------------------------------------------------------
    # Slope: 21-day linear slope of composite (captures acceleration)
    # -----------------------------------------------------------------------
    win = 21
    min_pts = 10
    n = len(df)
    slope_arr = np.full(n, np.nan)
    comp_vals = composite.values
    x = np.arange(win, dtype=float)
    x_mean = x.mean()
    x_denom = ((x - x_mean) ** 2).sum()
    if x_denom > 0:
        for i in range(win - 1, n):
            chunk = comp_vals[i - win + 1: i + 1]
            valid = ~np.isnan(chunk)
            if valid.sum() >= min_pts:
                xi = x[valid]
                yi = chunk[valid]
                xi_mean = xi.mean()
                xi_denom = ((xi - xi_mean) ** 2).sum()
                if xi_denom > 0:
                    slope_arr[i] = ((xi - xi_mean) * (yi - yi.mean())).sum() / xi_denom

    slope = pd.Series(slope_arr, index=df.index)

    # -----------------------------------------------------------------------
    # Assign outputs — only the three declared columns
    # -----------------------------------------------------------------------
    df["osap_retconglomerate_score"] = composite
    df["osap_retconglomerate_slope"] = slope
    df["osap_retconglomerate_mom"]   = z_mom   # expose medium-term momentum z-score

    # Guard: replace inf/-inf
    for col in ["osap_retconglomerate_score", "osap_retconglomerate_slope", "osap_retconglomerate_mom"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
