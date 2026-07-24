"""
_cand_ff0703e_cross_asset_dvix_signed_beta_convexity.py

Convexity/asymmetry of a stock's return response to rising vs. falling VIX.

METHOD (per spec ff0703e_cross_asset_dvix_signed_beta_convexity):
    r_t     = Close.pct_change()
    dVIX_t  = vix_close.diff()
    Over a trailing 90-trading-day window, run two *conditional* regressions
    using rolling (subset-restricted) means:
        beta_up = slope of r_t on dVIX_t restricted to days where dVIX_t > 0
        beta_dn = slope of r_t on dVIX_t restricted to days where dVIX_t < 0
    slope = sum((dVIX-mean)(r-mean)) / sum((dVIX-mean)^2)   over that subset only
    (guard denom > 0, require >= 12 observations per side else NaN)

    LEVEL column      = beta_up   (return sensitivity to worsening vol)
    ASYMMETRY column  = beta_up - beta_dn  (convexity: large negative ->
                         stock is punished by vol spikes far more than it is
                         rewarded by vol relief)

IMPLEMENTATION NOTE (vectorized, causal):
    The conditional (subset) mean/covariance sums are computed as rolling
    sums of *masked* series (values zeroed outside the subset, mask itself
    rolling-summed for the subset count n). This is algebraically identical
    to computing the regression only over the in-subset days within each
    trailing window, but runs as O(n) pandas rolling ops (no python loop).
    All rolling windows are trailing (causal) -- no negative shifts, no
    forward fill of future information.

VIX join uses merge_asof(direction="backward") on Date, which is lookahead
safe (only ever pulls the most recent VIX print as-of that day).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

WINDOW = 90
MIN_SIDE_OBS = 12

COL_LEVEL = "ff0703e_cross_asset_dvix_signed_beta_convexity_up"
COL_ASYM = "ff0703e_cross_asset_dvix_signed_beta_convexity_asym"

METADATA = {
    "name": "ff0703e_cross_asset_dvix_signed_beta_convexity",
    "description": (
        "Convexity/asymmetry of a stock's daily-return response to rising vs. "
        "falling VIX. Over a trailing 90-day window, runs two subset-conditional "
        "regressions of return on the day-over-day VIX change (dVIX): beta_up "
        "uses only days where dVIX>0 (worsening vol), beta_dn uses only days "
        "where dVIX<0 (vol relief). LEVEL=beta_up (sensitivity to vol spikes). "
        "ASYMMETRY=beta_up-beta_dn (large negative => punished by vol spikes "
        "far more than rewarded by vol relief, i.e. convex downside vol "
        "exposure). Faithful per-ticker implementation of the spec; each side "
        "requires >=12 in-window observations and a strictly positive "
        "variance denominator else NaN."
    ),
    "requires": ["Close"],
    "produces": [COL_LEVEL, COL_ASYM],
    "tags": ["cross_asset", "volatility", "beta", "convexity", "vix"],
    "version": "1.0",
    "author": "feature-factory (ff0703e codegen)",
}


def _rolling_conditional_slope(x: pd.Series, y: pd.Series, mask: pd.Series, window: int, min_obs: int) -> pd.Series:
    """
    Rolling slope of y on x restricted to rows where `mask` is True, using
    only trailing (causal) rolling sums of masked series. Returns NaN where
    the in-window subset has < min_obs observations or the variance
    denominator is not strictly positive.
    """
    mask_f = mask.fillna(False).astype(float)
    x_m = x.where(mask, 0.0).fillna(0.0)
    y_m = y.where(mask, 0.0).fillna(0.0)

    n = mask_f.rolling(window, min_periods=1).sum()
    sum_x = x_m.rolling(window, min_periods=1).sum()
    sum_y = y_m.rolling(window, min_periods=1).sum()
    sum_xy = (x_m * y_m).rolling(window, min_periods=1).sum()
    sum_x2 = (x_m * x_m).rolling(window, min_periods=1).sum()

    n_safe = n.where(n > 0, np.nan)
    numerator = sum_xy - (sum_x * sum_y) / n_safe
    denominator = sum_x2 - (sum_x * sum_x) / n_safe

    slope = numerator / denominator.where(denominator > 0, np.nan)
    slope = slope.where(n >= min_obs, np.nan)
    slope = slope.replace([np.inf, -np.inf], np.nan)
    return slope


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df[COL_LEVEL] = np.nan
    df[COL_ASYM] = np.nan

    n_rows = len(df)
    if n_rows == 0 or "Close" not in df.columns:
        return df

    close = pd.to_numeric(df["Close"], errors="coerce")
    r = close.pct_change()

    vix_daily = _indexes.vix_daily_close()
    if vix_daily.empty or "Date" not in df.columns:
        return df

    # df is contractually ascending-by-Date for a single ticker, so a direct
    # merge_asof (no re-sort needed) preserves row order 1:1.
    work = pd.DataFrame({"Date": pd.to_datetime(df["Date"]).values})
    merged = pd.merge_asof(work, vix_daily.sort_values("Date"), on="Date", direction="backward")

    vix_close = pd.Series(merged["vix_close"].to_numpy(), index=df.index)
    dvix = vix_close.diff()

    r_idx = pd.Series(r.to_numpy(), index=df.index)

    mask_up = dvix > 0
    mask_dn = dvix < 0

    beta_up = _rolling_conditional_slope(dvix, r_idx, mask_up, WINDOW, MIN_SIDE_OBS)
    beta_dn = _rolling_conditional_slope(dvix, r_idx, mask_dn, WINDOW, MIN_SIDE_OBS)

    df[COL_LEVEL] = beta_up.to_numpy()
    df[COL_ASYM] = (beta_up - beta_dn).to_numpy()

    return df
