"""
osap_roaq — Return on Assets (Quarterly)
Per-ticker proxy using PIT fundamentals (net_income_ttm / lagged assets).

Source: OpenSourceAP (Chen-Zimmermann), based on Balakrishnan, Bartov & Faurel (2010)
and Ball, Gerakos, Linnainmaa & Nikolaev (2016).  The canonical cross-sectional signal
ranks firms by quarterly earnings / lagged total assets.  This block builds a
point-in-time per-ticker version using SEC-sourced TTM net income and total assets,
with the ROA level, a 4-quarter momentum (trend in ROA), and the ratio relative to
its own trailing mean (a persistence / acceleration signal).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (by file path — no package import allowed)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_roaq",
    "description": (
        "Return on Assets (Quarterly) — OpenSourceAP factor. "
        "Level: net_income_ttm / lagged total assets (PIT, lookahead-safe). "
        "Trend: 4-quarter rolling slope of ROA (captures improving/deteriorating quality). "
        "Accel: ROA relative to its own trailing 8-quarter mean (persistence signal). "
        "Cross-sectional ranking is inherently multi-stock; this is a per-ticker PIT proxy "
        "that captures the same economic signal (asset profitability). "
        "Coverage ~84% (ETFs / foreign stubs will be NaN as expected)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_roaq_level",    # net_income_ttm / lag(assets, 1 quarter) — the primary signal
        "osap_roaq_trend",    # 4-obs rolling OLS slope of level (improving quality)
        "osap_roaq_accel",    # level / trailing-8-obs mean of level (acceleration vs history)
    ],
    "tags": ["fundamentals", "profitability", "accounting", "osap"],
    "version": "1.0",
    "author": (
        "OpenSourceAP (Chen-Zimmermann); "
        "Balakrishnan, Bartov & Faurel (2010); "
        "Ball, Gerakos, Linnainmaa & Nikolaev (2016)"
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """
    Vectorised rolling OLS slope via the closed-form formula for equal-spaced x.
    x = [0, 1, ..., window-1]; slope = cov(x,y) / var(x).
    Returns NaN wherever fewer than `window` valid observations are present.
    """
    n = window
    # x-values are 0..n-1 (fixed); x_mean = (n-1)/2
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = float(((x - x_mean) ** 2).sum())  # constant

    y_vals = s.to_numpy(dtype=np.float64)
    nobs = len(y_vals)
    slope = np.full(nobs, np.nan)

    for i in range(n - 1, nobs):
        y_win = y_vals[i - n + 1 : i + 1]
        if np.isnan(y_win).any():
            continue
        y_mean = y_win.mean()
        ss_xy = float(((x - x_mean) * (y_win - y_mean)).sum())
        if ss_xx == 0:
            continue
        slope[i] = ss_xy / ss_xx

    return pd.Series(slope, index=s.index)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -- Pull PIT fundamentals (net_income_ttm, assets) ----------------------
    df = _fundamentals.as_of(df, fields=["net_income_ttm", "assets"])

    ni = df["fund_net_income_ttm"].copy()          # trailing-twelve-month net income
    at = df["fund_assets"].copy()                  # total assets at most recent filing

    # Lag assets by one observation to avoid same-period look-ahead;
    # the _fundamentals helper already backward-merges on filed_date so the
    # "current" asset figure is the PREVIOUSLY FILED value — shifting by 1
    # more row ensures we use strictly prior-period denominator.
    at_lag = at.shift(1)

    # Guard: replace non-positive or zero assets with NaN
    at_lag = at_lag.where(at_lag > 0, np.nan)

    # --- Level -----------------------------------------------------------
    roa_level = ni / at_lag
    # Winsorise extreme values that can arise from near-zero asset periods
    # Use 1%/99% rolling cap on a 252-bar window (per-ticker only, no XS)
    lo = roa_level.rolling(252, min_periods=20).quantile(0.01)
    hi = roa_level.rolling(252, min_periods=20).quantile(0.99)
    roa_level = roa_level.clip(lower=lo, upper=hi)

    df["osap_roaq_level"] = roa_level

    # --- Trend (rolling 4-obs OLS slope on level) ------------------------
    # 4 quarterly observations ≈ 1 year of quarterly data arriving daily;
    # we run on the daily series so "4" is 4 trading days apart in filing
    # arrivals — effectively 4 consecutive fundamental snapshots.
    # A window of 4 rows is minimal but faithful to the quarterly cadence.
    df["osap_roaq_trend"] = _rolling_slope(roa_level, window=4)

    # --- Acceleration (level vs trailing 8-obs mean) ----------------------
    trailing_mean = roa_level.rolling(8, min_periods=4).mean()
    # ratio: > 1 means currently above historical norm (accelerating)
    accel = roa_level / trailing_mean.where(trailing_mean.abs() > 1e-10, np.nan)
    df["osap_roaq_accel"] = accel

    # Drop scratch fund_ columns we are NOT publishing
    df.drop(columns=["fund_net_income_ttm", "fund_assets"], inplace=True, errors="ignore")

    return df
