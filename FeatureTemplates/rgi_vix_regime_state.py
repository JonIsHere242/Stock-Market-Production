"""
rgi_vix_regime_state.py  --  VIX risk-regime STATE features (self-gated).

These describe the *current systemic-risk environment* using the VIX index,
distinct from the existing vix_/vix_regime_ blocks:
  - Own-history percentile rank over 120d and 250d windows (this block gates VIX
    on ITS OWN distribution, not the 30/60/252d windows the legacy block uses,
    and excludes the current bar from the ranking set so it is strictly past-only).
  - 5d and 20d VIX *log* change + the acceleration (change-of-change) of the 20d.
  - A "term-structure proxy": VIX level vs its own 20d EMA (>1 == backwardation-
    like stress build-up, <1 == calming).  EMA-based, distinct from the legacy
    SMA z-scores.
  - Binary HIGH-RISK flag (VIX above its own 80th percentile over 250d) and a
    LOW-RISK flag (below 20th percentile).  These flags are reused by the
    interaction block but are emitted here as standalone regime state.

Every VIX value is taken at/before the row's Date via a backward merge_asof, so
the features are strictly look-ahead safe.  df row order is never touched.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Load the underscore-prefixed shared index helper by file path.
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "rgi_vix_regime_state",
    "description": "VIX risk-regime state: own-history 120/250d percentile, 5/20d change + acceleration, EMA term proxy, high/low-risk flags.",
    "requires":    ["Date"],
    "produces":    [
        "rgi_vix_pctl_120",
        "rgi_vix_pctl_250",
        "rgi_vix_chg_5",
        "rgi_vix_chg_20",
        "rgi_vix_accel_20",
        "rgi_vix_term_ema20",
        "rgi_vix_high_flag",
        "rgi_vix_low_flag",
    ],
    "tags":        ["market_regime", "volatility", "rgi"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def _aligned_vix(df: pd.DataFrame) -> pd.Series:
    """Return VIX close aligned 1:1 to df rows via backward merge_asof (past-only)."""
    vix_daily = _indexes.vix_daily_close()  # ['Date','vix_close'], ascending
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    if vix_daily.empty:
        return pd.Series(np.nan, index=df.index)
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    return pd.Series(merged["vix_close"].ffill().to_numpy(), index=df.index)


def _trailing_pctl(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rank of the current value within the PRIOR `window` values (exclude self).

    Uses a window+1 rolling apply on raw arrays: compares the last element to the
    preceding `window` elements. Fraction <= current => [0,1]. Strictly past-only
    because only prior bars form the comparison set.
    """
    def _rank(arr):
        if len(arr) < 2:
            return np.nan
        hist = arr[:-1]
        cur = arr[-1]
        return float((hist <= cur).mean())
    return s.rolling(window + 1, min_periods=min_periods + 1).apply(_rank, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    vc = _aligned_vix(df)

    new = {}

    # Own-history percentile rank (gates VIX on its own distribution).
    new["rgi_vix_pctl_120"] = _trailing_pctl(vc, 120, 40)
    new["rgi_vix_pctl_250"] = _trailing_pctl(vc, 250, 60)

    # Log changes (symmetric, robust to level) over 5 and 20 trading days.
    log_vix = np.log(vc.clip(lower=1e-6))
    new["rgi_vix_chg_5"] = (log_vix - log_vix.shift(5)).replace([np.inf, -np.inf], np.nan)
    chg_20 = (log_vix - log_vix.shift(20)).replace([np.inf, -np.inf], np.nan)
    new["rgi_vix_chg_20"] = chg_20
    # Acceleration: change-of-change of the 20d log change.
    new["rgi_vix_accel_20"] = chg_20.diff(5)

    # Term-structure proxy: VIX level vs its own 20d EMA.  >1 == stress building.
    ema20 = vc.ewm(span=20, min_periods=10, adjust=False).mean()
    new["rgi_vix_term_ema20"] = (vc / ema20.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).clip(0.2, 5.0)

    # Regime flags from the 250d own-history percentile.
    p250 = new["rgi_vix_pctl_250"]
    new["rgi_vix_high_flag"] = pd.Series(
        np.where(p250.notna(), (p250 >= 0.80).astype(float), np.nan), index=df.index
    )
    new["rgi_vix_low_flag"] = pd.Series(
        np.where(p250.notna(), (p250 <= 0.20).astype(float), np.nan), index=df.index
    )

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
