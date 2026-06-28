"""
fvn_valuation_dynamics.py  --  Own-history RANK, valuation MOMENTUM, and a CHEAPNESS COMPOSITE.

Three distinct, non-redundant views of the valuation yields from fvn_valuation_yields, all keeping
the "vs OWN history" framing that strips the static per-name/sector level:

1. PERCENTILE RANK vs own 250d history  (fvn_<tag>_pctl_250)
   Where does today's yield sit in this stock's own trailing-year distribution, in [0,1]?
   Rank is robust to fat tails / non-normality, unlike the z-score (companion block), so it is a
   genuinely different transform -- not a rescaling. 1.0 = cheapest the stock has been all year.

2. VALUATION MOMENTUM = 20d change in the yield  (fvn_<tag>_chg20)
   Is the stock RE-RATING? A rising earnings/fcf yield means the multiple is compressing (getting
   cheaper) faster than fundamentals -- a directional, short-horizon valuation move distinct from
   the level or its rank. Plain difference over 20 trading days.

3. CHEAPNESS COMPOSITE  (fvn_cheapness_composite)
   Average of the standardized (250d-z) yields across the value pillars (earnings, sales, book, fcf,
   cfo). A single orthogonal "is this name cheap vs its own norm across all pillars" score; averaging
   independent pillars cancels single-metric noise. We standardize INSIDE this block (self-contained)
   so it does not depend on the z-score block's columns.

Yields are pulled (as produced columns) from fvn_valuation_yields via `requires`. Guards: std==0 ->
NaN; ranks use min_periods so leading rows are NaN (expected); all outputs clipped finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw yields from fvn_valuation_yields.py (framework runs that block first via `requires`).
_TAG = {
    "fvn_earnings_yield": "ey",
    "fvn_sales_yield":    "sy",
    "fvn_book_yield":     "by",
    "fvn_fcf_yield":      "fy",
    "fvn_cfo_yield":      "cy",
}
_YIELDS = list(_TAG.keys())

_RANK_WIN = 250
_CHG_LAG = 20
_Z_WIN = 250
_MIN_FRAC = 0.5
_CLIP = 6.0

# Pillars folded into the composite (the five value yields above).
_COMPOSITE_PILLARS = list(_YIELDS)

_PRODUCES = (
    [f"fvn_{_TAG[y]}_pctl_{_RANK_WIN}" for y in _YIELDS]
    + [f"fvn_{_TAG[y]}_chg{_CHG_LAG}" for y in _YIELDS]
    + ["fvn_cheapness_composite"]
)

METADATA = {
    "name":        "fvn_valuation_dynamics",
    "description": "Own-history percentile rank (250d), 20d change (valuation momentum), and a "
                   "cross-pillar cheapness composite of the valuation yields.",
    "requires":    list(_YIELDS),
    "produces":    list(_PRODUCES),
    "tags":        ["fundamentals", "valuation", "mean_reversion", "momentum"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _rolling_pctl(s: pd.Series, win: int, mp: int) -> pd.Series:
    """Percentile rank of the last value within each trailing window, in [0,1]."""
    def _rank_last(x: np.ndarray) -> float:
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        if valid.size < 2:
            return np.nan
        # fraction of window values <= current (strict-less + half-ties handled simply by <=)
        return float(np.mean(valid <= last))
    return s.rolling(win, min_periods=mp).apply(_rank_last, raw=True)


def _zscore(s: pd.Series, win: int, mp: int) -> pd.Series:
    mean = s.rolling(win, min_periods=mp).mean()
    std = s.rolling(win, min_periods=mp).std()
    std = std.where(std > 0)
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    mp_rank = max(2, int(_RANK_WIN * _MIN_FRAC))
    mp_z = max(2, int(_Z_WIN * _MIN_FRAC))

    z_pillars = []  # collect standardized pillar series for the composite

    for y in _YIELDS:
        s = _series(df, y)
        tag = _TAG[y]

        # 1. Own-history percentile rank over the trailing year.
        df[f"fvn_{tag}_pctl_{_RANK_WIN}"] = (
            _rolling_pctl(s, _RANK_WIN, mp_rank).clip(0.0, 1.0)
        )

        # 2. 20d change in the multiple (valuation momentum).
        chg = (s - s.shift(_CHG_LAG)).replace([np.inf, -np.inf], np.nan).clip(-_CLIP, _CLIP)
        df[f"fvn_{tag}_chg{_CHG_LAG}"] = chg

        # standardized pillar for the composite (only the value yields)
        if y in _COMPOSITE_PILLARS:
            z_pillars.append(_zscore(s, _Z_WIN, mp_z).clip(-_CLIP, _CLIP))

    # 3. Cheapness composite = mean of available standardized pillars (skip NaN pillars per row).
    if z_pillars:
        zmat = pd.concat(z_pillars, axis=1)
        composite = zmat.mean(axis=1, skipna=True)  # NaN only if ALL pillars NaN that row
        df["fvn_cheapness_composite"] = (
            composite.replace([np.inf, -np.inf], np.nan).clip(-_CLIP, _CLIP)
        )
    else:
        df["fvn_cheapness_composite"] = np.nan

    return df
