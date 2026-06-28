"""
multistate_conditional_drift.py - Conditional next-day drift across price states (Tier-2).

conditional_reversion_skew (a screen winner) measured E[r_next | above EWMA] - E[r_next | below
EWMA]. The screen taught two things: (1) the RAW conditional-mean difference beat the t-stat-
normalized form, and (2) this state-conditioned-drift vein carries real marginal edge. So this
block generalizes it to THREE different state variables, each emitting a RAW conditional-drift
asymmetry the way the winner did:

  rdd_cdrift_range : E[r_next | high in 20d range] - E[r_next | low in range]   (stochastic %K state)
  rdd_cdrift_ddown : E[r_next | shallow drawdown]  - E[r_next | deep drawdown]  (capitulation state)
  rdd_cdrift_streak: E[r_next | long up-streak]    - E[r_next | long down-streak](momentum-run state)

Each is a difference of conditional FIRST moments, so a symmetric high-volatility name nets ~0
(volatility-orthogonal). Leak-free: each past day's state is paired with its already-realized
next-day return (state lagged one bar; window ends at t). Vectorised via masked rolling sums.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 90
_MIN_SIDE = 6

METADATA = {
    "name":        "multistate_conditional_drift",
    "description": "Raw conditional next-day drift asymmetry E[r_next|high state]-E[r_next|low state] over 60d for three states: 20d range position (%K), 252d drawdown depth, and signed return-streak length. Generalizes conditional_reversion_skew (raw form, the screen winner) to multiple state definitions.",
    "requires":    ["Close"],
    "produces":    ["rdd_cdrift_range", "rdd_cdrift_ddown", "rdd_cdrift_streak"],
    "tags":        ["mean_reversion", "momentum", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 drift build (multi-state conditional drift)",
}


def _cdrift(y: pd.Series, hi: pd.Series, lo: pd.Series) -> pd.Series:
    hi = hi.astype(float); lo = lo.astype(float)
    n_hi = hi.rolling(_W, min_periods=_W // 2).sum()
    n_lo = lo.rolling(_W, min_periods=_W // 2).sum()
    e_hi = (y * hi).rolling(_W, min_periods=_W // 2).sum() / n_hi.replace(0, np.nan)
    e_lo = (y * lo).rolling(_W, min_periods=_W // 2).sum() / n_lo.replace(0, np.nan)
    return (e_hi - e_lo).where((n_hi >= _MIN_SIDE) & (n_lo >= _MIN_SIDE))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    y = close.pct_change()

    # state A: position in the trailing 20d range (stochastic %K), lagged one bar
    lo20 = close.rolling(20, min_periods=10).min()
    hi20 = close.rolling(20, min_periods=10).max()
    k = ((close - lo20) / (hi20 - lo20).replace(0, np.nan)).shift(1)
    df["rdd_cdrift_range"] = np.clip(_cdrift(y, k > 0.7, k < 0.3).values, -0.2, 0.2)

    # state B: drawdown from the trailing 252d high, by its OWN rolling percentile (relative, so
    # both shallow and deep states populate for every name regardless of its trendiness)
    dd = close / close.rolling(252, min_periods=60).max() - 1.0
    dd_pct = dd.rolling(120, min_periods=40).rank(pct=True).shift(1)
    df["rdd_cdrift_ddown"] = np.clip(_cdrift(y, dd_pct > 0.66, dd_pct < 0.33).values, -0.2, 0.2)

    # state C: signed consecutive-day streak length, lagged (>=2-day up vs down run)
    sgn = np.sign(y.fillna(0.0))
    brk = (sgn != sgn.shift()).cumsum()
    runlen = sgn.groupby(brk).cumcount() + 1
    signed_run = (runlen * sgn).shift(1)
    df["rdd_cdrift_streak"] = np.clip(_cdrift(y, signed_run >= 2, signed_run <= -2).values, -0.2, 0.2)

    return df
