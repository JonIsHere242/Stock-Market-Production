"""
ilr_illiquidity_restoring.py - Illiquidity-modulated restoring force (Tier-2).

NY Fed Staff Report 513 "The Cross Section of Stock Returns / Decomposing Short-Term Return
Reversal": short-term reversal is stronger and longer for ILLIQUID names -- illiquidity modulates
how hard a displaced price snaps back. We gate a volatility-standardized displacement by a
THINNESS proxy and emit the implied next-day reversion pressure:

  disp  = clip( z-score(log Close, 40d), -4, 4 )                 (how far from its own trend)
  thin  = rolling 60d percentile-rank of 1/(Close*Volume)        (dollar-volume thinness)
  g     = clip( (thin - 0.66)/0.34, 0, 1 )                       (soft top-tercile illiquidity hinge)
  ilr_restore = -disp * g     (zeroed when |disp| < 1)

De-trapped per the review: the thinness gate uses inverse DOLLAR-VOLUME percentile, NOT raw Amihud
(whose |return| numerator re-leaks range/vol); illiquidity is only a CONDITIONING gate on a vol-
standardized displacement, so the output is signed reversion pressure, not an illiquidity level
(micro_amihud already ships the level) -- orthogonal to the return-distribution (A) and cost-basis
(B) pockets. The OOS-decay-prone masked-slope (kappa) leg is intentionally NOT shipped. Vectorised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISP_W = 40
_THIN_W = 60

METADATA = {
    "name":        "ilr_illiquidity_restoring",
    "description": "Illiquidity-modulated restoring force (NY Fed SR513): a volatility-standardized displacement gated by a dollar-volume thinness percentile, emitting signed next-day reversion pressure (zeroed when displacement is small). Thinness is only a conditioning gate -> orthogonal to return-distribution and cost-basis pockets.",
    "requires":    ["Close", "Volume"],
    "produces":    ["ilr_restore"],
    "tags":        ["mean_reversion", "liquidity", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 open-room build (NY Fed SR513 reversal-illiquidity)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    logc = np.log(close.clip(lower=1e-9))
    mu = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).mean()
    sd = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).std().replace(0, np.nan)
    disp = ((logc - mu) / sd).clip(-4, 4)

    inv_dvol = 1.0 / (close * vol).replace(0, np.nan)
    thin = inv_dvol.rolling(_THIN_W, min_periods=20).rank(pct=True)
    g = ((thin - 0.66) / 0.34).clip(0, 1)

    restore = (-disp * g).where(disp.abs() >= 1.0, 0.0)
    df["ilr_restore"] = np.clip(restore.values, -4, 4)
    return df
