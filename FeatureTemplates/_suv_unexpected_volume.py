"""
suv_unexpected_volume.py - Standardized Unexpected Volume (Tier-2).

Garfinkel (2009, JAR) "Measuring Investors' Opinion Divergence"; Garfinkel & Sokobin (2006).
The part of (log) volume NOT explained by the day's price-move magnitude is trading heavy WITHOUT a
commensurate price move -- pure opinion divergence, which Garfinkel shows beats turnover, return
dispersion, and bid-ask spread as a disagreement proxy. We regress log volume on the up- and
down-move magnitudes and studentize the residual:

  y = log(1+Volume)  ~  intercept + b+ * max(r,0) + b- * max(-r,0)     (rolling 60d)
  suv = (y - yhat) / rolling_std(residual)

Volatility-orthogonal BY CONSTRUCTION: the move magnitude is the projected-out regressor, and the
residual is per-ticker studentized. Return is only a regressor, so this is orthogonal to the
return-distribution (A) and never references price level (B). NOTE: Miller/Hong-Stein implies high
SUV -> overpricing correcting DOWN, so it likely concentrates the next-day LOSER tail (use as a
short-side / gate signal); the screen reports which tail. Closed-form rolling OLS, leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 60

METADATA = {
    "name":        "suv_unexpected_volume",
    "description": "Standardized Unexpected Volume (Garfinkel 2009): studentized residual of log-volume regressed on up/down move magnitude over 60d -- volume orthogonal to return magnitude = opinion divergence. Volatility-orthogonal by construction; likely a short-side/gate signal.",
    "requires":    ["Close", "Volume"],
    "produces":    ["suv_resid"],
    "tags":        ["volume", "behavioral", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 fresh-lead build (Garfinkel 2009 SUV)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].astype(float).pct_change()
    rp = r.clip(lower=0)
    rn = (-r).clip(lower=0)
    y = np.log1p(df["Volume"].astype(float).clip(lower=0))

    valid = np.isfinite(rp) & np.isfinite(rn) & np.isfinite(y)
    rp = rp.where(valid); rn = rn.where(valid); y = y.where(valid)
    mp = int(_W * 0.6)

    def rs(s):
        return s.rolling(_W, min_periods=mp).sum()

    n = valid.astype(float).rolling(_W, min_periods=mp).sum().replace(0, np.nan)
    Sp, Sn, Sy = rs(rp), rs(rn), rs(y)
    Spp, Snn, Spn = rs(rp * rp), rs(rn * rn), rs(rp * rn)
    Spy, Sny = rs(rp * y), rs(rn * y)

    cpp = Spp - Sp * Sp / n
    cnn = Snn - Sn * Sn / n
    cpn = Spn - Sp * Sn / n
    cpy = Spy - Sp * Sy / n
    cny = Sny - Sn * Sy / n
    det = (cpp * cnn - cpn * cpn).replace(0, np.nan)
    bp = (cnn * cpy - cpn * cny) / det
    bn = (cpp * cny - cpn * cpy) / det
    a0 = Sy / n - bp * Sp / n - bn * Sn / n

    e = y - (a0 + bp * rp + bn * rn)
    estd = e.rolling(_W, min_periods=mp).std().replace(0, np.nan)

    df["suv_resid"] = np.clip((e / estd).values, -5, 5)
    return df
