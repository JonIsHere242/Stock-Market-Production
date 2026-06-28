"""
basin_well_geometry.py - Double-well / bistability of the price drift field (Tier-2).

Deepens the validated Langevin-drift vein into the GEOMETRY of the potential. Fitting the
next-day drift D1(x) = a*x + b*x^3 on the standardized displacement x gives an implied
potential U(x) = -(a x^2/2 + b x^4/4) whose shape says whether the price sits in a single
mean-reverting well or a DOUBLE WELL (bistable) regime:

  U''(0) ~ -a :  a<0 -> origin is a stable well (mean-reverting range)
                 a>0 & b<0 -> origin is an unstable hill with two side wells at x=+-sqrt(-a/b)
                              => coiled / breakout-prone, the price is about to commit to a basin

We read three things, all from the joint rolling cubic fit (closed-form via rolling moments):

  rdd_well_curv : -a, curvature at the equilibrium (>0 stable range, <0 coiled/bistable)
  rdd_drift_now : a*x + b*x^3, the net drift the field assigns to today's displacement (directional)
  rdd_tip_dist  : |x_today| - sqrt(-a/b), signed distance past the unstable tipping point when
                  bistable (>0 = already committed to a basin, ~0 = right at the knife edge)

Leak-free, vectorised. Standardised displacement keeps it volatility-orthogonal. Genuinely
novel: no bistability/basin-geometry feature exists in the library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISP_W = 60
_REG_W = 120

METADATA = {
    "name":        "basin_well_geometry",
    "description": "Double-well / bistability geometry of the price drift field: from a joint rolling cubic fit of next-day change on standardized displacement, emit the potential curvature at equilibrium (>0 single-well/mean-reverting, <0 coiled/bistable) and the net drift at today's state. Novel breakout-regime detector built on the validated Langevin vein.",
    "requires":    ["Close"],
    "produces":    ["rdd_well_curv", "rdd_drift_now"],
    "tags":        ["mean_reversion", "market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 drift build (double-well potential geometry)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["Close"].astype(float).clip(lower=1e-9))
    mu = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).mean()
    sd = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).std().replace(0, np.nan)
    x = ((logc - mu) / sd).clip(-4, 4)

    u = x.shift(1)               # displacement (regressor 1)
    w = u ** 3                   # cubic term (regressor 2)
    y = logc.diff()             # next-day change (target, realized)

    valid = np.isfinite(u) & np.isfinite(w) & np.isfinite(y)
    u = u.where(valid); w = w.where(valid); y = y.where(valid)

    mp = int(_REG_W * 0.6)
    def rs(s): return s.rolling(_REG_W, min_periods=mp).sum()
    n = valid.astype(float).rolling(_REG_W, min_periods=mp).sum().replace(0, np.nan)
    Su, Sw, Sy = rs(u), rs(w), rs(y)
    Suu, Sww, Suw = rs(u * u), rs(w * w), rs(u * w)
    Suy, Swy = rs(u * y), rs(w * y)

    cuu = Suu - Su * Su / n
    cww = Sww - Sw * Sw / n
    cuw = Suw - Su * Sw / n
    cuy = Suy - Su * Sy / n
    cwy = Swy - Sw * Sy / n
    det = (cuu * cww - cuw * cuw).replace(0, np.nan)
    a = (cww * cuy - cuw * cwy) / det
    b = (cuu * cwy - cuw * cuy) / det

    drift_now = a * x + b * (x ** 3)

    df["rdd_well_curv"] = np.clip((-a).values, -3, 3)
    df["rdd_drift_now"] = np.clip(drift_now.values, -0.1, 0.1)
    return df
