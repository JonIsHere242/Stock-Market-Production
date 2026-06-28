"""
asymmetric_drift_field.py - Up/down-asymmetric Langevin restoring force (Tier-2).

The screen flagged the Langevin/Kramers-Moyal drift vein (langevin_cubic_drift) and the
state-conditioned drift (conditional_reversion_skew) as genuine marginal edge. This deepens
that vein: instead of one restoring force, estimate the LINEAR restoring SLOPE separately
ABOVE and BELOW equilibrium. A name whose price snaps back fast from BELOW its equilibrium but
drifts freely ABOVE it has an upward-tilted basin (bullish); the reverse is bearish. This is
the SPEED (slope) asymmetry of the drift field, complementary to conditional_reversion_skew
which measured the conditional MEAN-return asymmetry.

  beta_up = slope of next-day change on displacement,  for days ABOVE the EWMA equilibrium
  beta_dn = same, for days BELOW
  rdd_drift_asym = beta_up - beta_dn   (basin tilt;  more-negative beta_dn => fast snap from below)
  rdd_drift_dn   = beta_dn             (downside restoring speed alone)
  rdd_drift_pred = side-conditioned predicted next-day drift at today's displacement

Leak-free (past displacement paired with realized next-day change), vectorised via masked
rolling sums. Standardised displacement -> volatility-orthogonal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISP_W = 60
_REG_W = 120
_MIN_SIDE = 20

METADATA = {
    "name":        "asymmetric_drift_field",
    "description": "Up/down-asymmetric Langevin restoring slope: linear drift of next-day log change on standardized displacement fitted separately above vs below the EWMA equilibrium; emits the slope asymmetry, the downside restoring speed, and the side-conditioned predicted drift. Extends the validated drift/Kramers-Moyal vein.",
    "requires":    ["Close"],
    "produces":    ["rdd_drift_asym", "rdd_drift_dn", "rdd_drift_pred"],
    "tags":        ["mean_reversion", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 drift build (asymmetric Friedrich-Peinke drift field)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["Close"].astype(float).clip(lower=1e-9))
    mu = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).mean()
    sd = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).std().replace(0, np.nan)
    x = ((logc - mu) / sd).clip(-4, 4)
    y = logc.diff()
    xlag = x.shift(1)

    mp = int(_REG_W * 0.6)

    def _beta(mask):
        m = mask.astype(float)
        n = m.rolling(_REG_W, min_periods=mp).sum()
        sx = (xlag * m).rolling(_REG_W, min_periods=mp).sum()
        sy = (y * m).rolling(_REG_W, min_periods=mp).sum()
        sxy = (xlag * y * m).rolling(_REG_W, min_periods=mp).sum()
        sxx = (xlag * xlag * m).rolling(_REG_W, min_periods=mp).sum()
        nn = n.replace(0, np.nan)
        cov = sxy / nn - (sx / nn) * (sy / nn)
        var = (sxx / nn - (sx / nn) ** 2).replace(0, np.nan)
        beta = (cov / var).where(n >= _MIN_SIDE)
        return beta.clip(-3, 3)

    beta_up = _beta(xlag > 0)
    beta_dn = _beta(xlag < 0)
    pred = np.where(x.values > 0, (beta_up * x).values, (beta_dn * x).values)

    df["rdd_drift_asym"] = np.clip((beta_up - beta_dn).values, -6, 6)
    df["rdd_drift_dn"] = np.clip(beta_dn.values, -3, 3)
    df["rdd_drift_pred"] = np.clip(pred, -0.1, 0.1)
    return df
