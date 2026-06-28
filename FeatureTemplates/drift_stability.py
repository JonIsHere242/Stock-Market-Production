"""
drift_stability.py - Stability/instability of the mean-reversion restoring force (Tier-2).

The screen flagged the conditional-drift / Langevin restoring-force vein as real edge. This adds
the META layer: how STABLE is that restoring force? A name whose short-window restoring
coefficient is steady (always pulling back) behaves very differently from one whose drift keeps
flipping between mean-reverting and trending regimes. We estimate the short-window OU restoring
slope beta = cov(x_lag, dx_next)/var(x_lag) on the standardized displacement, then read its
own dynamics:

  dst_beta_instab : 120d rolling std of the restoring coefficient (regime churn)
  dst_regime      : the smoothed current restoring strength (>0 = mean-reverting now, <0 = trending)

Standardised displacement -> volatility-orthogonal. Leak-free (past displacement vs realized
next change), vectorised via rolling covariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISP_W = 60
_BETA_W = 40
_LONG_W = 120

METADATA = {
    "name":        "drift_stability",
    "description": "Stability of the OU restoring force: 120d instability (std) of the short-window mean-reversion coefficient and the smoothed current restoring strength. Meta-layer on the validated conditional-drift vein.",
    "requires":    ["Close"],
    "produces":    ["dst_beta_instab", "dst_regime"],
    "tags":        ["mean_reversion", "market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 drift build (restoring-force stability)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["Close"].astype(float).clip(lower=1e-9))
    mu = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).mean()
    sd = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).std().replace(0, np.nan)
    x = ((logc - mu) / sd).clip(-4, 4)
    y = logc.diff()
    xlag = x.shift(1)

    mp = int(_BETA_W * 0.6)
    beta = (xlag.rolling(_BETA_W, min_periods=mp).cov(y)
            / xlag.rolling(_BETA_W, min_periods=mp).var().replace(0, np.nan)).clip(-3, 3)

    instab = beta.rolling(_LONG_W, min_periods=_LONG_W // 2).std()
    regime = -beta.ewm(span=10, min_periods=5).mean()        # >0 = restoring (mean-reverting)

    df["dst_beta_instab"] = np.clip(instab.values, 0, 3)
    df["dst_regime"] = np.clip(regime.values, -3, 3)
    return df
