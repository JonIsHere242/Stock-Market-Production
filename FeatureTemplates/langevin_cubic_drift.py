"""
langevin_cubic_drift.py - OU-residualized nonlinear (cubic) drift field (Tier-2).

Stochastic-thermodynamics view of a price as a Langevin process dx = D1(x)dt + sqrt(2 D2(x))dW,
where the drift field D1(x) is recovered by the Kramers-Moyal expansion (Friedrich-Peinke):
D1(x) ~ E[ x_next - x | x ]. A linear D1(x) = -theta * x is just Ornstein-Uhlenbeck mean
reversion (already built). The INFORMATIVE, under-exploited part is the NONLINEARITY: a cubic
term captures an asymmetric / double-well restoring force (strong snap-back far from
equilibrium, weak or runaway behavior near it) that the linear half-life misses.

We standardize the displacement x = (log-price - rolling mean) / rolling std, regress the
next-day log change on x and x^3 SEQUENTIALLY -- first remove the linear OU component, then fit
the cubic coefficient on the residual -- so the output is ORTHOGONAL to the existing OU block by
construction. Two names at the same displacement and the same volatility can carry opposite
nonlinear drift, which is why this conditions out the diffusion (volatility) term.

  xdm_langevin_cubic : the cubic drift coefficient (sign: <0 restoring, >0 accelerating)
  xdm_langevin_drift : predicted nonlinear drift at today's displacement (cubic_coef * x_today^3)

Speculative (own-history drift estimation is noisy); leak-free (past displacement paired with
already-realized next-day change). Vectorized via rolling covariance/variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DISP_W = 60       # window for the standardized displacement
_REG_W = 120       # rolling-regression window

METADATA = {
    "name":        "langevin_cubic_drift",
    "description": "OU-residualized cubic drift field via Kramers-Moyal D1(x): sequentially strip the linear (OU) drift of next-day log change on standardized displacement, then fit the cubic coefficient on the residual; emit the coefficient and the predicted nonlinear drift at today's displacement. Orthogonal to the OU block by construction.",
    "requires":    ["Close"],
    "produces":    ["xdm_langevin_cubic", "xdm_langevin_drift"],
    "tags":        ["mean_reversion", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (Friedrich-Peinke Kramers-Moyal drift)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["Close"].astype(float).clip(lower=1e-9))

    mu = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).mean()
    sd = logc.rolling(_DISP_W, min_periods=_DISP_W // 2).std().replace(0, np.nan)
    x = ((logc - mu) / sd).clip(-4, 4)

    y = logc.diff()                      # next-day log change, realized
    xlag = x.shift(1)                    # displacement known the prior day -> paired with y (past)

    mp = int(_REG_W * 0.6)
    beta_lin = xlag.rolling(_REG_W, min_periods=mp).cov(y) / \
        xlag.rolling(_REG_W, min_periods=mp).var().replace(0, np.nan)
    resid = y - beta_lin * xlag

    x3lag = xlag ** 3
    beta_cub = x3lag.rolling(_REG_W, min_periods=mp).cov(resid) / \
        x3lag.rolling(_REG_W, min_periods=mp).var().replace(0, np.nan)

    drift = beta_cub * (x ** 3)

    df["xdm_langevin_cubic"] = np.clip(beta_cub.values, -2, 2)
    df["xdm_langevin_drift"] = np.clip(drift.values, -0.1, 0.1)

    return df
