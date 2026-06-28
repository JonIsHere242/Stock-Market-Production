"""
conditional_reversion_skew.py - State-conditioned next-day drift asymmetry (Tier-2).

From the early-warning-signals / critical-slowing-down literature (Scheffer et al. 2009;
Dakos et al. PLoS ONE 2012): the local restoring dynamics around an equilibrium are
asymmetric near a tilt in the basin of attraction. We estimate, from a name's OWN trailing
history, the difference between its average next-day return when it sits ABOVE vs BELOW its
short equilibrium (an EWMA of price):

    diff_t = E[ r_next | price above EWMA ]  -  E[ r_next | price below EWMA ]

A positive diff means the basin is tilted up (continuation from above, weak pull from below).
Because it is a difference of conditional FIRST moments, a symmetric high-volatility name nets
~0, so the raw difference is divided by the pooled dispersion and the two-sample standard error
to form a t-statistic -- this strips the residual volatility loading and leaves a clean,
sign-coherent directional tilt. Leak-free: each past day's displacement is paired with its
already-realized next-day return (window ends at t; the target r_{t+1} is never used).

NOTE: own-history forward-conditional drift estimates decay out of sample (the project's
85-95% IS->OOS pattern), so treat as speculative pending the in-model marginal-lift check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 60
_SPAN = 20
_MIN_SIDE = 8

METADATA = {
    "name":        "conditional_reversion_skew",
    "description": "State-conditioned next-day drift asymmetry: E[r_next | above EWMA] - E[r_next | below EWMA] over 60d, as a vol-normalized two-sample t-statistic and the raw return-unit difference. Early-warning-signal basin-tilt analog; leak-free (past displacement paired with realized next-day return).",
    "requires":    ["Close"],
    "produces":    ["xdm_creskew_t", "xdm_creskew_raw"],
    "tags":        ["mean_reversion", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 cross-domain build (Scheffer/Dakos EWS basin asymmetry)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    r = close.pct_change()
    ewma = close.ewm(span=_SPAN, min_periods=_SPAN).mean()

    # displacement sign known at end of day s; pair with the return realized the NEXT day.
    # At index k: yesterday's displacement (shift 1) paired with today's return r_k -> all past.
    disp_lag = np.sign(close - ewma).shift(1)
    up = (disp_lag > 0).astype(float)
    dn = (disp_lag < 0).astype(float)

    n_up = up.rolling(_W, min_periods=_W // 2).sum()
    n_dn = dn.rolling(_W, min_periods=_W // 2).sum()
    sum_up = (r * up).rolling(_W, min_periods=_W // 2).sum()
    sum_dn = (r * dn).rolling(_W, min_periods=_W // 2).sum()

    e_up = sum_up / n_up.replace(0, np.nan)
    e_dn = sum_dn / n_dn.replace(0, np.nan)
    diff = e_up - e_dn

    sd = r.rolling(_W, min_periods=_W // 2).std()
    se = sd * np.sqrt(1.0 / n_up.replace(0, np.nan) + 1.0 / n_dn.replace(0, np.nan))
    tstat = diff / se.replace(0, np.nan)

    enough = (n_up >= _MIN_SIDE) & (n_dn >= _MIN_SIDE)
    df["xdm_creskew_t"] = np.clip(tstat.where(enough).values, -6, 6)
    df["xdm_creskew_raw"] = np.clip(diff.where(enough).values, -0.2, 0.2)

    return df
