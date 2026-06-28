"""
asym_sign_persistence.py — Persistence of return sign & AR(1) structure (Tier-2).

Short-horizon return autocorrelation and runs structure separate continuation
names from mean-reverters. Two sound constructs:
  * Sign-persistence / runs: the fraction of consecutive days that keep the same
    sign. High same-sign persistence == trending/streaky tape; low == choppy
    reversal. Related to the classic runs test (Wald-Wolfowitz 1940) for
    randomness of a sign sequence.
  * Lag-1 autocorrelation of returns (Lo & MacKinlay 1988, "Stock market prices
    do not follow random walks"): positive AR(1) == momentum/continuation,
    negative == bid-ask-bounce / reversal. Computed vectorised via the rolling
    covariance identity   rho1 = cov(r_t, r_{t-1}) / var(r_t).

Over a rolling window W of daily simple returns r:
    same_sign_frac = mean( 1[sign(r_t)==sign(r_{t-1})] )  in [0,1]
    signed_persist = same_sign_frac * sign(Σr)            persistence × direction
    ar1            = cov(r_t, r_{t-1}) / var(r_t)          lag-1 autocorrelation
    up_run_frac    = mean( 1[r_t>0 & r_{t-1}>0] )          density of up-streak days

Pure per-ticker, fully vectorised (no rolling.apply). Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "asym_sign_persistence",
    "description": "Return sign persistence and lag-1 autocorrelation: same-sign fraction, direction-signed persistence, AR(1) via rolling covariance, and up-streak density at 21d/63d (Wald-Wolfowitz runs; Lo-MacKinlay AR(1)).",
    "requires":    ["Close"],
    "produces":    [
        f"asy_{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("samesignfrac", "signedpersist", "ar1", "uprunfrac")
    ],
    "tags":        ["momentum", "mean_reversion", "autocorrelation", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 dist-shape build (Wald-Wolfowitz runs / Lo-MacKinlay)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change()
    r_lag = r.shift(1)

    sgn = np.sign(r)
    sgn_lag = np.sign(r_lag)
    # same non-zero sign on consecutive days
    same_sign = ((sgn != 0) & (sgn == sgn_lag)).astype(float)
    up_run = ((r > 0) & (r_lag > 0)).astype(float)

    for w, mp in _WINDOWS:
        same_frac = same_sign.rolling(w, min_periods=mp).mean()
        up_frac = up_run.rolling(w, min_periods=mp).mean()
        net = r.rolling(w, min_periods=mp).sum()

        # vectorised rolling lag-1 autocorrelation via covariance identity.
        # pair-aligned window (mp-1 because the first pair needs r_{t-1}).
        pmp = max(mp - 1, 2)
        mean_t = r.rolling(w, min_periods=pmp).mean()
        mean_l = r_lag.rolling(w, min_periods=pmp).mean()
        cov = (r * r_lag).rolling(w, min_periods=pmp).mean() - mean_t * mean_l
        var_t = (r ** 2).rolling(w, min_periods=pmp).mean() - mean_t ** 2
        ar1 = cov / var_t.replace(0, np.nan)

        df[f"asy_samesignfrac_{w}"] = same_frac.values
        df[f"asy_signedpersist_{w}"] = (same_frac * np.sign(net)).values
        df[f"asy_ar1_{w}"] = ar1.clip(-1, 1).values
        df[f"asy_uprunfrac_{w}"] = up_frac.values

    return df
