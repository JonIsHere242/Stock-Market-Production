"""
hill_tail_index.py — Rolling tail-thickness & up/down tail asymmetry (Tier-2).

Hill (1975) tail-index intuition: the heaviness of a return distribution's tail is
governed by how slowly the largest order statistics decay. A fully vectorized,
rolling proxy uses the RATIO of an extreme quantile to a moderate quantile — this
ratio rises monotonically with tail heaviness (it is large for fat-tailed series,
near 1 for light tails), and unlike a true top-k Hill sum it needs no per-window
sorting loop (keeps us < 15 ms/ticker).

    tail_thick = q(|r|, 0.99) / q(|r|, 0.75)          overall fat-tailedness
    rtail      = q(r_up,  0.99) / q(r_up,  0.75)       RIGHT-tail thickness  (gains)
    ltail      = q(r_dn,  0.99) / q(r_dn,  0.75)       LEFT-tail  thickness  (losses)
    tail_asym  = rtail - ltail                         fat-right vs fat-left

Names with a fat RIGHT tail and a thin LEFT tail (tail_asym > 0) have lottery-like
upside with limited crash risk — a within-cross-section shape sorter (Tier-2) that
ordinary volatility cannot see. Pure per-ticker, vectorized via rolling quantiles.
Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [63, 126]
_MIN = {63: 40, 126: 80}
_Q_HI = 0.99
_Q_LO = 0.75


METADATA = {
    "name":        "hill_tail_index",
    "description": "Rolling tail-thickness (extreme/moderate |return| quantile ratio) + right/left tail thickness and their asymmetry at 63d/126d, Hill-style proxy.",
    "requires":    ["Close"],
    "produces":    [f"htail_{p}_{w}" for w in _WINDOWS for p in ("tail_thick", "rtail", "ltail", "tail_asym")],
    "tags":        ["tail", "volatility", "higher_moments", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Hill 1975 tail index)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Close"].pct_change()
    absret = ret.abs()
    # one-sided magnitudes (only realized gains / only realized losses)
    up_mag = ret.clip(lower=0)
    dn_mag = (-ret).clip(lower=0)

    for w in _WINDOWS:
        mp = _MIN[w]

        def _ratio(s):
            hi = s.rolling(w, min_periods=mp).quantile(_Q_HI)
            lo = s.rolling(w, min_periods=mp).quantile(_Q_LO).replace(0, np.nan)
            return (hi / lo).clip(1.0, 50.0)

        tail_thick = _ratio(absret)
        rtail = _ratio(up_mag)
        ltail = _ratio(dn_mag)
        tail_asym = (rtail - ltail).clip(-25, 25)

        df[f"htail_tail_thick_{w}"] = tail_thick.values
        df[f"htail_rtail_{w}"] = rtail.values
        df[f"htail_ltail_{w}"] = ltail.values
        df[f"htail_tail_asym_{w}"] = tail_asym.values

    return df
