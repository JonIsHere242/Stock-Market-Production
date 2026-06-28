"""
variance_ratio.py - Lo-MacKinlay variance ratios: trending vs mean-reverting (Tier-2).

Lo & MacKinlay (1988, RFS) "Stock Market Prices Do Not Follow Random Walks: Evidence
from a Simple Specification Test". The variance ratio

    VR(q) = Var(q-period return) / (q * Var(1-period return))

diagnoses the autocorrelation STRUCTURE of a price path independently of its volatility
LEVEL (it is a ratio, so the vol scale cancels -- this is what keeps it out of the
volatility trap):
    VR(q) > 1  -> positive serial correlation -> trending / continuation
    VR(q) < 1  -> negative serial correlation -> mean-reverting
    VR(q) = 1  -> random walk

This is a pure regime classifier the tree can split on directionally: among top-of-book
candidates, the ones in a trending micro-regime continue, the ones in a reverting regime
fade. Distinct from the OU / AR(1) blocks (those fit a model; this is the model-free
overlapping-return variance test). Per-ticker, fully vectorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (q, window) pairs
_SPECS = [(2, 126), (5, 126), (10, 252)]

METADATA = {
    "name":        "variance_ratio",
    "description": "Lo-MacKinlay (1988) variance ratios VR(q)=Var(q-ret)/(q*Var(1-ret)) at (q=2,5,10) over 126d/252d windows, plus a centered trending-vs-reverting signal. Volatility-orthogonal (ratio cancels the vol level).",
    "requires":    ["Close"],
    "produces":    ["csa_vr_2_126", "csa_vr_5_126", "csa_vr_10_252", "csa_vr_signal_126"],
    "tags":        ["mean_reversion", "momentum", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Lo-MacKinlay 1988)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    logc = np.log(df["Close"].clip(lower=1e-9))
    r1 = logc.diff()

    vr5 = None
    for q, w in _SPECS:
        mp = int(w * 0.7)
        rq = logc.diff(q)                                     # overlapping q-period log return
        var1 = r1.rolling(w, min_periods=mp).var().replace(0, np.nan)
        varq = rq.rolling(w, min_periods=mp).var()
        vr = (varq / (q * var1)).clip(0, 5)
        df[f"csa_vr_{q}_{w}"] = vr.values
        if q == 5 and w == 126:
            vr5 = vr

    # Centered classifier: >0 trending (VR>1), <0 mean-reverting (VR<1).
    df["csa_vr_signal_126"] = (vr5 - 1.0).clip(-2, 2).values

    return df
