"""
realized_higher_moments.py — Realized skewness / kurtosis / signed-jump (Tier-2).

Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE) "Does realized skewness
predict the cross-section of equity returns?" — YES, NEGATIVELY: high realized
skewness this period -> lower next-period returns (a lottery/asymmetry effect).
We add the realized higher moments + an upside/downside variation split (signed
jump) over rolling daily windows. These sort the cross-section by the SHAPE of a
name's recent return distribution, not its level — a Tier-2 tail separator.

Over window W (sum over the window of daily log returns r):
    rskew = sqrt(W) * Σr³ / (Σr²)^1.5         realized skewness
    rkurt = W * Σr⁴ / (Σr²)²                   realized kurtosis
    signed_jump = (RV+ - RV-) / RV in [-1,1]   upside vs downside variation
    upvar_ratio = RV+ / RV in [0,1]            share of variation that is upside
        where RV+ = Σ max(r,0)², RV- = Σ min(r,0)², RV = Σ r²

Pure per-ticker, vectorised. Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "realized_higher_moments",
    "description": "Realized skewness, kurtosis, signed-jump and upside-variance ratio at 21d/63d, per Amaya-Christoffersen-Jacobs-Vasquez 2015.",
    "requires":    ["Close"],
    "produces":    [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("rskew", "rkurt", "signed_jump", "upvar_ratio")
    ],
    "tags":        ["volatility", "tail", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (ACJV 2015)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    r = np.log(close / close.shift(1))
    rp2 = r.clip(lower=0) ** 2
    rn2 = r.clip(upper=0) ** 2

    for w, mp in _WINDOWS:
        s2 = (r ** 2).rolling(w, min_periods=mp).sum()
        s3 = (r ** 3).rolling(w, min_periods=mp).sum()
        s4 = (r ** 4).rolling(w, min_periods=mp).sum()
        s2_safe = s2.replace(0, np.nan)
        rvp = rp2.rolling(w, min_periods=mp).sum()
        rvn = rn2.rolling(w, min_periods=mp).sum()

        df[f"rskew_{w}"] = (np.sqrt(w) * s3 / s2_safe ** 1.5).clip(-20, 20).values
        df[f"rkurt_{w}"] = (w * s4 / s2_safe ** 2).clip(0, 200).values
        df[f"signed_jump_{w}"] = ((rvp - rvn) / s2_safe).values
        df[f"upvar_ratio_{w}"] = (rvp / s2_safe).values

    return df
