"""
dist_quantile_geometry.py — Robust quantile geometry of recent returns (Tier-2).

The cross-section of next-day winners tends to come from names whose recent
return distribution has a particular SHAPE: a fat right tail relative to the
left, and positive robust skew. Rather than moment-based skew (dominated by a
single outlier and noisy), we use order-statistic / quantile geometry, which is
the classic robust-statistics toolkit (Bowley 1920 quartile skewness; Hinkley
1975 generalized quantile skewness; Kim & White 2004 "On more robust estimation
of skewness and kurtosis" recommend exactly these for financial returns).

Over a rolling window W of daily simple returns r, with quantiles q_p:
    iqr_50      = q75 - q25                       (interquartile dispersion)
    idr_80      = q90 - q10                       (inter-decile dispersion)
    tail_ratio  = q95 / |q05|                     (right vs left tail size)
    bowley_skew = ((q90-q50)-(q50-q10))/(q90-q10) (robust skew in [-1,1])
    qkurt       = (q90-q10)/(q75-q25)             (robust tail/shoulder kurtosis)

These sort the cross-section by the geometry of a name's recent return cloud,
not its level — a Tier-2 best-of-best separator. Pure per-ticker, vectorised
rolling quantiles. Two horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]

METADATA = {
    "name":        "dist_quantile_geometry",
    "description": "Robust quantile geometry of rolling daily returns: IQR/IDR dispersion, tail ratio, Bowley quantile skew, and quantile kurtosis at 21d/63d (Bowley/Hinkley/Kim-White robust shape).",
    "requires":    ["Close"],
    "produces":    [
        f"dsh_{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("iqr50", "idr80", "tailratio", "bowleyskew", "qkurt")
    ],
    "tags":        ["volatility", "tail", "distributional", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 dist-shape build (Bowley/Hinkley/Kim-White)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change()

    for w, mp in _WINDOWS:
        roll = r.rolling(w, min_periods=mp)
        q05 = roll.quantile(0.05)
        q10 = roll.quantile(0.10)
        q25 = roll.quantile(0.25)
        q50 = roll.quantile(0.50)
        q75 = roll.quantile(0.75)
        q90 = roll.quantile(0.90)
        q95 = roll.quantile(0.95)

        iqr50 = q75 - q25
        idr80 = q90 - q10
        idr80_safe = idr80.replace(0, np.nan)
        iqr50_safe = iqr50.replace(0, np.nan)
        # |q05| floored so a near-zero left tail can't explode the ratio
        left = q05.abs().clip(lower=1e-6)

        df[f"dsh_iqr50_{w}"] = iqr50.values
        df[f"dsh_idr80_{w}"] = idr80.values
        df[f"dsh_tailratio_{w}"] = (q95 / left).clip(-50, 50).values
        df[f"dsh_bowleyskew_{w}"] = (
            ((q90 - q50) - (q50 - q10)) / idr80_safe
        ).clip(-1, 1).values
        df[f"dsh_qkurt_{w}"] = (idr80 / iqr50_safe).clip(0, 50).values

    return df
