"""
range_volatility.py — Range-based (OHLC) volatility estimators (Tier-2).

Classic intraday-range volatility estimators that use the full OHLC bar and are
far more efficient than close-to-close vol:
    Parkinson (1980)        : uses High-Low range only.
    Garman-Klass (1980)     : uses High,Low,Open,Close (no overnight drift).
    Rogers-Satchell (1991)  : drift-independent; valid when the series trends.
    Yang-Zhang (2000)       : combines overnight + open-to-close + Rogers-Satchell;
                              minimum-variance, handles drift AND opening jumps.

Each is computed as a rolling-window estimate, annualisation OMITTED (we want a
clean per-day vol level for cross-sectional ranking, not an annual number). For
the cross-section, a name with low *true* range vol but a recent extreme close
move is a different animal than a high-range-vol name — these separate them, and
the YZ/RS pair captures whether recent vol is drift-driven or noise-driven, a
Tier-2 tail discriminator.

Per-bar variance proxies (log prices o,h,l,c with previous close c_prev):
    park_d  = (1/(4 ln2)) * (h-l)^2
    gk_d    = 0.5*(h-l)^2 - (2 ln2 -1)*(c-o)^2
    rs_d    = (h-c)(h-o) + (l-c)(l-o)        [Rogers-Satchell, drift-free]
    overnight = (o - c_prev)^2 ; openclose = (c - o)^2
The rolling vols are sqrt(mean(per-bar proxy)). Yang-Zhang blends the three
variance pieces with the canonical k weight.

Pure per-ticker, fully vectorised (no loops, no rolling.apply).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [(21, 15), (63, 40)]
_LN2 = np.log(2.0)

METADATA = {
    "name":        "range_volatility",
    "description": "Rolling range-based volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) at 21d/63d from OHLC bars.",
    "requires":    ["Open", "High", "Low", "Close"],
    "produces":    [
        f"{p}_{w}"
        for w, _ in _WINDOWS
        for p in ("rng_parkinson", "rng_garmanklass",
                  "rng_rogerssatchell", "rng_yangzhang")
    ],
    "tags":        ["volatility", "range", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Parkinson 1980 / Garman-Klass 1980 / Rogers-Satchell 1991 / Yang-Zhang 2000)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Guard non-positive prices before taking logs.
    o = np.log(df["Open"].where(df["Open"] > 0))
    h = np.log(df["High"].where(df["High"] > 0))
    l = np.log(df["Low"].where(df["Low"] > 0))
    c = np.log(df["Close"].where(df["Close"] > 0))
    c_prev = c.shift(1)

    hl = h - l
    co = c - o

    # Per-bar variance proxies.
    park_d = (hl ** 2) / (4.0 * _LN2)
    gk_d = 0.5 * hl ** 2 - (2.0 * _LN2 - 1.0) * co ** 2
    rs_d = (h - c) * (h - o) + (l - c) * (l - o)        # Rogers-Satchell
    overnight = (o - c_prev) ** 2                        # close-to-open
    openclose = co ** 2                                  # open-to-close

    for w, mp in _WINDOWS:
        park = np.sqrt(park_d.rolling(w, min_periods=mp).mean().clip(lower=0))
        gk = np.sqrt(gk_d.rolling(w, min_periods=mp).mean().clip(lower=0))
        rs_mean = rs_d.rolling(w, min_periods=mp).mean()
        rs = np.sqrt(rs_mean.clip(lower=0))

        # Yang-Zhang: sigma_o^2 (overnight var) + k*sigma_c^2 (openclose var)
        #            + (1-k)*sigma_rs^2,  k = 0.34/(1.34 + (w+1)/(w-1))
        var_o = overnight.rolling(w, min_periods=mp).var()
        var_c = openclose.rolling(w, min_periods=mp).var()
        k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
        yz_var = var_o + k * var_c + (1.0 - k) * rs_mean
        yz = np.sqrt(yz_var.clip(lower=0))

        df[f"rng_parkinson_{w}"] = park.clip(upper=2.0).values
        df[f"rng_garmanklass_{w}"] = gk.clip(upper=2.0).values
        df[f"rng_rogerssatchell_{w}"] = rs.clip(upper=2.0).values
        df[f"rng_yangzhang_{w}"] = yz.clip(upper=2.0).values

    return df
