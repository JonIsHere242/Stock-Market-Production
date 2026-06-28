"""
info_discreteness.py — Frog-in-the-Pan / information discreteness (Tier-2).

Da, Gurun & Warachka (2014, RFS) "Frog in the Pan: Continuous Information and
Momentum". Investors under-react to information that arrives in many small
continuous steps vs few large discrete jumps. So among names with the SAME
cumulative move, the ones whose move was *continuous* keep going (8.86% vs 2.91%
6-mo momentum). This is a pure within-winner sorter — exactly Tier-2.

Information Discreteness over a formation window W:
    PRET   = cumulative return over W
    pos    = fraction of up days,  neg = fraction of down days
    ID     = sign(PRET) * (neg - pos)
       ID < 0  -> continuous (mostly same-direction small days) -> CONTINUES
       ID > 0  -> discrete (lumpy, jump-driven)                 -> fades
    cont_mom = PRET * (pos - neg)
       ranks continuous winners ABOVE discrete winners (and continuous losers
       below), which is the discriminating signal you actually trade on.

Pure per-ticker, vectorised. Two formation horizons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [60, 120]
_MIN = {60: 40, 120: 80}

METADATA = {
    "name":        "info_discreteness",
    "description": "Frog-in-the-Pan information discreteness (ID) + continuity-weighted momentum at 60d/120d, per Da-Gurun-Warachka 2014.",
    "requires":    ["Close"],
    "produces":    [f"{p}_{w}" for w in _WINDOWS for p in ("info_discreteness", "cont_mom")],
    "tags":        ["momentum", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (DGW 2014)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    ret = close.pct_change()
    up = (ret > 0).astype(float)
    dn = (ret < 0).astype(float)

    for w in _WINDOWS:
        mp = _MIN[w]
        pos = up.rolling(w, min_periods=mp).mean()
        neg = dn.rolling(w, min_periods=mp).mean()
        pret = close / close.shift(w) - 1.0
        sign = np.sign(pret)
        df[f"info_discreteness_{w}"] = (sign * (neg - pos)).values
        df[f"cont_mom_{w}"] = (pret * (pos - neg)).values

    return df
