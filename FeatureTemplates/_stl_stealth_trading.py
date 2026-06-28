"""
stl_stealth_trading.py - Stealth-trading concentration (Tier-2).

Barclay & Warner (1993) "Stealth Trading"; Chakravarty (2001). Informed traders camouflage their
orders in MEDIUM size to avoid moving the price, so a disproportionate share of cumulative price
change is delivered on MEDIUM-volume days rather than on the loudest days. We measure WHERE in the
name's own volume distribution the recent drift landed:

  rv  = Volume / 60d-median volume
  medium-mask = rv between its trailing 33rd and 66th percentile
  stl_share_W  = sum( r * medium ) / sum( |r| )    over W days   (signed drift share on medium days)
  stl_vs_loud  = medium-bucket drift share  -  loud-bucket (rv >= 66th pct) drift share

De-trapped per the review: SHARE / RATIO forms only (the raw unnormalized drift sum leaks volume),
and the statistic DOWN-WEIGHTS the loudest days -- the opposite of a MAX/volatility feature. Return
enters as a signed share normalized by total absolute move, so orthogonal to the return-distribution
(A) and cost-basis (B) pockets; the thesis (positive medium-bucket drift = quiet informed
accumulation) is sign-coherent. Vectorised; leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINS = [20, 40]

METADATA = {
    "name":        "stl_stealth_trading",
    "description": "Stealth-trading concentration (Barclay-Warner/Chakravarty): signed share of recent cumulative price change delivered on MEDIUM-volume days (informed camouflage), at 20d/40d, plus medium-minus-loud bucket drift-share. Share/ratio forms only, down-weights loud days -> volatility-orthogonal.",
    "requires":    ["Close", "Volume"],
    "produces":    ["stl_share_20", "stl_share_40", "stl_vs_loud"],
    "tags":        ["volume", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 fresh-lead build (Barclay-Warner / Chakravarty stealth trading)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    r = close.pct_change()
    rv = vol / vol.rolling(60, min_periods=20).median().replace(0, np.nan)

    med40 = None
    loud40 = None
    for w in _WINS:
        mp = int(w * 0.6)
        q33 = rv.rolling(w, min_periods=mp).quantile(0.33)
        q66 = rv.rolling(w, min_periods=mp).quantile(0.66)
        medium = ((rv > q33) & (rv < q66)).astype(float)
        denom = r.abs().rolling(w, min_periods=mp).sum().replace(0, np.nan)
        share = (r * medium).rolling(w, min_periods=mp).sum() / denom
        df[f"stl_share_{w}"] = np.clip(share.values, -1, 1)
        if w == 40:
            med40, loud40 = medium, (rv >= q66).astype(float)

    denom40 = r.abs().rolling(40, min_periods=24).sum().replace(0, np.nan)
    loud_share = (r * loud40).rolling(40, min_periods=24).sum() / denom40
    med_share = (r * med40).rolling(40, min_periods=24).sum() / denom40
    df["stl_vs_loud"] = np.clip((med_share - loud_share).values, -1, 1)
    return df
