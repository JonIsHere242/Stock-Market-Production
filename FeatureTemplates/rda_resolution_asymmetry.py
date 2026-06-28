"""
rda_resolution_asymmetry.py - Directional resolution asymmetry of unexpected volume (Tier-2).

Hong-Stein / Miller: under short-sale constraints, disagreement that trades on DOWN days (forced
sellers and pessimists who CAN trade) carries different information than disagreement on up days.
The intensity of ABNORMAL volume conditioned on the day's sign is a one-sided opinion-divergence
signal -- not a both-tails dispersion. We route abnormal volume into up/down buckets by the day's
return sign and read the down-day leg (primary) and the up-vs-down asymmetry:

  abn = Volume / mean(Volume, trailing 60) - 1
  rda_dn_disagree   = COUNT-normalized mean of abn over down days in the last 20d
  rda_resolution_asym = mean(abn | up days) - mean(abn | down days)

De-trapped per the review: abnormal volume is a relative (vol-normalized) quantity; the conditional
means are COUNT-normalized (divided by the number of up/down days, not the window length, so
frequency is not entangled with intensity); return enters only as a sign-mask (not a magnitude), so
orthogonal to the return-distribution (A) and cost-basis (B) pockets. Leak-free, vectorised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 20

METADATA = {
    "name":        "rda_resolution_asymmetry",
    "description": "Directional resolution asymmetry of unexpected volume (Hong-Stein/Miller): count-normalized mean abnormal volume on down days, and the up-minus-down asymmetry. Return enters only as a sign-mask routing volume into buckets -> orthogonal to return-distribution and cost-basis pockets.",
    "requires":    ["Close", "Volume"],
    "produces":    ["rda_dn_disagree", "rda_resolution_asym"],
    "tags":        ["volume", "behavioral", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 fresh-lead build (Hong-Stein / Miller disagreement)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    r = close.pct_change()

    base = vol.shift(1).rolling(60, min_periods=30).mean().replace(0, np.nan)
    abn = vol / base - 1.0

    dn = (r < 0).astype(float)
    up = (r > 0).astype(float)
    n_dn = dn.rolling(_W, min_periods=8).sum()
    n_up = up.rolling(_W, min_periods=8).sum()
    dn_dis = (abn * dn).rolling(_W, min_periods=8).sum() / n_dn.replace(0, np.nan)
    up_dis = (abn * up).rolling(_W, min_periods=8).sum() / n_up.replace(0, np.nan)

    df["rda_dn_disagree"] = np.clip(dn_dis.values, -1, 5)
    df["rda_resolution_asym"] = np.clip((up_dis - dn_dis).values, -3, 3)
    return df
