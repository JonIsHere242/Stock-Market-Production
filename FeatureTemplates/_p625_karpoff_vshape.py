"""
_p625_karpoff_vshape.py - Karpoff V-shape volume asymmetry.

Karpoff (1987, JFQA) "The Relation Between Price Changes and Trading Volume: A Survey";
Epps (1975, JF); Chen, Firth & Rui (2001). The volume-return relation is V-shaped and
asymmetric: volume responds more strongly to price increases than to equal-size decreases
(the "Epps effect"). We measure this directly via a 2-regressor closed-form rolling OLS of
log-volume on the up-move magnitude and the down-move magnitude:

  y = log(clip(Volume,1))  ~  intercept + b_up * max(r,0) + b_dn * max(-r,0)   (rolling 60d)

  kva_up_slope_60  = clip(b_up, -100, 100)   volume per unit gain
  kva_dn_slope_60  = clip(b_dn, -100, 100)   volume per unit loss
  kva_asym_60      = clip(b_up - b_dn, -100, 100)   up-vs-down volume asymmetry

Plus an extreme-move volume tilt: net volume on big-up vs big-down days, normalized by total
volume, using TRAILING rolling quantile thresholds (q80 / q20 over the same window):

  kva_extreme_tilt_20 = clip( (V[r>q80] - V[r<q20]).rolling(20).sum() / V.rolling(20).sum(), -1, 1 )

Strictly causal: all stats are trailing rolling windows; rolling-sum normal equations as in _suv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 60          # OLS window
_WE = 20         # extreme-tilt window

METADATA = {
    "name":        "_p625_karpoff_vshape",
    "description": "Karpoff V-shape volume asymmetry (Karpoff 1987 JFQA; Epps 1975 JF; Chen Firth Rui 2001): rolling 2-regressor OLS slopes of log-volume on up/down move magnitude (60d), their asymmetry, plus an extreme-move volume tilt (20d, trailing q80/q20).",
    "requires":    [],
    "produces":    ["kva_up_slope_60", "kva_dn_slope_60", "kva_asym_60", "kva_extreme_tilt_20"],
    "tags":        ["volume", "behavioral", "experimental"],
    "version":     "1.0",
    "author":      "paper:Karpoff (1987) JFQA; Epps (1975) JF; Chen, Firth & Rui (2001)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    r = close.pct_change()
    rp = r.clip(lower=0)
    rn = (-r).clip(lower=0)
    lv = np.log(vol.clip(lower=1.0))

    # ---- 2-regressor closed-form rolling OLS: lv ~ a0 + b_up*rp + b_dn*rn ----
    valid = np.isfinite(rp) & np.isfinite(rn) & np.isfinite(lv)
    rp = rp.where(valid)
    rn = rn.where(valid)
    lv = lv.where(valid)
    mp = int(_W * 0.6)

    def rs(s):
        return s.rolling(_W, min_periods=mp).sum()

    n = valid.astype(float).rolling(_W, min_periods=mp).sum().replace(0, np.nan)
    Sp, Sn, Sy = rs(rp), rs(rn), rs(lv)
    Spp, Snn, Spn = rs(rp * rp), rs(rn * rn), rs(rp * rn)
    Spy, Sny = rs(rp * lv), rs(rn * lv)

    cpp = Spp - Sp * Sp / n
    cnn = Snn - Sn * Sn / n
    cpn = Spn - Sp * Sn / n
    cpy = Spy - Sp * Sy / n
    cny = Sny - Sn * Sy / n
    det = (cpp * cnn - cpn * cpn).replace(0, np.nan)
    b_up = (cnn * cpy - cpn * cny) / det
    b_dn = (cpp * cny - cpn * cpy) / det

    df["kva_up_slope_60"] = np.clip(b_up.values, -100.0, 100.0)
    df["kva_dn_slope_60"] = np.clip(b_dn.values, -100.0, 100.0)
    df["kva_asym_60"] = np.clip((b_up - b_dn).values, -100.0, 100.0)

    # ---- extreme-move volume tilt: net big-up vs big-down volume / total ----
    # Trailing rolling quantile thresholds over the same extreme window.
    mpe = int(_WE * 0.6)
    q80 = r.rolling(_WE, min_periods=mpe).quantile(0.80)
    q20 = r.rolling(_WE, min_periods=mpe).quantile(0.20)

    up_vol = vol.where(r > q80, 0.0)
    dn_vol = vol.where(r < q20, 0.0)
    net_vol = (up_vol - dn_vol).rolling(_WE, min_periods=mpe).sum()
    tot_vol = vol.rolling(_WE, min_periods=mpe).sum().replace(0, np.nan)

    tilt = net_vol / tot_vol
    df["kva_extreme_tilt_20"] = np.clip(tilt.values, -1.0, 1.0)

    return df
