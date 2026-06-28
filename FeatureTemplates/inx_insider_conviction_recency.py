"""
inx_insider_conviction_recency.py  --  EXTENDED insider (Form 4) recency-decay, seniority-weighted
intensity, count-ratio, and dollar-based conviction-acceleration features.

Distinct angle from the existing insdr_ blocks:

  * RECENCY DECAY 1/(1+days_since_buy): the existing blocks keep raw days_since (a linear age that
    rolling-trees split awkwardly). A reciprocal-decay turns "a buy 2 days ago" into a near-1 spike
    and "a buy 300 days ago" into ~0 -- a smooth freshness weight. We also gate it by the net sign so
    fresh-NET-buying and fresh-NET-selling separate.
  * SENIORITY-WEIGHTED INTENSITY maxrank * buy_n: the most-senior buyer's rank times how many buys --
    a "how many people, how important" product. The existing rank features are value-weighted MEAN
    ranks (intensive); this is an EXTENSIVE seniority * activity scale, a different quantity.
  * BUY/SELL COUNT RATIO log(buy_n+1)-log(sell_n+1) at 90/365: the existing blocks use net SHARE and
    net DOLLAR ratios; a pure transaction-COUNT ratio is orthogonal -- it ignores size entirely and
    asks only "how many buy events vs sell events". Robust where sizes are noisy/missing.
  * CONVICTION ACCELERATION: net-DOLLAR ratio over the fresh 30d minus over 90d. The existing accel is
    net-SHARE 30d-180d; this dollar-based, shorter-baseline version captures whether recent dollar
    conviction is RAMPING relative to the medium term.

All from _insider.windowed_primitives (strictly BACKWARD on filed_date -> lookahead-safe). The
`_ins_*` scratch columns are dropped at the end so only inx_ columns remain.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_HERE = _Path(__file__).resolve().parent
_spec = _ilu.spec_from_file_location("_insider", _HERE / "_insider.py")
_insider = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_insider)

_WINDOWS = (30, 90, 365)

METADATA = {
    "name":        "inx_insider_conviction_recency",
    "description": "Extended insider (Form 4) conviction/recency: reciprocal recency-decay of the last "
                   "buy (raw and net-signed), seniority-weighted buy intensity (maxrank x buy_n) at "
                   "30/90d, log buy-to-sell COUNT ratios at 90/365d, and dollar-based 30d-vs-90d "
                   "conviction acceleration. Strictly PIT on filed_date (backward as-of).",
    "requires":    ["Close"],
    "produces":    [
        "inx_ins_recency_decay_buy",
        "inx_ins_recency_decay_net",
        "inx_ins_seniority_intensity_30",
        "inx_ins_seniority_intensity_90",
        "inx_ins_buy_sell_count_ratio_90",
        "inx_ins_buy_sell_count_ratio_365",
        "inx_ins_conviction_accel_30_90",
    ],
    "tags":        ["insider", "sec", "form4", "informed-flow", "conviction", "recency", "experimental"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(len(den), np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = _insider.windowed_primitives(df, windows=_WINDOWS)

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")

    # --- reciprocal recency decay of the most recent open-market buy ---
    dsb = col("_ins_days_since_buy")              # NaN before first buy
    decay_buy = np.where(np.isnan(dsb), 0.0, 1.0 / (1.0 + dsb))
    df["inx_ins_recency_decay_buy"] = decay_buy

    # net-signed recency: positive when 90d net DOLLAR flow is positive, negative when selling-heavy.
    bv90, sv90 = col("_ins_buy_val_90"), col("_ins_sell_val_90")
    net_sign = np.sign(_ratio(bv90 - sv90, bv90 + sv90))      # NaN where no flow
    df["inx_ins_recency_decay_net"] = decay_buy * np.nan_to_num(net_sign)

    # --- seniority-weighted buy intensity: most-senior buyer rank x number of buys ---
    for w in (30, 90):
        df[f"inx_ins_seniority_intensity_{w}"] = col(f"_ins_maxrank_{w}") * np.log1p(col(f"_ins_buy_n_{w}"))

    # --- log buy-to-sell COUNT ratio (size-agnostic) ---
    for w in (90, 365):
        df[f"inx_ins_buy_sell_count_ratio_{w}"] = (np.log1p(col(f"_ins_buy_n_{w}"))
                                                   - np.log1p(col(f"_ins_sell_n_{w}")))

    # --- dollar-based conviction acceleration: net-dollar ratio 30d minus 90d ---
    bv30, sv30 = col("_ins_buy_val_30"), col("_ins_sell_val_30")
    ndr30 = _ratio(bv30 - sv30, bv30 + sv30)
    ndr90 = _ratio(bv90 - sv90, bv90 + sv90)
    df["inx_ins_conviction_accel_30_90"] = np.nan_to_num(ndr30) - np.nan_to_num(ndr90)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_ins_")])
    return df
