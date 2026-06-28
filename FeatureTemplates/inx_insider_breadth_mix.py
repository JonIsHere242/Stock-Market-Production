"""
inx_insider_breadth_mix.py  --  EXTENDED insider (Form 4) breadth-scale, role-MIX, cluster-strength,
and grant-overhang features.

Distinct angle from the existing insdr_ blocks (which keep a unique_buyers COUNT, a >=2-buyer cluster
flag, a buyer-vs-seller breadth RATIO, and a purchase-vs-grant ratio):

  * BREADTH SCALE log1p(unique_buyers_90): a log-compressed count -- the existing block keeps the raw
    count and a breadth ratio; the log scale separates "1 buyer" / "few" / "many" more cleanly for a
    tree and is robust to the heavy right tail of crowded names.
  * STRONG CLUSTER FLAG (>=3 distinct buyers in 90d): a stricter conviction bar than the existing >=2
    flag -- three independent insiders buying is a materially rarer, higher-conviction cluster event.
  * OFFICER-vs-DIRECTOR BUY MIX officer_buy/(officer_buy + dir_buy) at 90/365: WHO is buying. Officer
    (operating-side) buying is more informative than outside-director buying; the mix is orthogonal to
    the count/breadth families and to the existing role FLAGS (which only test CEO/CFO presence).
  * BREADTH x CONVICTION = unique_buyers_90 * net-dollar-ratio_90: many DISTINCT insiders AND a
    positive dollar tilt -- a consensus-conviction interaction not present in either parent.
  * GRANT OVERHANG = log(grant_sh) - log(buy_sh) at 90d: routine compensation grants relative to
    open-market conviction buying. High overhang = dilution/compensation dominates, a different read
    than the existing purchase-vs-grant fraction.

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

_WINDOWS = (90, 365)

METADATA = {
    "name":        "inx_insider_breadth_mix",
    "description": "Extended insider (Form 4) breadth/role: log breadth of distinct 90d buyers, a "
                   "strict >=3-buyer cluster flag, officer-vs-director buy mix at 90/365d, a "
                   "breadth-x-net-dollar consensus interaction, and grant-vs-buy overhang. Strictly "
                   "PIT on filed_date (backward as-of).",
    "requires":    ["Close"],
    "produces":    [
        "inx_ins_buyer_breadth_log_90",
        "inx_ins_strong_cluster_flag_90",
        "inx_ins_officer_dir_mix_90",
        "inx_ins_officer_dir_mix_365",
        "inx_ins_breadth_x_conviction_90",
        "inx_ins_grant_overhang_90",
    ],
    "tags":        ["insider", "sec", "form4", "informed-flow", "breadth", "role-mix", "experimental"],
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

    # --- log breadth of distinct buyers over the breadth window (90d in the helper) ---
    ub = col("_ins_unique_buyers_90")
    df["inx_ins_buyer_breadth_log_90"] = np.log1p(np.clip(ub, 0.0, None))

    # --- strict cluster: >=3 distinct insiders buying within 90d ---
    df["inx_ins_strong_cluster_flag_90"] = (ub >= 3).astype("float64")

    # --- officer-vs-director buy mix: 1.0 = all-officer, 0.0 = all-director, NaN when neither ---
    for w in (90, 365):
        off, dirb = col(f"_ins_officer_buy_{w}"), col(f"_ins_dir_buy_{w}")
        df[f"inx_ins_officer_dir_mix_{w}"] = _ratio(off, off + dirb)

    # --- breadth x conviction: distinct buyers times the 90d net-dollar tilt ---
    bv90, sv90 = col("_ins_buy_val_90"), col("_ins_sell_val_90")
    ndr90 = _ratio(bv90 - sv90, bv90 + sv90)
    df["inx_ins_breadth_x_conviction_90"] = ub * np.nan_to_num(ndr90)

    # --- grant overhang: routine grant shares vs open-market buy shares (log gap) ---
    gsh, bsh = col("_ins_grant_sh_90"), col("_ins_buy_sh_90")
    df["inx_ins_grant_overhang_90"] = (np.log1p(np.clip(gsh, 0.0, None))
                                      - np.log1p(np.clip(bsh, 0.0, None)))

    df = df.drop(columns=[c for c in df.columns if c.startswith("_ins_")])
    return df
