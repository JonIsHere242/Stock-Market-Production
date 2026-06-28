"""
inx_insider_dollar_flow.py  --  EXTENDED insider (SEC Form 4) DOLLAR-magnitude flow features.

Distinct angle from the existing insdr_ blocks (_insider_activity.py / _insider_burst.py), which
score insider flow mostly on SHARE counts and a single 90-day net-dollar ratio. Here the whole family
is about the DOLLAR economics of the trades:

  * net-buy DOLLAR ratio at the LONGER 180/365-day horizons  -- the existing dollar ratio is 90d only
    (its 180/365 ratios are SHARE-based). Dollars weight large convicted purchases over many tiny ones.
  * dollar-vs-share DIVERGENCE: when the dollar ratio >> the share ratio, a few WHALES are buying;
    when it is below, breadth of small buyers dominates. Orthogonal to either ratio alone.
  * average BUY TICKET size (dollars per purchase) -- a per-transaction conviction scale, not a flow.
  * buy-ticket vs sell-ticket asymmetry -- are insiders buying in bigger clips than they sell?
  * sign of the exponentially-decayed net-dollar flow -- a clean direction gate (the existing block
    keeps only the signed-log MAGNITUDE; the bare sign is a different, sparser signal).

All values come from _insider.windowed_primitives (strictly BACKWARD aggregation on filed_date ->
lookahead-safe). The `_ins_*` scratch columns are dropped at the end so only inx_ columns remain.
Open-market P (purchase) / S (sale) dollars carry the signal; grants/exercises are excluded upstream.
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

_WINDOWS = (90, 180, 365)

METADATA = {
    "name":        "inx_insider_dollar_flow",
    "description": "Extended insider (Form 4) dollar-magnitude flow: long-horizon net-buy DOLLAR "
                   "ratios (180/365), dollar-vs-share whale divergence, average buy/sell ticket size "
                   "and their asymmetry, and exponentially-decayed net-dollar sign. Strictly PIT on "
                   "filed_date (backward as-of).",
    "requires":    ["Close"],
    "produces":    [
        "inx_ins_net_dollar_ratio_180",
        "inx_ins_net_dollar_ratio_365",
        "inx_ins_dollar_share_div_90",
        "inx_ins_avg_buy_ticket_90",
        "inx_ins_buy_sell_ticket_gap_90",
        "inx_ins_ewm_net_val_sign",
    ],
    "tags":        ["insider", "sec", "form4", "informed-flow", "dollar-flow", "experimental"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Safe num/den; NaN where the denominator is non-positive."""
    out = np.full(len(den), np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Lookahead-safe trailing-window insider primitives (backward on filed_date).
    df = _insider.windowed_primitives(df, windows=_WINDOWS)

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")

    # --- long-horizon net-buy DOLLAR ratios: (buy_val - sell_val)/(buy_val + sell_val) ---
    for w in (180, 365):
        bv, sv = col(f"_ins_buy_val_{w}"), col(f"_ins_sell_val_{w}")
        df[f"inx_ins_net_dollar_ratio_{w}"] = _ratio(bv - sv, bv + sv)

    # --- dollar-vs-share divergence at 90d: whales (>0) vs breadth-of-smalls (<0) ---
    bv90, sv90 = col("_ins_buy_val_90"), col("_ins_sell_val_90")
    bsh90, ssh90 = col("_ins_buy_sh_90"), col("_ins_sell_sh_90")
    dollar_ratio_90 = _ratio(bv90 - sv90, bv90 + sv90)
    share_ratio_90 = _ratio(bsh90 - ssh90, bsh90 + ssh90)
    df["inx_ins_dollar_share_div_90"] = dollar_ratio_90 - share_ratio_90

    # --- average BUY TICKET size (dollars per purchase), log-scaled for heavy tails ---
    bn90 = col("_ins_buy_n_90")
    avg_buy_ticket = _ratio(bv90, bn90)                       # NaN when no buys
    df["inx_ins_avg_buy_ticket_90"] = np.log1p(np.where(np.isnan(avg_buy_ticket), 0.0,
                                                        np.clip(avg_buy_ticket, 0.0, None)))

    # --- buy-ticket vs sell-ticket asymmetry: log(avg buy clip) - log(avg sell clip) ---
    sn90 = col("_ins_sell_n_90")
    avg_sell_ticket = _ratio(sv90, sn90)
    lbt = np.log1p(np.where(np.isnan(avg_buy_ticket), 0.0, np.clip(avg_buy_ticket, 0.0, None)))
    lst = np.log1p(np.where(np.isnan(avg_sell_ticket), 0.0, np.clip(avg_sell_ticket, 0.0, None)))
    df["inx_ins_buy_sell_ticket_gap_90"] = lbt - lst

    # --- direction-only sign of the exponentially-decayed net-dollar flow (sparse gate) ---
    df["inx_ins_ewm_net_val_sign"] = np.sign(col("_ins_ewm_net_val"))

    # Drop scratch so only METADATA["produces"] is added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("_ins_")])
    return df
