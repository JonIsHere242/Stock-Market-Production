"""
_insider_burst.py  --  CANDIDATE block: the gauntlet-survivor insider features (burst + role-rank).

Leading underscore == UNPROVEN candidate (not auto-discovered; can't leak pre-promotion), same
convention as _insider_activity.py / _fundamentals_valuation.py. Promote only after the multi-seed
in-model ablation.

These 10 are the survivors of the EDA gauntlet (analysis_output/insider_expand_eda.py): they each
beat or matched the two parents on VOL/SIZE-NEUTRALIZED top-decile tail lift, and -- critically --
their lift PERSISTS through confound-neutralization (so they are not the vol-confound mirage that
killed the technical features). All composed from _insider.windowed_primitives (strictly BACKWARD on
filed_date) + Close/Volume; nothing here needs a builder change.

  insdr_burst_dom_5_30    buy_val_5/(buy_val_30+1)            parent F2  (neut 1.63)
  insdr_rank_conv_90      value-wtd role rank over 90d buys   parent F5  (neut 1.30)
  insdr_burst_on_dip      F2 * max(0,-ret21)                  E9  (neut 2.93)  <- informed dip-buying, the breakout
  insdr_burst_illiq       F2 / (log1p($vol)+1)                E10 (neut 1.92)  <- insider buys bite in illiquid names
  insdr_burst_n_5_30      buy_n_5/(buy_n_30+1)                E2  (neut 1.75)  <- count burst, denoises $ outliers
  insdr_maxrank_5         most-senior buyer in freshest 5d    E5  (neut 1.75)
  insdr_rankconv_burst_5  value-wtd role rank over 5d buys    E6  (neut 1.68)
  insdr_burst_x_rank      F2 * rank_conv_90                   E7  (neut 1.78)
  insdr_net_burst_5       (buy_sh_5-sell_sh_5)/(sum)          E3  (neut 1.42)  <- broad-coverage sign gate
  insdr_rank_x_breadth    rank_conv_90 * log1p(buy_n_90)      E12 (neut 1.33)

CAVEAT carried from the EDA: insdr_burst_on_dip's dip gate is ret21, which overlaps medium-term
REVERSAL features the model may already have -- the insider burst is the new part; the multi-seed
ablation arbitrates its marginal contribution.
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

_WINDOWS = (5, 30, 90)

METADATA = {
    "name":        "insider_burst",
    "description": "Insider (Form 4) burst + role-rank signals that survived the vol-neutralized tail "
                   "gauntlet: 5-day buying-burst dominance, role-rank conviction, and their denoising "
                   "combinations (burst-into-dip, burst-in-illiquid, count-burst, freshest seniority). "
                   "Strictly point-in-time on filed_date (backward as-of).",
    "requires":    ["Close", "Volume"],
    "produces":    [
        "insdr_burst_dom_5_30", "insdr_rank_conv_90", "insdr_burst_on_dip", "insdr_burst_illiq",
        "insdr_burst_n_5_30", "insdr_maxrank_5", "insdr_rankconv_burst_5", "insdr_burst_x_rank",
        "insdr_net_burst_5", "insdr_rank_x_breadth",
    ],
    "tags":        ["insider", "sec", "form4", "burst", "informed-flow", "experimental"],
    "version":     "1.0",
    "author":      "insider burst/rank candidate — EDA gauntlet survivors (insider_expand_eda.py)",
}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(len(den), np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce")

    df = _insider.windowed_primitives(df, windows=_WINDOWS)

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")

    bv5, bv30, bv90 = col("_ins_buy_val_5"), col("_ins_buy_val_30"), col("_ins_buy_val_90")
    bn5, bn30, bn90 = col("_ins_buy_n_5"), col("_ins_buy_n_30"), col("_ins_buy_n_90")
    bsh5, ssh5 = col("_ins_buy_sh_5"), col("_ins_sell_sh_5")
    rw5, rlw5 = col("_ins_rank_w_5"), col("_ins_rank_lw_5")
    rw90, rlw90 = col("_ins_rank_w_90"), col("_ins_rank_lw_90")

    # parents
    burst = bv5 / (bv30 + 1.0)                              # F2
    rank_conv_90 = _ratio(rw90, rlw90)                      # F5 (NaN where no 90d buys)
    df["insdr_burst_dom_5_30"] = burst
    df["insdr_rank_conv_90"] = rank_conv_90

    # denoising combinations with price / liquidity (the EDA breakouts)
    ret21 = (close / close.shift(21) - 1.0).to_numpy()
    df["insdr_burst_on_dip"] = burst * np.clip(-ret21, 0.0, None)                 # E9
    logdvol = np.log1p((close * vol).clip(lower=0).to_numpy())
    df["insdr_burst_illiq"] = burst / (logdvol + 1.0)                            # E10

    # window/encoding variants
    df["insdr_burst_n_5_30"] = bn5 / (bn30 + 1.0)                                # E2
    df["insdr_maxrank_5"] = col("_ins_maxrank_5")                                # E5
    df["insdr_rankconv_burst_5"] = _ratio(rw5, rlw5)                             # E6
    df["insdr_burst_x_rank"] = burst * np.nan_to_num(rank_conv_90)               # E7
    df["insdr_net_burst_5"] = _ratio(bsh5 - ssh5, bsh5 + ssh5)                   # E3
    df["insdr_rank_x_breadth"] = np.nan_to_num(rank_conv_90) * np.log1p(bn90)    # E12

    # drop scratch so only METADATA["produces"] is added
    df = df.drop(columns=[c for c in df.columns if c.startswith(("_ins_", "fund_"))])
    return df
