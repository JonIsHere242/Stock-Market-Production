"""
_ovn_vol_scaled_gap.py  --  CANDIDATE block for the `gap` arm (risk_adj-flavored).

Overnight INFORMATION RATIO: trailing mean overnight gap divided by gap-vol (mean gap per unit of
gap risk) at 21/63d, plus today's KNOWN gap normalized by its TRAILING overnight vol. Built purely on
the overnight (gap) return series as a RATIO -> the gap-flavored analogue of the risk_adj_topq target
(return/vol RANK). Because it DIVIDES OUT overnight vol it is the mechanical inverse of a vol-level
proxy, directly dodging the documented vol trap.

ovn_intraday_decomp carries only the UN-scaled mean overnight return; project notes flag "signals
built AS return-per-unit-risk" as explicit white-space. Trees construct ratios poorly, so the
explicit ratio is the lever.

Refs: Moreira & Muir (2017) JF 72(4):1611-1644 (volatility-managed portfolios); Barroso & Santa-Clara
(2015) JFE 116; Lou-Polk-Skouras (2019) JFE 134(1). Candidate -- NOT promoted until multi-seed gate.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_ovn_vol_scaled_gap",
    "description": "Overnight information ratio (trailing mean gap / gap-vol) at 21/63d plus today's "
                   "gap normalized by trailing overnight vol -- the gap-flavored return-per-unit-risk "
                   "signal. Moreira-Muir 2017 volatility-managed.",
    "requires":    ["Open", "Close"],
    "produces":    ["vsg_ir_21", "vsg_ir_63", "vsg_today_z_63"],
    "tags":        ["gap", "risk_adjusted", "candidate"],
    "version":     "0.1",
    "author":      "alt-target-feature-research 2026-06-26 (gap return-per-risk white-space)",
}

_EPS = 1e-9


def compute(df: pd.DataFrame) -> pd.DataFrame:
    on = (df["Open"] / df["Close"].shift(1) - 1.0).clip(-0.5, 0.5)   # causal overnight gap

    for W in (21, 63):
        mp = int(0.6 * W)
        mu = on.rolling(W, min_periods=mp).mean()
        sig = on.rolling(W, min_periods=mp).std()
        df[f"vsg_ir_{W}"] = (mu / (sig + _EPS)).clip(-5.0, 5.0)

    trail_sd = on.rolling(63, min_periods=35).std().shift(1)        # today's gap can't set its own scale
    df["vsg_today_z_63"] = (on / (trail_sd + _EPS)).clip(-6.0, 6.0)
    return df
