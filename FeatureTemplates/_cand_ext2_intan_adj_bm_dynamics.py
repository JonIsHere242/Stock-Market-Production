"""
Feature block: ext2_intan_adj_bm_dynamics
Dynamics of intangible-adjusted book-to-market.

Organizational capital (OC) and R&D capital (RDC) are capitalized via perpetual inventory
methods. Both are added to book equity to form intangible-adjusted book equity, which is
then divided by market cap to get intangible-adjusted B/M. The produced signals are:
  - 126-day change in intangible-adjusted B/M (value-momentum signal)
  - spread between intangible-adjusted B/M change and plain B/M change (intangible-specific momentum)
  - the intangible-adjusted B/M level itself

All computations are per-ticker, causal, and use PIT fundamentals.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load PIT fundamentals helper
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)


METADATA = {
    "name": "ext2_intan_adj_bm_dynamics",
    "description": (
        "Intangible-adjusted book-to-market dynamics. Capitalizes SG&A (organizational capital, OC) "
        "at 30%/yr decay and R&D (RDC) at 20%/yr decay via perpetual inventory, then adds both to "
        "reported equity to form intangible-adjusted book equity. Produces: (1) level of intangible-adj "
        "B/M, (2) 126-day change in intangible-adj B/M (value-signal momentum), (3) spread between "
        "intangible-adj B/M change and plain B/M change (intangible-specific momentum). Per-ticker proxy "
        "using PIT fundamentals; cross-sectional rank not available at this layer."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_intan_adj_bm_dynamics_level",
        "ext2_intan_adj_bm_dynamics_chg126",
        "ext2_intan_adj_bm_dynamics_spread126",
    ],
    "tags": ["fundamentals", "value", "intangibles", "momentum", "book_to_market"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of osap_orgcap winner vein (extends parent feature osap_orgcap)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intangible-adjusted B/M level and dynamics."""

    # Pull PIT fundamental fields needed
    needed = [
        "equity",
        "shares_outstanding",
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
    ]
    df = _fundamentals.as_of(df, fields=needed)

    # ------------------------------------------------------------------ #
    # 1. Quarterly SG&A proxy (from TTM income statement items)           #
    #    SGA_quarterly = (gross_profit_ttm - operating_income_ttm         #
    #                     - rnd_expense_ttm).clip(0) / 4                  #
    # ------------------------------------------------------------------ #
    gp = df["fund_gross_profit_ttm"].fillna(method="ffill")
    oi = df["fund_operating_income_ttm"].fillna(method="ffill")
    rnd = df["fund_rnd_expense_ttm"].fillna(method="ffill")

    # SG&A quarterly add; clipped >= 0
    sga_q = ((gp - oi - rnd).clip(lower=0)) / 4.0
    rnd_q = rnd / 4.0

    # ------------------------------------------------------------------ #
    # 2. Perpetual inventory for Organizational Capital (OC)              #
    #    Annual decay 30% => quarterly delta = 1 - (0.7)^0.25            #
    # ------------------------------------------------------------------ #
    delta_oc = 1.0 - (0.7 ** 0.25)  # ~0.08182

    # Seed with first valid SGA observation
    sga_arr = sga_q.to_numpy(dtype=float)
    oc_arr = np.full(len(sga_arr), np.nan)

    seed_set = False
    oc_prev = np.nan
    for i in range(len(sga_arr)):
        s = sga_arr[i]
        if np.isnan(s):
            if not seed_set:
                oc_arr[i] = np.nan
            else:
                # carry forward with decay only (no new add)
                oc_prev = (1.0 - delta_oc) * oc_prev
                oc_arr[i] = oc_prev
        else:
            if not seed_set:
                # seed: OC_0 = SGA_0 / (0.025 + delta)
                oc_prev = s / (0.025 + delta_oc)
                seed_set = True
            else:
                oc_prev = (1.0 - delta_oc) * oc_prev + s
            oc_arr[i] = oc_prev

    # ------------------------------------------------------------------ #
    # 3. Perpetual inventory for R&D Capital (RDC)                       #
    #    Annual decay 20% => quarterly delta = 1 - (0.8)^0.25            #
    # ------------------------------------------------------------------ #
    delta_rdc = 1.0 - (0.8 ** 0.25)  # ~0.05132

    rnd_arr = rnd_q.to_numpy(dtype=float)
    rdc_arr = np.full(len(rnd_arr), np.nan)

    seed_set_r = False
    rdc_prev = np.nan
    for i in range(len(rnd_arr)):
        r = rnd_arr[i]
        if np.isnan(r):
            if not seed_set_r:
                rdc_arr[i] = np.nan
            else:
                rdc_prev = (1.0 - delta_rdc) * rdc_prev
                rdc_arr[i] = rdc_prev
        else:
            if not seed_set_r:
                rdc_prev = r / (0.025 + delta_rdc)
                seed_set_r = True
            else:
                rdc_prev = (1.0 - delta_rdc) * rdc_prev + r
            rdc_arr[i] = rdc_prev

    # ------------------------------------------------------------------ #
    # 4. Intangible-adjusted book equity & B/M                           #
    # ------------------------------------------------------------------ #
    equity = df["fund_equity"].fillna(method="ffill").to_numpy(dtype=float)
    shares = df["fund_shares_outstanding"].fillna(method="ffill").to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    mktcap = close * shares
    # Guard divide by zero / missing
    mktcap_safe = np.where((mktcap == 0) | np.isnan(mktcap), np.nan, mktcap)

    # Intangible-adjusted book equity
    intan_equity = equity + oc_arr + rdc_arr  # may be NaN when components NaN

    # Plain B/M = equity / mktcap
    plain_bm = np.where(np.isnan(equity), np.nan, equity / mktcap_safe)

    # Intangible-adjusted B/M
    intan_bm = np.where(np.isnan(intan_equity), np.nan, intan_equity / mktcap_safe)

    # ------------------------------------------------------------------ #
    # 5. 126-day changes (approx 6 months)                               #
    # ------------------------------------------------------------------ #
    intan_bm_s = pd.Series(intan_bm, index=df.index)
    plain_bm_s = pd.Series(plain_bm, index=df.index)

    intan_bm_chg126 = intan_bm_s - intan_bm_s.shift(126)
    plain_bm_chg126 = plain_bm_s - plain_bm_s.shift(126)

    # Spread: how much of the change is driven by intangibles vs plain
    spread126 = intan_bm_chg126 - plain_bm_chg126

    # ------------------------------------------------------------------ #
    # 6. Guard inf / -inf across all outputs                             #
    # ------------------------------------------------------------------ #
    def _clean(arr):
        return np.where(np.isinf(arr), np.nan, arr)

    df["ext2_intan_adj_bm_dynamics_level"] = _clean(intan_bm)
    df["ext2_intan_adj_bm_dynamics_chg126"] = _clean(intan_bm_chg126.to_numpy(dtype=float))
    df["ext2_intan_adj_bm_dynamics_spread126"] = _clean(spread126.to_numpy(dtype=float))

    # Drop scratch fund_ columns not listed in produces
    scratch_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch_cols)

    return df
