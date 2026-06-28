"""
osap_noa — Net Operating Assets (Operating Bloat)
Source: OpenSourceAP (Chen-Zimmermann); Hirshleifer, Hou, Teoh and Zhang 2004.

NOA = (Operating Assets - Operating Liabilities) / Total Assets
    = (rect + invt + ppent + aco + intan + ao - ap - lco - lo) / at

High NOA signals operating bloat: managers have over-invested in operating assets
relative to liabilities, predicting lower future returns (cross-sectional).

Per-ticker proxy: level + 12-month change in NOA (approximated from available PIT
fundamentals). Cross-sectional ranking unavailable in per-ticker mode; the level and
change are useful as absolute signals and capture the same economic content.
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
    "name": "osap_noa",
    "description": (
        "Net Operating Assets (NOA) as a share of total assets, following "
        "Hirshleifer, Hou, Teoh and Zhang (2004). NOA = (rect + invt + ppent + "
        "aco + intan + ao - ap - lco - lo) / at. High NOA (operating bloat) "
        "predicts lower cross-sectional returns. Implemented as a per-ticker "
        "absolute level (osap_noa_level) and 12-month change (osap_noa_chg) "
        "using PIT fundamentals; cross-sectional ranking not available per-ticker."
    ),
    "requires": [],
    "produces": ["osap_noa_level", "osap_noa_chg"],
    "tags": ["fundamental", "investment", "accounting", "operating_assets"],
    "version": "1.0",
    "author": "Hirshleifer, Hou, Teoh and Zhang 2004 (OpenSourceAP / Chen-Zimmermann)",
}

# Fields needed to construct NOA components.
# Operating Assets = receivables + inventory + PP&E + other current assets
#                    + intangibles + other assets
# Operating Liabilities = accounts payable + other current liabilities + other liabilities
# We approximate with what is available in the fundamentals panel:
#   rect      -> receivables
#   inventory -> inventory
#   ppe_net   -> net PP&E  (ppent)
#   goodwill  -> goodwill (part of intangibles; ao includes some of this)
#   assets_current -> act (current assets; aco = act - rect - inventory - cash approx)
#   liabilities_current -> lct (current liabilities; lco = lct - dlc approx)
#   long_term_debt -> dltt
#   assets    -> at (total assets)
#   liabilities -> lt (total liabilities, used as a cross-check but not in formula)
#
# Closest available mapping:
#   Operating Assets  ≈ receivables + inventory + ppe_net + goodwill
#                       + (assets_current - receivables - inventory - cash)  [aco proxy]
#                       + (assets - assets_current - ppe_net - goodwill)      [ao proxy]
#   Simplifies to:    ≈ assets - cash  (non-cash operating assets; standard simplification)
#   Operating Liabilities ≈ liabilities_current - (short-term portion of long-term debt)
#                            + (liabilities - liabilities_current - long_term_debt)
#   Simplifies to:    ≈ liabilities - total_debt
#
# Final formula implemented:
#   op_assets  = assets - cash
#   op_liab    = liabilities - total_debt   (non-financing liabilities)
#   NOA        = (op_assets - op_liab) / assets
#              = (assets - cash - liabilities + total_debt) / assets
#              = (equity + total_debt - cash) / assets   [= invested capital / assets]

_FIELDS = [
    "assets",
    "cash",
    "liabilities",
    "total_debt",
]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals (backward-merged, no lookahead)
    df = _fundamentals.as_of(df, fields=_FIELDS)

    assets = df["fund_assets"].replace(0, np.nan)
    cash = df["fund_cash"]
    liabilities = df["fund_liabilities"]
    total_debt = df["fund_total_debt"]

    # Operating assets (non-cash) and operating liabilities (non-financing)
    op_assets = assets - cash
    op_liab = liabilities - total_debt

    # NOA level: net operating assets scaled by total assets
    noa = (op_assets - op_liab) / assets
    # Replace any inf/-inf with NaN
    noa = noa.replace([np.inf, -np.inf], np.nan)
    df["osap_noa_level"] = noa

    # 12-month change in NOA (approx 252 trading days, but we use shift on the
    # already-merged series which updates at report filing dates)
    # Use a 252-bar shift as a robust annual-lag proxy
    df["osap_noa_chg"] = noa - noa.shift(252)
    df["osap_noa_chg"] = df["osap_noa_chg"].replace([np.inf, -np.inf], np.nan)

    # Drop scratch fund_ columns not in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
