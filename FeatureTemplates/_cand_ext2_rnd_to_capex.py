"""
ext2_rnd_to_capex — Innovation vs physical investment mix.

Measures the ratio of R&D spend to capital expenditure and an R&D intensity
share (rnd / (rnd + capex)), both sourced from point-in-time TTM fundamentals.
A 252-day change captures the trend in a firm's mix between intangible and
physical investment over the prior year.

Per-ticker proxy: uses PIT _fundamentals.as_of; no cross-sectional component.
Coverage ~84% (ETFs/foreign → NaN, expected).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_rnd_to_capex",
    "description": (
        "Innovation vs physical investment mix. "
        "Produces: (1) rnd_expense_ttm / capex_ttm (innovation-to-capex ratio), "
        "(2) rnd_expense_ttm / (rnd_expense_ttm + capex_ttm) (R&D intensity share), "
        "(3) 252-day change in the R&D intensity share. "
        "All values are point-in-time (filed_date backward merge) to avoid lookahead. "
        "Negative or zero capex is treated as NaN in the ratio. "
        "Per-ticker proxy — no cross-sectional ranking."
    ),
    "requires": [],
    "produces": [
        "ext2_rnd_to_capex_ratio",
        "ext2_rnd_to_capex_share",
        "ext2_rnd_to_capex_share_chg252",
    ],
    "tags": ["fundamentals", "innovation", "capex", "rnd", "investment_mix"],
    "version": "1.0.0",
    "author": "Round-3 deep exploration of a rich winner vein (osap_orgcap)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull TTM R&D and capex; as_of does backward merge on filed_date (PIT-safe)
    df = _fundamentals.as_of(df, fields=["rnd_expense_ttm", "capex_ttm"])

    rnd = df["fund_rnd_expense_ttm"]
    capex = df["fund_capex_ttm"]

    # capex is often reported as a negative cash-flow number; use absolute value
    capex_abs = capex.abs()

    # Guard: zero or missing capex -> NaN ratio
    safe_capex = capex_abs.where(capex_abs > 0, other=np.nan)
    safe_denom = (rnd + capex_abs).where((rnd + capex_abs) > 0, other=np.nan)

    # (1) rnd / |capex|  — innovation-to-physical-investment ratio
    df["ext2_rnd_to_capex_ratio"] = rnd / safe_capex

    # (2) rnd / (rnd + |capex|)  — R&D intensity share in [0, 1]
    share = rnd / safe_denom
    df["ext2_rnd_to_capex_share"] = share

    # (3) 252-day change in share  — captures drift in innovation mix
    df["ext2_rnd_to_capex_share_chg252"] = share.diff(252)

    # Drop scratch fund_ columns we are NOT publishing
    df = df.drop(columns=["fund_rnd_expense_ttm", "fund_capex_ttm"], errors="ignore")

    return df
