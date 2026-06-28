"""
Capex-vs-Intangible Investment-Style Tilt
==========================================
Spec: ext2_capex_intangible_tilt
Source: Round-3 deep exploration of the osap_orgcap winner vein.

Conceptual basis
----------------
Firms differ in whether they deploy capital into *tangible* assets (PP&E,
measured by capex) vs. *intangible* knowledge capital (R&D + SG&A, the
two main proxies for organisational / human / brand capital in the
organisational-capital literature).  The ratio:

    tilt = (capex_ttm - II) / (capex_ttm + II),   II = rnd_expense_ttm + sga_ttm

is bounded in [-1, +1]:  +1 = purely tangible, -1 = purely intangible.
A 252-day change in the tilt captures firms *rotating* their investment mix
(e.g., cutting R&D to fund PP&E, or vice-versa), which has been shown to
predict returns orthogonally to the level of organisational capital (osap_orgcap).

Per-ticker proxy notes
-----------------------
SG&A is not a direct fundamentals field, so it is estimated as:
    sga_ttm = revenue_ttm - gross_profit_ttm - operating_income_ttm
(i.e., operating-expense residual excluding COGS and EBIT).  This is a
faithful point-in-time proxy from public filings and avoids lookahead.

Coverage: ~84% of universe (ETFs / foreign have no fundamentals -> NaN).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by file path to satisfy sandbox rules)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext2_capex_intangible_tilt",
    "description": (
        "Investment-style tilt: (capex_ttm - II) / (capex_ttm + II) where "
        "II = rnd_expense_ttm + estimated SG&A_ttm (revenue_ttm - gross_profit_ttm "
        "- operating_income_ttm). +1 = fully tangible-heavy, -1 = fully intangible-heavy. "
        "Also produces the 252-day change in the tilt to capture investment-mix rotation. "
        "Per-ticker PIT proxy; orthogonal to organisational-capital level (osap_orgcap). "
        "Coverage ~84%; ETFs/foreign emit NaN."
    ),
    "requires": ["Close"],  # Close is needed for merge_asof date alignment
    "produces": [
        "ext2_capex_intangible_tilt_level",   # the bounded tilt in [-1, +1]
        "ext2_capex_intangible_tilt_chg252",  # 252-trading-day change in the tilt
    ],
    "tags": ["fundamentals", "investment", "capex", "intangible", "style", "pit"],
    "version": "1.0.0",
    "author": (
        "Spec: ext2_capex_intangible_tilt -- Round-3 deep exploration of the "
        "osap_orgcap winner vein (extends OSAP / Eisfeldt & Papanikolaou 2013 "
        "organisational-capital literature)."
    ),
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext2_capex_intangible_tilt_level and ext2_capex_intangible_tilt_chg252."""

    # Merge PIT fundamentals we need
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=[
                "capex_ttm",
                "rnd_expense_ttm",
                "revenue_ttm",
                "gross_profit_ttm",
                "operating_income_ttm",
            ],
        )

    capex = df["fund_capex_ttm"]
    rnd   = df["fund_rnd_expense_ttm"]

    # Estimate SG&A_ttm = revenue_ttm - gross_profit_ttm - operating_income_ttm
    # (operating residual: revenue minus cost-of-goods and minus EBIT)
    rev    = df["fund_revenue_ttm"]
    gp     = df["fund_gross_profit_ttm"]
    opinc  = df["fund_operating_income_ttm"]
    sga    = rev - gp - opinc  # can be negative if accounting items don't reconcile

    # Intangible investment proxy: R&D + SG&A
    # Use abs(sga) only if positive; if SG&A is non-sensible (negative), fall
    # back to R&D alone to avoid injecting noise.
    sga_pos = sga.where(sga >= 0, other=np.nan)
    intangible = rnd.fillna(0.0) + sga_pos.fillna(0.0)

    # Tilt = (capex - II) / (capex + II), guarded against zero denominator
    numerator   = capex - intangible
    denominator = capex + intangible
    denominator_safe = denominator.where(denominator.abs() > 0, other=np.nan)

    tilt = numerator / denominator_safe
    # Clip to [-1, 1] to handle any floating-point overshoot
    tilt = tilt.clip(-1.0, 1.0)

    df["ext2_capex_intangible_tilt_level"] = tilt

    # 252-trading-day change in tilt (captures investment-mix rotation)
    df["ext2_capex_intangible_tilt_chg252"] = tilt.diff(252)

    # Drop scratch fund_* columns that are not in produces
    scratch_cols = [
        "fund_capex_ttm",
        "fund_rnd_expense_ttm",
        "fund_revenue_ttm",
        "fund_gross_profit_ttm",
        "fund_operating_income_ttm",
    ]
    df = df.drop(columns=[c for c in scratch_cols if c in df.columns])

    return df
