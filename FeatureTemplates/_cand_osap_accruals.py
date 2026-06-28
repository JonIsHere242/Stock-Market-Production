"""
Accruals feature block (Sloan 1996 / OpenSourceAP Chen-Zimmermann).

Per-ticker proxy using PIT fundamentals.  The classic formula requires
balance-sheet line items not fully available here (debt-in-current-liabilities
dlc, income-taxes-payable txp), so those components are omitted and noted in
description.  Coverage ~84% of universe (ETFs / foreign = NaN, as expected).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_accruals",
    "description": (
        "Sloan (1996) balance-sheet accruals scaled by average total assets. "
        "Per-ticker PIT proxy: accruals = (Δassets_current - Δcash - Δliabilities_current) "
        "/ avg(assets).  dlc (debt-in-current-liabilities) and txp (taxes-payable) are "
        "omitted because they are not in the available fundamentals fields; the remaining "
        "components capture most of the accruals variation.  Predicted sign = -1 "
        "(high accruals → lower future returns).  Produces: "
        "osap_accruals_val (point-in-time accruals ratio), "
        "osap_accruals_chg (annual change in accruals ratio, trend signal), "
        "osap_accruals_mag (|accruals|, magnitude/extremity screen)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_accruals_val",
        "osap_accruals_chg",
        "osap_accruals_mag",
    ],
    "tags": ["accruals", "fundamentals", "accounting", "sloan1996", "balance-sheet"],
    "version": "1.0",
    "author": "Sloan 1996 / OpenSourceAP (Chen-Zimmermann); per-ticker PIT proxy implementation",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_accruals_* columns to df (one ticker, ascending Date)."""

    # --- 1. Attach PIT fundamentals ------------------------------------------
    fields = ["assets_current", "cash", "liabilities_current", "assets"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=fields)

    # Convenience aliases
    act = df["fund_assets_current"]   # current total assets
    che = df["fund_cash"]             # cash & short-term investments
    lct = df["fund_liabilities_current"]  # current liabilities
    at  = df["fund_assets"]           # total assets (denominator)

    # --- 2. Annual (lagged) deltas -------------------------------------------
    # Fundamentals update at most quarterly; a 252-row lag (≈1 yr) is the
    # cleanest annual diff without look-ahead.  We use shift(252) as the
    # "prior year" anchor — fundamentals are already PIT so this is safe.
    LAG = 252

    delta_act = act - act.shift(LAG)
    delta_che = che - che.shift(LAG)
    delta_lct = lct - lct.shift(LAG)

    # Average total assets over the period
    avg_at = (at + at.shift(LAG)) / 2.0

    # --- 3. Accruals ratio ---------------------------------------------------
    # accruals_val: negative = earnings heavily cash-backed (good); positive = accrual-heavy (bad)
    numerator = delta_act - delta_che - delta_lct
    accruals_val = numerator / avg_at.replace(0, np.nan)

    # Clip to [-2, 2] to remove data / stale-filing artefacts
    accruals_val = accruals_val.clip(-2.0, 2.0)

    # Replace inf/-inf
    accruals_val = accruals_val.replace([np.inf, -np.inf], np.nan)

    # --- 4. Dynamic variants -------------------------------------------------
    # Year-over-year change in the accruals ratio (momentum/trend of accruals)
    accruals_chg = accruals_val - accruals_val.shift(LAG)
    accruals_chg = accruals_chg.clip(-2.0, 2.0).replace([np.inf, -np.inf], np.nan)

    # Magnitude screen (abs accruals — high = distorted earnings in either direction)
    accruals_mag = accruals_val.abs()

    # --- 5. Write produced columns -------------------------------------------
    df["osap_accruals_val"] = accruals_val
    df["osap_accruals_chg"] = accruals_chg
    df["osap_accruals_mag"] = accruals_mag

    # --- 6. Drop scratch fund_ columns not in produces -----------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
