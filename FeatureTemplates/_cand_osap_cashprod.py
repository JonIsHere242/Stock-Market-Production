"""
Cash Productivity feature block.

Chandrashekar & Rao (2009) via OpenSourceAP (Chen-Zimmermann).

Cash Productivity = (Market Cap - Total Assets) / Cash
  = (Close * shares_outstanding - assets) / cash

Cross-sectional predicted sign: -1 (long LOW cash-productivity stocks).

Per-ticker implementation: all inputs are available from PIT fundamentals
(shares_outstanding, assets, cash) merged to daily Close -- fully lookahead-safe
via backward merge_asof on filed_date.

Produces:
  osap_cashprod_level  -- raw ratio (level)
  osap_cashprod_chg    -- 63-trading-day change in the ratio (captures when
                          cash productivity is deteriorating/improving)
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
    "name": "osap_cashprod",
    "description": (
        "Cash Productivity (Chandrashekar & Rao 2009, OpenSourceAP / Chen-Zimmermann). "
        "Defined as (Market Cap - Total Assets) / Cash, where Market Cap = "
        "Close * shares_outstanding (PIT). Predicted cross-sectional sign: -1 "
        "(low cash-productivity stocks outperform). "
        "Implemented as a per-ticker time series using PIT SEC fundamentals; "
        "coverage ~84% (ETFs/foreign return NaN). "
        "osap_cashprod_level is the raw ratio; osap_cashprod_chg is the 63-day "
        "change (one fiscal quarter lag approximation)."
    ),
    "requires": ["Close"],
    "produces": ["osap_cashprod_level", "osap_cashprod_chg"],
    "tags": ["fundamentals", "profitability", "cash", "osap"],
    "version": "1.0",
    "author": "Chandrashekar & Rao (2009) via OpenSourceAP (Chen-Zimmermann); block impl by Claude",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cash Productivity columns for a single ticker."""

    # Merge PIT fundamentals -- backward-safe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["shares_outstanding", "assets", "cash"])

    so_col = "fund_shares_outstanding"
    at_col = "fund_assets"
    ch_col = "fund_cash"

    # Market cap (Close * shares_outstanding)
    mve = df["Close"] * df[so_col]

    # Numerator: market cap - total assets
    numerator = mve - df[at_col]

    # Denominator: cash (guard against zero / negative)
    cash = df[ch_col].copy()
    cash = cash.where(cash > 0, np.nan)  # cash <= 0 → NaN (undefined)

    level = numerator / cash

    # Replace any inf/-inf that might slip through
    level = level.replace([np.inf, -np.inf], np.nan)

    df["osap_cashprod_level"] = level

    # 63-bar change (approx one fiscal quarter in trading days)
    df["osap_cashprod_chg"] = level.diff(63)

    # Drop scratch fund_* columns not listed in produces
    for col in [so_col, at_col, ch_col]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
