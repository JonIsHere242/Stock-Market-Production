"""
_fundamentals_valuation.py  --  DEMONSTRATOR / CANDIDATE feature block.

Leading underscore == UNPROVEN candidate: the framework does NOT auto-discover it, so it can't
leak into the model before it earns promotion (same convention generate_feature.py uses).

It shows the sanctioned way to build a SEC-fundamentals feature: pull point-in-time fundamentals
through the _fundamentals.as_of helper (a BACKWARD merge on filed_date -> lookahead-safe), then
combine them with the daily Close to form price-relative valuation multiples. The fund_* columns
are SCRATCH -- we compute the multiples and drop them, so only METADATA["produces"] is added.

Promote (drop the leading underscore) only after the heavy backtest gate.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Import the underscore helper by file path (it is hidden from normal discovery), exactly like
# vix_features.py imports _indexes.py.
_spec = _ilu.spec_from_file_location("_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py")
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "fundamentals_valuation",
    "description": "Price-relative valuation multiples (P/E, P/S, P/B, earnings yield, FCF yield) "
                   "from point-in-time SEC fundamentals (filed-date as-of) combined with Close.",
    "requires":    ["Close"],
    "produces":    ["pe_ratio_ttm", "ps_ratio_ttm", "pb_ratio", "earnings_yield_ttm", "fcf_yield_ttm"],
    "tags":        ["fundamentals", "valuation", "experimental"],
    "version":     "1.0",
    "author":      "demonstrator for the SEC fundamentals ingestion (build_fundamentals_panel.py)",
}

# Canonical fundamentals fields this block needs (helper returns them as fund_<field>).
_FIELDS = ["eps_diluted_ttm", "sales_per_share", "book_value_per_share",
           "fcf_ttm", "shares_outstanding"]


def _pos(s: pd.Series) -> pd.Series:
    """Keep only strictly-positive denominators; others -> NaN (a negative P/E is meaningless)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s > 0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Lookahead-safe PIT merge: each row sees only fundamentals filed on/before its Date.
    df = _fundamentals.as_of(df, fields=_FIELDS)

    eps   = _pos(df["fund_eps_diluted_ttm"])
    sps   = _pos(df["fund_sales_per_share"])
    bvps  = _pos(df["fund_book_value_per_share"])
    fcf   = pd.to_numeric(df["fund_fcf_ttm"], errors="coerce")
    sh    = _pos(df["fund_shares_outstanding"])

    df["pe_ratio_ttm"]       = close / eps
    df["ps_ratio_ttm"]       = close / sps
    df["pb_ratio"]           = close / bvps
    df["earnings_yield_ttm"] = eps / close            # inverse P/E: defined for negative earnings too via sign
    market_cap = close * sh
    df["fcf_yield_ttm"]      = fcf / market_cap.where(market_cap > 0)

    # Drop the scratch helper columns so only METADATA["produces"] is added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
