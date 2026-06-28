"""
Feature block: osap_dnoa — Change in Net Operating Assets (DNOA)
Source: Hirshleifer, Hou, Teoh, Zhang (2004); OpenSourceAP (Chen-Zimmermann)
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by file path — do NOT change this block)
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
    "name": "osap_dnoa",
    "description": (
        "Change in Net Operating Assets (DNOA) scaled by lagged total assets. "
        "NOA = (total_assets - cash) - (total_assets - long_term_debt - book_equity), "
        "which simplifies to long_term_debt + equity - cash. Items not available in the "
        "fundamentals panel (minority interest, deferred charges, preferred stock) are "
        "treated as 0 per the spec. DNOA_t = (NOA_t - NOA_{t-1}) / assets_{t-1}. "
        "Predicted sign: -1 (higher investment = lower future returns). "
        "This is a PIT accounting signal; coverage ~84% (ETFs/foreign = NaN). "
        "Also produces a 2-period momentum variant (dnoa_chg) and a price-scaled "
        "NOA level (noa_to_price). Per-ticker implementation; cross-sectional "
        "ranks are not applied here."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_dnoa_main",      # 12-month DNOA / lagged assets
        "osap_dnoa_chg",       # 1-period change in DNOA (acceleration)
        "osap_dnoa_noa_price", # NOA scaled by market cap proxy (Close * shares_outstanding)
    ],
    "tags": ["accounting", "investment", "fundamentals", "osap", "dnoa"],
    "version": "1.0",
    "author": "Hirshleifer, Hou, Teoh, Zhang (2004); OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DNOA and related features using PIT SEC fundamentals.

    NOA = long_term_debt + book_equity - cash
          (derived from the full formula with missing items set to 0)
    DNOA = (NOA_t - NOA_{t-1}) / assets_{t-1}

    The .as_of() merge is backward on filed_date, so fully lookahead-safe.
    """
    # --- pull PIT fundamentals --------------------------------------------------
    fields = ["assets", "cash", "long_term_debt", "equity", "shares_outstanding"]
    df = _fundamentals.as_of(df, fields=fields)

    # Convenience aliases
    at    = df["fund_assets"]           # total assets
    che   = df["fund_cash"]             # cash & short-term investments (proxy for che)
    dltt  = df["fund_long_term_debt"]   # long-term debt (dltt); mib/dlc/pstk → 0
    ceq   = df["fund_equity"]           # book equity
    shrs  = df["fund_shares_outstanding"]

    # Replace NaN denominators/missing items with 0 per spec for OL components
    dltt_filled = dltt.fillna(0.0)
    # Note: we do NOT fill at or ceq with 0 — if assets or equity is missing the
    # whole row stays NaN, which is correct behaviour.

    # --- Net Operating Assets ---------------------------------------------------
    # Full derivation:
    #   OA = at - che            (operating assets)
    #   OL = at - dltt - mib - dlc - ceq - pstk   (mib/dlc/pstk = 0 if missing)
    #      = at - dltt_filled - ceq
    #   NOA = OA - OL
    #       = (at - che) - (at - dltt_filled - ceq)
    #       = dltt_filled + ceq - che
    noa = dltt_filled + ceq - che

    # Lagged NOA and lagged total assets (prev observation = prior quarter filing)
    noa_lag1 = noa.shift(1)
    at_lag1  = at.shift(1)

    # --- DNOA (primary signal) -------------------------------------------------
    # Guard against zero / NaN denominator
    denom = at_lag1.replace(0.0, np.nan)
    dnoa  = (noa - noa_lag1) / denom        # raw 12-month change / lagged assets

    df["osap_dnoa_main"] = dnoa

    # --- DNOA change (acceleration = second difference) ------------------------
    dnoa_lag1 = dnoa.shift(1)
    df["osap_dnoa_chg"] = dnoa - dnoa_lag1

    # --- NOA-to-price (NOA / market cap proxy) ---------------------------------
    # market cap proxy = Close * shares_outstanding (fund_shares_outstanding in shares)
    # shares_outstanding typically reported in millions, so multiply by 1e6
    shrs_adj = shrs * 1e6
    mktcap   = (df["Close"] * shrs_adj).replace(0.0, np.nan)
    noa_price = noa / mktcap
    df["osap_dnoa_noa_price"] = noa_price

    # --- Clean up scratch fund_ columns ----------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=fund_cols, inplace=True)

    return df
