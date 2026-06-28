"""
Candidate feature block: ext2_knowledge_cap_to_market
Knowledge (R&D) capital to market value -- per-ticker proxy using PIT fundamentals.

R&D capital (RDC) is capitalized via perpetual inventory method at 20%/yr depreciation.
RDC is then related to market cap and total assets to produce a knowledge-capital yield
(value signal specific to intangible/R&D intensity).

Per-ticker; uses _fundamentals.as_of for PIT-safe rnd_expense_ttm, assets, shares_outstanding.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PIT fundamentals helper
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
    "name": "ext2_knowledge_cap_to_market",
    "description": (
        "Knowledge (R&D) capital to market value. "
        "Capitalizes rnd_expense_ttm via a perpetual inventory model at 20%/yr depreciation "
        "(quarterly delta = 1 - 0.8^0.25; each quarter adds rnd_expense_ttm/4 and depreciates the stock). "
        "Produces: (1) RDC / (Close * shares_outstanding) -- knowledge-capital yield, a value signal "
        "orthogonal to book value since R&D is expensed not capitalised under GAAP; "
        "(2) RDC / assets -- intangible intensity ratio; "
        "(3) 252-trading-day change in knowledge-capital yield (momentum/growth signal). "
        "Per-ticker proxy -- cross-sectional ranking not available here. "
        "Stocks without R&D data (ETFs, financials, foreign ADRs) produce NaN throughout."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_knowledge_cap_to_market_yield",
        "ext2_knowledge_cap_to_market_asset_ratio",
        "ext2_knowledge_cap_to_market_yield_chg252",
    ],
    "tags": ["fundamentals", "rnd", "intangibles", "value", "knowledge_capital"],
    "version": "1.0",
    "author": "Spec: Round-3 deep exploration of osap_orgcap winner vein; impl by Claude Code",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ANNUAL_DEPR = 0.20                         # 20% per year depreciation rate
_QUARTERLY_SURVIVAL = (1.0 - _ANNUAL_DEPR) ** 0.25  # per-quarter survival factor
_QUARTERLY_DEPR_DELTA = 1.0 - _QUARTERLY_SURVIVAL   # per-quarter depreciation fraction (~5.13%)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R&D capital (perpetual inventory) and relate to market cap & assets.

    Steps:
    1. Pull PIT fundamentals: rnd_expense_ttm, shares_outstanding, assets.
    2. Derive quarterly additions: rnd_expense_ttm / 4 (proxy for quarterly R&D spend).
    3. Walk the time series to build a perpetual-inventory RDC stock.
       RDC[t] = RDC[t-1] * (1 - quarterly_depr) + quarterly_add[t]
       We compute this on the DAILY series by treating every row as a potential
       new quarter boundary (fundamentals update when filed_date changes).
       Between fundamental updates the RDC continues to depreciate daily.
    4. Normalise:
       - yield  = RDC / (Close * shares_outstanding)
       - ratio  = RDC / assets
       - chg252 = yield - yield.shift(252)
    """
    if df.empty:
        df["ext2_knowledge_cap_to_market_yield"] = np.nan
        df["ext2_knowledge_cap_to_market_asset_ratio"] = np.nan
        df["ext2_knowledge_cap_to_market_yield_chg252"] = np.nan
        return df

    # Pull PIT fundamentals (backward-safe merge_asof on filed_date inside as_of)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df, fields=["rnd_expense_ttm", "shares_outstanding", "assets"]
        )

    rnd_ttm = df["fund_rnd_expense_ttm"].values.copy().astype(float)
    shares   = df["fund_shares_outstanding"].values.copy().astype(float)
    assets   = df["fund_assets"].values.copy().astype(float)
    close    = df["Close"].values.copy().astype(float)

    n = len(df)

    # -----------------------------------------------------------------------
    # Perpetual-inventory RDC on daily bars.
    #
    # Fundamentals (rnd_expense_ttm) are updated at each quarterly filing.
    # Between filings the value is flat (as_of backward-fills).
    # We treat each day as a fraction of a quarter (1/63 of a quarter ≈ 1/252 yr).
    #
    # Per-day survival: (1 - annual_depr)^(1/252)
    # Per-day addition: we credit rnd_expense_ttm/252 each trading day
    #   (smooth approximation; equals the quarterly addition over ~63 days).
    #
    # This is more faithful to daily data than detecting quarter boundaries.
    # -----------------------------------------------------------------------
    daily_survival = (1.0 - _ANNUAL_DEPR) ** (1.0 / 252.0)
    rdc = np.empty(n, dtype=float)
    rdc[:] = np.nan

    # Find first non-NaN rnd observation to seed RDC
    first_valid = -1
    for i in range(n):
        if not np.isnan(rnd_ttm[i]) and rnd_ttm[i] >= 0:
            first_valid = i
            break

    if first_valid == -1:
        # No R&D data at all
        df["ext2_knowledge_cap_to_market_yield"] = np.nan
        df["ext2_knowledge_cap_to_market_asset_ratio"] = np.nan
        df["ext2_knowledge_cap_to_market_yield_chg252"] = np.nan
        # Drop scratch fund_ columns
        for c in ["fund_rnd_expense_ttm", "fund_shares_outstanding", "fund_assets"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        return df

    # Seed: start with a rough steady-state estimate
    # At steady state: RDC = (daily_add) / (1 - daily_survival)
    # where daily_add = rnd_ttm / 252
    seed_rnd = rnd_ttm[first_valid]
    daily_add_seed = seed_rnd / 252.0
    rdc_ss = daily_add_seed / max(1.0 - daily_survival, 1e-12)
    rdc[first_valid] = rdc_ss

    for i in range(first_valid + 1, n):
        if np.isnan(rnd_ttm[i]):
            # Depreciate only
            rdc[i] = rdc[i - 1] * daily_survival
        else:
            daily_add = rnd_ttm[i] / 252.0
            rdc[i] = rdc[i - 1] * daily_survival + daily_add

    # -----------------------------------------------------------------------
    # Normalise
    # -----------------------------------------------------------------------
    market_cap = close * shares  # element-wise
    market_cap = np.where(market_cap <= 0, np.nan, market_cap)
    assets_safe = np.where((assets <= 0) | np.isnan(assets), np.nan, assets)

    yield_col = np.where(np.isnan(rdc) | np.isnan(market_cap), np.nan, rdc / market_cap)
    asset_ratio = np.where(np.isnan(rdc) | np.isnan(assets_safe), np.nan, rdc / assets_safe)

    # Guard inf
    yield_col = np.where(np.isinf(yield_col), np.nan, yield_col)
    asset_ratio = np.where(np.isinf(asset_ratio), np.nan, asset_ratio)

    df["ext2_knowledge_cap_to_market_yield"] = yield_col
    df["ext2_knowledge_cap_to_market_asset_ratio"] = asset_ratio

    # 252-day change in yield (momentum / growth of knowledge intensity vs mkt)
    yield_series = pd.Series(yield_col, index=df.index)
    df["ext2_knowledge_cap_to_market_yield_chg252"] = yield_series - yield_series.shift(252)

    # Drop scratch fund_ columns not in produces
    for c in ["fund_rnd_expense_ttm", "fund_shares_outstanding", "fund_assets"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df
