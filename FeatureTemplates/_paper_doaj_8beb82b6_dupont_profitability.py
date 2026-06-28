"""
_paper_doaj_8beb82b6_dupont_profitability.py  --  CANDIDATE feature block.

Leading underscore keeps this out of auto-discovery until promoted after backtest gate.

Based on DOAJ paper 8beb82b6: "The predictive ability of earnings-forecasting models".
The paper ranks predictors of future earnings by accuracy: accruals > sales > operating cash flows
> return on equity > operating income > non-operating income. HONEST PROXY: DuPont decomposition
of ROE = net margin x asset turnover x equity multiplier, capturing the three drivers the paper
identifies as most predictive (margin, asset efficiency, financial leverage).

Fundamentals are pulled point-in-time through _fundamentals.as_of (backward merge on filed_date,
fully lookahead-safe). fund_* scratch columns are dropped before returning.

Candidate fields requested from _fundamentals (with graceful fallback for absent fields):
  net_margin, gross_margin, operating_margin, revenue_ttm, assets, equity,
  net_income_ttm, asset_turnover (pre-computed), roe (pre-computed).
  Canonical names match the by-ticker parquet produced by build_fundamentals_panel.py.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Load the underscore helper by file path (hidden from discovery), same pattern as
# _fundamentals_valuation.py.
_spec = _ilu.spec_from_file_location(
    "_fundamentals",
    _Path(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name": "dupont_profitability",
    "description": (
        "DuPont decomposition of ROE into net margin, asset turnover, and equity multiplier "
        "from point-in-time SEC fundamentals (filed-date as-of), motivated by the earnings "
        "predictability ranking in DOAJ paper 8beb82b6."
    ),
    "requires": ["Close"],
    "produces": [
        "dup_net_margin",
        "dup_gross_margin",
        "dup_operating_margin",
        "dup_asset_turnover",
        "dup_equity_multiplier",
        "dup_roe_reconstructed",
    ],
    "tags": ["fundamentals", "profitability", "dupont", "experimental"],
    "version": "1.0",
    "author": "doaj 8beb82b6 — DuPont profitability proxy",
}

# All candidate fields we attempt to pull from the fundamentals store.
# Field names match the canonical store produced by build_fundamentals_panel.py:
#   assets (not total_assets), equity (not total_equity/stockholders_equity),
#   asset_turnover (pre-computed), roe (pre-computed).
# Some may not exist for all tickers; as_of returns NaN for absent fields.
_FIELDS = [
    "net_margin",
    "gross_margin",
    "operating_margin",
    "revenue_ttm",
    "assets",
    "equity",
    "net_income_ttm",
    "asset_turnover",
    "roe",
]


def _to_num(s: pd.Series) -> pd.Series:
    """Coerce to float; non-numeric -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series, require_positive_den: bool = False) -> pd.Series:
    """
    Element-wise division with division-by-zero and negative-denominator guards.

    Zero or negative denominators yield NaN when require_positive_den=True (e.g. equity
    multiplier where negative equity is economically meaningless); zero denominators always
    yield NaN. Never returns inf.
    """
    n = _to_num(num)
    d = _to_num(den)
    if require_positive_den:
        d = d.where(d > 0)
    else:
        d = d.where(d != 0)
    return n / d


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Point-in-time backward merge of fundamentals -------------------------
    df = _fundamentals.as_of(df, fields=_FIELDS)

    # --- 2. Resolve raw fundamental series (fund_ prefix added by as_of) ---------
    # Canonical field names in the fundamentals store: assets, equity, revenue_ttm,
    # net_margin, gross_margin, operating_margin, net_income_ttm, asset_turnover, roe.
    raw_net_margin       = _to_num(df.get("fund_net_margin"))
    raw_gross_margin     = _to_num(df.get("fund_gross_margin"))
    raw_operating_margin = _to_num(df.get("fund_operating_margin"))
    raw_revenue          = _to_num(df.get("fund_revenue_ttm"))
    raw_assets           = _to_num(df.get("fund_assets"))
    raw_net_income       = _to_num(df.get("fund_net_income_ttm"))
    raw_equity           = _to_num(df.get("fund_equity"))
    raw_asset_turnover_f = _to_num(df.get("fund_asset_turnover"))  # pre-computed if present
    raw_roe_f            = _to_num(df.get("fund_roe"))             # pre-computed if present

    # --- 3. dup_net_margin -------------------------------------------------------
    # Prefer the pre-computed ratio field; fall back to net_income_ttm / revenue_ttm.
    if raw_net_margin is not None and raw_net_margin.notna().any():
        dup_net_margin = raw_net_margin.copy()
    elif raw_net_income is not None and raw_revenue is not None:
        dup_net_margin = _safe_div(raw_net_income, raw_revenue)
    else:
        dup_net_margin = pd.Series(np.nan, index=df.index)

    # --- 4. dup_gross_margin -----------------------------------------------------
    if raw_gross_margin is not None:
        dup_gross_margin = raw_gross_margin.copy()
    else:
        dup_gross_margin = pd.Series(np.nan, index=df.index)

    # --- 5. dup_operating_margin -------------------------------------------------
    if raw_operating_margin is not None:
        dup_operating_margin = raw_operating_margin.copy()
    else:
        dup_operating_margin = pd.Series(np.nan, index=df.index)

    # --- 6. dup_asset_turnover = revenue_ttm / assets ----------------------------
    # Prefer the pre-computed store field; compute from parts if absent.
    # Assets must be positive (negative total_assets is a data error -> NaN).
    if raw_asset_turnover_f is not None and raw_asset_turnover_f.notna().any():
        dup_asset_turnover = raw_asset_turnover_f.copy()
    elif raw_revenue is not None and raw_assets is not None:
        dup_asset_turnover = _safe_div(raw_revenue, raw_assets, require_positive_den=True)
    else:
        dup_asset_turnover = pd.Series(np.nan, index=df.index)

    # --- 7. dup_equity_multiplier = assets / equity ------------------------------
    # Negative equity (distressed firm) -> NaN; equity == 0 -> NaN.
    if raw_assets is not None and raw_equity is not None:
        dup_equity_multiplier = _safe_div(raw_assets, raw_equity, require_positive_den=True)
    else:
        dup_equity_multiplier = pd.Series(np.nan, index=df.index)

    # --- 8. dup_roe_reconstructed = net_margin * asset_turnover * equity_multiplier
    # The pre-computed roe field from the store can serve as a cross-check reference,
    # but the reconstructed value is the DuPont signal (it decomposes roe into its
    # three drivers and so carries more information than the scalar roe itself).
    dup_roe_reconstructed = dup_net_margin * dup_asset_turnover * dup_equity_multiplier

    # --- 9. Assign produced columns to df ----------------------------------------
    df["dup_net_margin"]        = dup_net_margin
    df["dup_gross_margin"]      = dup_gross_margin
    df["dup_operating_margin"]  = dup_operating_margin
    df["dup_asset_turnover"]    = dup_asset_turnover
    df["dup_equity_multiplier"] = dup_equity_multiplier
    df["dup_roe_reconstructed"] = dup_roe_reconstructed

    # --- 10. Drop all fund_* scratch columns so only METADATA["produces"] remains
    drop_cols = [c for c in df.columns if c.startswith("fund_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df
