"""
_paper_doaj_8e4deb03_cashflow_coverage.py  --  CANDIDATE feature block (leading _ = unproven).

Source: "Integrative Analysis of Traditional and Cash Flow Financial Ratios: Insights from a
Systematic Comparative Review" (DOAJ id 8e4deb03). The paper finds cash-flow-based ratios
often dominate traditional accrual metrics in predicting financial performance. This block
implements six per-ticker cash-flow coverage / quality ratios using SEC point-in-time data.

The fund_* scratch columns are dropped before returning; only METADATA['produces'] columns
are added to df.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Load the shared PIT helper by file path (hidden from auto-discovery by its leading underscore).
_spec = _ilu.spec_from_file_location(
    "_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name": "cashflow_coverage",
    "description": (
        "Six point-in-time cash-flow coverage / quality ratios from SEC filings: "
        "OCF-to-sales, OCF-to-debt, FCF-to-assets, CapEx intensity, cash conversion, FCF margin."
    ),
    "requires": ["Close"],
    "produces": [
        "cfc_ocf_to_sales",
        "cfc_ocf_to_debt",
        "cfc_fcf_to_assets",
        "cfc_capex_intensity",
        "cfc_cash_conversion",
        "cfc_fcf_margin",
    ],
    "tags": ["fundamentals", "cashflow", "quality", "experimental"],
    "version": "1.0",
    "author": "DOAJ 8e4deb03 — cash-flow ratio systematic review",
}

# Fundamental fields this block requests from the PIT loader.
_FIELDS = [
    "operating_cash_flow_ttm",  # operating cash flow (trailing twelve months)
    "fcf_ttm",                  # free cash flow TTM
    "capex_ttm",                # capital expenditure TTM
    "revenue_ttm",              # revenue TTM
    "net_income_ttm",           # net income TTM
    "long_term_debt",           # long-term debt (best available total-debt proxy)
    "assets",                   # total assets
]


def _to_num(s: pd.Series) -> pd.Series:
    """Coerce to float; non-numeric -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide num by den; zero or negative denominator -> NaN (never inf)."""
    d = _to_num(den)
    n = _to_num(num)
    return n.where(d.abs() > 0).div(d.where(d.abs() > 0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Lookahead-safe backward merge: each row sees only filings with filed_date <= that row's Date.
    df = _fundamentals.as_of(df, fields=_FIELDS)

    ocf   = _to_num(df["fund_operating_cash_flow_ttm"])
    fcf   = _to_num(df["fund_fcf_ttm"])
    capex = _to_num(df["fund_capex_ttm"]).abs()  # capex is often reported negative; use magnitude
    rev   = _to_num(df["fund_revenue_ttm"])
    ni    = _to_num(df["fund_net_income_ttm"])
    debt  = _to_num(df["fund_long_term_debt"])
    assets = _to_num(df["fund_assets"])

    # cfc_ocf_to_sales: operating cash generation per unit of revenue (positive revenue required)
    df["cfc_ocf_to_sales"] = _safe_div(ocf, rev.where(rev > 0))

    # cfc_ocf_to_debt: how well OCF covers total debt (positive debt required)
    df["cfc_ocf_to_debt"] = _safe_div(ocf, debt.where(debt > 0))

    # cfc_fcf_to_assets: free cash yield on total assets (positive assets required)
    df["cfc_fcf_to_assets"] = _safe_div(fcf, assets.where(assets > 0))

    # cfc_capex_intensity: reinvestment ratio (|capex| / |ocf|); negative/zero OCF -> NaN
    df["cfc_capex_intensity"] = _safe_div(capex, ocf.abs().where(ocf.abs() > 0))

    # cfc_cash_conversion: OCF vs accrual earnings quality (any non-zero net income)
    df["cfc_cash_conversion"] = _safe_div(ocf, ni.where(ni.abs() > 0))

    # cfc_fcf_margin: free cash flow as fraction of revenue (positive revenue required)
    df["cfc_fcf_margin"] = _safe_div(fcf, rev.where(rev > 0))

    # Drop all scratch fund_* columns so only METADATA['produces'] columns are added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
