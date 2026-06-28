"""
_paper_oalex_W3039879956_fundamental_momentum.py  --  CANDIDATE feature block.

Leading underscore == UNPROVEN candidate; the framework does NOT auto-discover it until promoted.

PAPER: OpenAlex W3039879956 -- "Forecasting Banks Return on Equity Using Leading Economic Indicators".
Abstract: forecasts ROE over short horizons using leading economic indicators; argues that
leading indicators improve short-term ROE forecasting.

HONEST PROXY: per-ticker FUNDAMENTAL MOMENTUM -- the trailing year-over-year growth and
acceleration of a firm's own fundamentals (revenue, EPS, net income, margins, ROE).
These are the firm-level leading signals analogous to what the paper captures at the macro level.

APPROACH:
  - Load point-in-time fundamentals via _fundamentals.as_of (filed_date backward merge).
  - Compute YoY growth as ratio of current as-of value vs its value 252 rows ago.
  - Compute acceleration as the change in the 126-row-ago growth rate.
  - Guard all divisions: zero/negative denominators -> NaN, never inf.
  - Drop all fund_* scratch columns; return only the fmom_* produced columns.

Promote (drop the leading underscore) only after marginal-contribution backtest gate.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Load the underscore helper by file path (hidden from framework discovery).
_spec = _ilu.spec_from_file_location(
    "_fundamentals",
    _Path(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "paper_oalex_W3039879956_fundamental_momentum",
    "description": (
        "Per-ticker fundamental momentum: YoY growth and acceleration of revenue, EPS, "
        "net income, net margin, and ROE from point-in-time SEC filings (filed-date as-of). "
        "Inspired by OpenAlex W3039879956 (ROE forecasting via leading indicators)."
    ),
    "requires":    ["Close"],
    "produces":    [
        "fmom_revenue_growth_252d",
        "fmom_eps_growth_252d",
        "fmom_net_income_growth_252d",
        "fmom_margin_trend_252d",
        "fmom_roe_trend_252d",
        "fmom_revenue_accel",
    ],
    "tags":        ["fundamentals", "momentum", "experimental"],
    "version":     "1.0",
    "author":      "paper oalex W3039879956 -- fundamental momentum proxy",
}

# Candidate fields to request from _fundamentals.as_of.
_FIELDS = [
    "revenue_ttm",
    "net_income_ttm",
    "eps_diluted_ttm",
    "net_margin",
    "roe",
]


def _ratio_growth(series: pd.Series, lag: int) -> pd.Series:
    """
    Year-over-year growth ratio: (current / past) - 1.
    Only defined when the base (past) value is strictly positive.
    Returns NaN where base <= 0 or either value is NaN.
    """
    current = pd.to_numeric(series, errors="coerce")
    past = current.shift(lag)
    base = past.where(past > 0)        # NaN for zero/negative denominators
    return current / base - 1.0


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Lookahead-safe PIT merge: each row sees only fundamentals filed on/before its Date.
    df = _fundamentals.as_of(df, fields=_FIELDS)

    # ---- 1. Revenue YoY growth (252 trading-day lag ~= 1 year) -------------------
    rev = pd.to_numeric(df.get("fund_revenue_ttm", pd.Series(dtype=float, index=df.index)),
                        errors="coerce")
    df["fmom_revenue_growth_252d"] = _ratio_growth(rev, 252)

    # ---- 2. EPS YoY growth -------------------------------------------------------
    eps = pd.to_numeric(df.get("fund_eps_diluted_ttm", pd.Series(dtype=float, index=df.index)),
                        errors="coerce")
    df["fmom_eps_growth_252d"] = _ratio_growth(eps, 252)

    # ---- 3. Net income YoY growth ------------------------------------------------
    ni = pd.to_numeric(df.get("fund_net_income_ttm", pd.Series(dtype=float, index=df.index)),
                       errors="coerce")
    df["fmom_net_income_growth_252d"] = _ratio_growth(ni, 252)

    # ---- 4. Net margin trend (level change, 252 rows) ----------------------------
    nm = pd.to_numeric(df.get("fund_net_margin", pd.Series(dtype=float, index=df.index)),
                       errors="coerce")
    df["fmom_margin_trend_252d"] = nm - nm.shift(252)

    # ---- 5. ROE trend (level change, 252 rows) -----------------------------------
    roe = pd.to_numeric(df.get("fund_roe", pd.Series(dtype=float, index=df.index)),
                        errors="coerce")
    df["fmom_roe_trend_252d"] = roe - roe.shift(252)

    # ---- 6. Revenue growth acceleration (growth now minus growth 126 rows ago) ---
    rev_g_now  = df["fmom_revenue_growth_252d"]
    rev_g_lag  = _ratio_growth(rev, 252).shift(126)
    df["fmom_revenue_accel"] = rev_g_now - rev_g_lag

    # ---- Drop all fund_* scratch columns ----------------------------------------
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])

    return df
