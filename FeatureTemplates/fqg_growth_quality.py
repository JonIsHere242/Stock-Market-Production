"""
fqg_growth_quality.py  --  Point-in-time fundamental GROWTH & EARNINGS-QUALITY dynamics.

Theme: how a firm's *own* fundamentals are changing through time, with an emphasis on the
quality of that growth (cash vs accrual earnings, margin direction, asset efficiency).

All inputs come through _fundamentals.as_of() -- a BACKWARD merge_asof on `filed_date`, so every
trading-day row only sees filings that were already public by that date (lookahead-safe / PIT).

DISTINCTNESS NOTE
-----------------
The hidden candidate _paper_oalex_W3039879956_fundamental_momentum.py already covers YoY growth of
revenue / EPS / net_income and the TREND of net_margin and ROE. To stay orthogonal this block
deliberately AVOIDS those exact quantities and instead measures:
  - growth of the CASH-FLOW and GROSS-PROFIT lines (operating_cash_flow_ttm, fcf_ttm, gross_profit_ttm)
  - ASSET growth (balance-sheet expansion -- the Cooper-Gulen-Schill asset-growth anomaly)
  - SCALED accruals (net_income - operating_cash_flow) / assets  (Sloan accrual anomaly; the
    hidden Piotroski block only emits the raw CFO-NI difference, not the assets-scaled ratio)
  - direction of GROSS and OPERATING margin (not net margin) and ROA momentum (not ROE)

These are finance-grounded anomalies (Sloan 1996 accruals; Cooper-Gulen-Schill 2008 asset growth;
Novy-Marx 2013 gross profitability) that are economically distinct from plain price momentum.

Windowing: fundamentals are quarterly, so we compare the current as-of value to the value ~252
trading rows earlier (~1 calendar year) to form a year-over-year change that spans 4 quarters.
The first ~252 rows are NaN by construction (insufficient history) -- expected and fine.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Load the hidden _fundamentals helper by file path (identical idiom to _fundamentals_valuation.py).
_spec = _ilu.spec_from_file_location(
    "_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "fqg_growth_quality",
    "description": (
        "Point-in-time fundamental growth & earnings-quality dynamics: YoY growth of cash flow / "
        "FCF / gross profit, asset growth, assets-scaled accruals (Sloan), gross & operating "
        "margin trend, and ROA momentum -- from filed-date SEC fundamentals."
    ),
    "requires":    ["Ticker", "Date"],
    "produces":    [
        "fqg_ocf_growth_yoy",       # YoY growth of operating cash flow (TTM)
        "fqg_fcf_growth_yoy",       # YoY growth of free cash flow (TTM)
        "fqg_gross_profit_growth_yoy",  # YoY growth of gross profit (TTM) -- Novy-Marx line
        "fqg_asset_growth_yoy",     # YoY growth of total assets (Cooper-Gulen-Schill anomaly)
        "fqg_accruals_to_assets",   # (net_income_ttm - ocf_ttm) / assets (Sloan accrual ratio)
        "fqg_gross_margin_trend",   # gross_margin now minus ~1yr ago (level change)
        "fqg_operating_margin_trend",  # operating_margin now minus ~1yr ago
        "fqg_roa_momentum",         # roa now minus ~1yr ago (asset-profitability momentum)
    ],
    "tags":        ["fundamentals", "growth", "quality", "experimental"],
    "version":     "1.0",
    "author":      "feature-gen",
}

_LAG = 252  # ~1 trading year

_FIELDS = [
    "operating_cash_flow_ttm",
    "fcf_ttm",
    "gross_profit_ttm",
    "net_income_ttm",
    "assets",
    "gross_margin",
    "operating_margin",
    "roa",
]


def _num(df: pd.DataFrame, field: str) -> pd.Series:
    """Pull a fund_<field> column as float, robust to absence."""
    return pd.to_numeric(
        df.get(f"fund_{field}", pd.Series(np.nan, index=df.index)), errors="coerce"
    )


def _yoy_growth(series: pd.Series, lag: int = _LAG) -> pd.Series:
    """
    Year-over-year growth ratio (current/past - 1), defined only when the base is strictly
    positive (a growth rate off a zero/negative base is meaningless). NaN otherwise, never inf.
    """
    past = series.shift(lag)
    base = past.where(past > 0)
    out = series / base - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # PIT merge: each row sees only fundamentals filed on/before its Date.
    df = _fundamentals.as_of(df, fields=_FIELDS)

    ocf    = _num(df, "operating_cash_flow_ttm")
    fcf    = _num(df, "fcf_ttm")
    gp     = _num(df, "gross_profit_ttm")
    ni     = _num(df, "net_income_ttm")
    assets = _num(df, "assets")
    gm     = _num(df, "gross_margin")
    om     = _num(df, "operating_margin")
    roa    = _num(df, "roa")

    # ---- Cash-flow & gross-profit growth (orthogonal to the revenue/EPS/NI growth elsewhere) --
    df["fqg_ocf_growth_yoy"]          = _yoy_growth(ocf)
    df["fqg_fcf_growth_yoy"]          = _yoy_growth(fcf)
    df["fqg_gross_profit_growth_yoy"] = _yoy_growth(gp)

    # ---- Asset growth (Cooper-Gulen-Schill: high asset growth -> low future returns) ----------
    df["fqg_asset_growth_yoy"] = _yoy_growth(assets)

    # ---- Sloan accruals scaled by assets: (NI - OCF)/assets. High accruals -> low quality. ----
    assets_pos = assets.where(assets > 0)
    accruals = (ni - ocf) / assets_pos
    df["fqg_accruals_to_assets"] = accruals.replace([np.inf, -np.inf], np.nan).clip(-2.0, 2.0)

    # ---- Margin TREND (level change vs ~1yr ago) -- gross & operating (NOT net, handled elsewhere)
    df["fqg_gross_margin_trend"]     = (gm - gm.shift(_LAG)).clip(-2.0, 2.0)
    df["fqg_operating_margin_trend"] = (om - om.shift(_LAG)).clip(-2.0, 2.0)

    # ---- ROA momentum (asset-profitability direction; ROE momentum is handled elsewhere) ------
    df["fqg_roa_momentum"] = (roa - roa.shift(_LAG)).clip(-2.0, 2.0)

    # Drop scratch fundamentals columns so only METADATA["produces"] is added.
    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
