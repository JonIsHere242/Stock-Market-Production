"""
fqg_capital_structure.py  --  Point-in-time CAPITAL-ALLOCATION & BALANCE-SHEET dynamics.

Theme: where a firm spends its money and how its balance sheet is changing -- investment
intensity (R&D, capex), shareholder distribution / dilution (share count), leverage and liquidity
direction, and cash-flow conversion. These are *intensity ratios* and *changes*, deliberately kept
distinct from the price-relative valuation multiples block (_fundamentals_valuation.py) and from
the growth/quality block (fqg_growth_quality.py).

All inputs come through _fundamentals.as_of() -- a BACKWARD merge_asof on `filed_date`, fully PIT.

FINANCE GROUNDING
-----------------
  - R&D intensity (rnd/revenue): innovation investment; high-R&D firms earn an intangible premium
    (Chan-Lakonishok-Sougiannis 2001; Eberhart-Maxwell-Siddique 2004).
  - Capex intensity (capex/revenue): the investment / "overinvestment" anomaly (Titman-Wei-Xie 2004).
  - Net share issuance / buyback (YoY share-count change): the net-issuance anomaly
    (Daniel-Titman 2006; Pontiff-Woodgate 2008) -- buyers of own stock outperform issuers.
  - Leverage & liquidity DIRECTION (debt_to_equity change, current_ratio change): financial-health
    dynamics distinct from the static F-score flags.
  - FCF margin & cash buffer: operating quality and balance-sheet flexibility.

Windowing: fundamentals are quarterly; YoY changes compare the current as-of value to the value
~252 trading rows earlier (4 quarters). First ~252 rows are NaN by construction.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location(
    "_fundamentals", _Path(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fundamentals)

METADATA = {
    "name":        "fqg_capital_structure",
    "description": (
        "Point-in-time capital-allocation & balance-sheet dynamics: R&D and capex intensity (and "
        "R&D trend), net share issuance / buyback, leverage & current-ratio change, FCF margin, "
        "and cash-to-assets -- from filed-date SEC fundamentals."
    ),
    "requires":    ["Ticker", "Date"],
    "produces":    [
        "fqg_rnd_intensity",        # rnd_expense_ttm / revenue_ttm
        "fqg_rnd_intensity_trend",  # change in R&D intensity vs ~1yr ago
        "fqg_capex_intensity",      # capex_ttm / revenue_ttm (magnitude of reinvestment)
        "fqg_net_issuance_yoy",     # YoY change in shares_outstanding (>0 = dilution, <0 = buyback)
        "fqg_debt_to_equity_chg",   # change in debt_to_equity vs ~1yr ago
        "fqg_current_ratio_chg",    # change in current_ratio vs ~1yr ago
        "fqg_fcf_margin",           # fcf_ttm / revenue_ttm
        "fqg_cash_to_assets",       # cash / assets (balance-sheet liquidity buffer)
    ],
    "tags":        ["fundamentals", "capital_allocation", "quality", "experimental"],
    "version":     "1.0",
    "author":      "feature-gen",
}

_LAG = 252  # ~1 trading year

_FIELDS = [
    "rnd_expense_ttm",
    "capex_ttm",
    "revenue_ttm",
    "shares_outstanding",
    "debt_to_equity",
    "current_ratio",
    "fcf_ttm",
    "cash",
    "assets",
]


def _num(df: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(
        df.get(f"fund_{field}", pd.Series(np.nan, index=df.index)), errors="coerce"
    )


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = _fundamentals.as_of(df, fields=_FIELDS)

    rnd     = _num(df, "rnd_expense_ttm")
    capex   = _num(df, "capex_ttm")
    revenue = _num(df, "revenue_ttm")
    shares  = _num(df, "shares_outstanding")
    d2e     = _num(df, "debt_to_equity")
    curr    = _num(df, "current_ratio")
    fcf     = _num(df, "fcf_ttm")
    cash    = _num(df, "cash")
    assets  = _num(df, "assets")

    rev_pos    = revenue.where(revenue > 0)
    assets_pos = assets.where(assets > 0)

    # ---- Investment intensity ---------------------------------------------------------------
    # R&D is often reported as a positive expense; intensity = fraction of sales reinvested.
    rnd_int = (rnd.abs() / rev_pos).replace([np.inf, -np.inf], np.nan).clip(0.0, 5.0)
    df["fqg_rnd_intensity"]       = rnd_int
    df["fqg_rnd_intensity_trend"] = (rnd_int - rnd_int.shift(_LAG)).clip(-5.0, 5.0)

    # Capex is typically reported negative (cash outflow); use magnitude over sales.
    df["fqg_capex_intensity"] = (capex.abs() / rev_pos).replace([np.inf, -np.inf], np.nan).clip(0.0, 5.0)

    # ---- Net share issuance / buyback (YoY % change in share count) -------------------------
    sh_past = shares.shift(_LAG)
    sh_base = sh_past.where(sh_past > 0)
    df["fqg_net_issuance_yoy"] = (shares / sh_base - 1.0).replace([np.inf, -np.inf], np.nan).clip(-2.0, 2.0)

    # ---- Leverage & liquidity DIRECTION -----------------------------------------------------
    df["fqg_debt_to_equity_chg"] = (d2e - d2e.shift(_LAG)).replace([np.inf, -np.inf], np.nan).clip(-10.0, 10.0)
    df["fqg_current_ratio_chg"]  = (curr - curr.shift(_LAG)).replace([np.inf, -np.inf], np.nan).clip(-10.0, 10.0)

    # ---- Cash-flow conversion & cash buffer -------------------------------------------------
    df["fqg_fcf_margin"]     = (fcf / rev_pos).replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)
    df["fqg_cash_to_assets"] = (cash / assets_pos).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)

    df = df.drop(columns=[c for c in df.columns if c.startswith("fund_")])
    return df
