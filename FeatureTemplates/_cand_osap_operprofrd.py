"""
Operating Profitability R&D Adjusted (osap_operprofrd)
Source: OpenSourceAP (Chen-Zimmermann), Ball et al. 2016

Signal: (Revenue - COGS - (SG&A - R&D)) / Total Assets
Predicted sign: +1 (long high profitability)

Economic intuition: R&D-adjusted operating profitability treats R&D spending
as an investment rather than a pure expense. By adding back R&D from the SG&A
deduction, high R&D spenders get a "credit", making the metric fairer across
growth vs. mature firms. Higher values predict better forward returns.

Per-ticker proxy note: The cross-sectional sort is not replicable per-ticker;
we compute the level and a trailing change (slope) as the per-stock signal.
SIC-based financial exclusions and missing-mv screens cannot be applied here;
coverage gaps (~16% ETF/foreign) will appear as NaN rows.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load fundamentals helper (by file path, as required)
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
    "name": "osap_operprofrd",
    "description": (
        "Operating Profitability R&D Adjusted (Ball et al. 2016, via OpenSourceAP / "
        "Chen-Zimmermann). Numerator = Revenue - COGS - (SG&A - R&D); denominator = "
        "Total Assets. Missing numerator components are treated as 0 per the spec. "
        "Per-ticker proxy: we compute the level and a 4-quarter rolling change "
        "(momentum of the ratio). Cross-sectional sort and SIC/share-code "
        "exclusions are not applied at the per-ticker level. NaN where "
        "fundamentals coverage is missing (~16% ETFs/foreign)."
    ),
    "requires": ["Close"],  # needs Close only as a date anchor; fundamentals via helper
    "produces": [
        "osap_operprofrd_level",   # (Rev - COGS - SGA + RND) / Assets, PIT
        "osap_operprofrd_chg",     # 4-quarter change in level (momentum of ratio)
        "osap_operprofrd_norm",    # 2-year rolling z-score of level (per-ticker norm)
    ],
    "tags": ["profitability", "fundamentals", "accounting", "ball2016", "opensourceap"],
    "version": "1.0",
    "author": "Ball et al. 2016; OpenSourceAP (Chen-Zimmermann); block by Claude",
}

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull needed PIT fundamental fields
    # revenue_ttm  ~ revt (annual revenue)
    # cost_of_revenue_ttm ~ cogs
    # rnd_expense_ttm  ~ xrd (R&D)
    # assets  ~ at (total assets, balance-sheet)
    # operating_income_ttm ~ we won't use this; we need gross_profit and xsga proxy
    #
    # SG&A is not directly available; we approximate:
    #   SG&A = Revenue - COGS - Operating_Income   (income-statement identity)
    # Then: Numerator = Rev - COGS - (SGA - RND)
    #                 = Rev - COGS - SGA + RND
    #                 = Operating_Income + RND     (since OI = Rev - COGS - SGA)
    #
    # So the metric simplifies to: (Operating_Income_TTM + RND_TTM) / Assets
    # Missing fields replaced with 0 per spec.

    needed = [
        "operating_income_ttm",
        "rnd_expense_ttm",
        "assets",
    ]
    df = _fundamentals.as_of(df, fields=needed)

    # Replace NaN with 0 for numerator components (spec: treat missing as 0)
    oi = df["fund_operating_income_ttm"].fillna(0.0)
    rnd = df["fund_rnd_expense_ttm"].fillna(0.0)
    assets = df["fund_assets"]  # keep NaN -- missing denom means undefined ratio

    # Level: (OI + R&D) / Total Assets
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.where(
            assets.notna() & (assets != 0),
            (oi + rnd) / assets,
            np.nan,
        )
    df["osap_operprofrd_level"] = level

    # 4-quarter change: difference between current and ~252-trading-day-ago value
    # Using a 252-bar shift as a proxy for ~1 year lag (annual filing cadence)
    level_series = df["osap_operprofrd_level"]
    df["osap_operprofrd_chg"] = level_series - level_series.shift(252)

    # 2-year rolling z-score of the level (504 trading days)
    roll_mean = level_series.rolling(window=504, min_periods=63).mean()
    roll_std = level_series.rolling(window=504, min_periods=63).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(
            roll_std.notna() & (roll_std > 0),
            (level_series - roll_mean) / roll_std,
            np.nan,
        )
    df["osap_operprofrd_norm"] = norm

    # Drop scratch fund_ columns we are NOT listing in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=fund_cols, inplace=True)

    return df
