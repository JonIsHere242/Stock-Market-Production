"""
Gross Profitability feature block.

Source: OpenSourceAP (Chen-Zimmermann); Novy-Marx (2013).
Definition: (Revenue - COGS) / Total Assets, i.e. gross profits scaled by assets.
Cross-sectional signal mapped faithfully per-ticker via PIT fundamentals.
Produces level, a YoY change (improvement trend), and an asset-scaled gross-margin proxy.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (loaded by file path per hard rules)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_grossprofitability",
    "description": (
        "Gross profitability (Novy-Marx 2013): (revenue_ttm - cost_of_revenue_ttm) / assets. "
        "Scaled gross profit measures operational efficiency with stronger pricing-power signal "
        "than net income. Implemented per-ticker via PIT SEC fundamentals (backward merge on "
        "filed_date, lookahead-safe). Also produces a YoY change in the ratio to capture "
        "trend (improving/deteriorating profitability) and the raw gross margin (GP/Revenue) "
        "as a complementary level. Coverage ~84%; ETFs/foreign get NaN. "
        "Inherently a cross-sectional factor (rank high = long); per-ticker level is a "
        "faithful proxy -- cross-sectional ranking done by the predictor at inference."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_grossprofitability_level",   # GP / total assets (the primary factor)
        "osap_grossprofitability_chg",     # YoY change in GP/Assets (improvement trend)
        "osap_grossprofitability_margin",  # Gross profit margin (GP / Revenue) companion
    ],
    "tags": ["profitability", "fundamentals", "quality", "novy_marx", "osap"],
    "version": "1.0",
    "author": "Novy-Marx (2013) 'The Other Side of Value: The Gross Profitability Premium', "
              "JFE; via OpenSourceAP (Chen-Zimmermann). Block by Claude.",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach osap_grossprofitability_* columns to df (one ticker, ascending by Date).

    Uses PIT fundamentals:
      gross_profit_ttm  = revenue_ttm - cost_of_revenue_ttm
      level             = gross_profit_ttm / assets
      chg               = level - level shifted ~252 trading days ago (YoY)
      margin            = gross_profit_ttm / revenue_ttm
    """
    # Pull the three fundamental fields we need (PIT, backward merge on filed_date)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=["revenue_ttm", "cost_of_revenue_ttm", "assets"],
        )

    rev = df["fund_revenue_ttm"]
    cogs = df["fund_cost_of_revenue_ttm"]
    assets = df["fund_assets"]

    # Gross profit (TTM)
    gp = rev - cogs

    # Primary factor: GP / Total Assets
    assets_safe = assets.replace(0, np.nan)
    gp_over_assets = gp / assets_safe
    # clip extreme outliers (e.g. micro-cap with near-zero assets one quarter)
    gp_over_assets = gp_over_assets.clip(-10, 10)

    df["osap_grossprofitability_level"] = gp_over_assets

    # YoY change: compare to the value ~252 trading days ago (1 year lag)
    # We forward-fill the level first so a stale filing doesn't create spurious jumps,
    # then diff by 252 steps.  This stays lookahead-free because all values are PIT.
    level_ffill = gp_over_assets.ffill()
    df["osap_grossprofitability_chg"] = level_ffill - level_ffill.shift(252)

    # Companion: gross margin (scale-independent, useful cross-sectionally)
    rev_safe = rev.replace(0, np.nan)
    gross_margin = (gp / rev_safe).clip(-5, 5)
    df["osap_grossprofitability_margin"] = gross_margin

    # Drop scratch fund_* columns not listed in produces
    for col in ["fund_revenue_ttm", "fund_cost_of_revenue_ttm", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
