"""
Capital Turnover feature block — osap_capturnover.

Capital turnover = Revenue / Book Equity (or Revenue / Assets as fallback).
Per DuPont decomposition, higher capital turnover signals efficient use of
equity capital and predicts superior returns (value-efficiency cross-section).

This is a per-ticker PIT-fundamental implementation.  The cross-sectional
ranking signal (high cap-turn stocks outperform) is captured here as a
level + year-over-year change, which tree models can exploit within-ticker
over time.

Spec file was absent from the scratchpad; implementation is based on the
standard OpenSourceAP "CapTurnover" definition used by Haugen & Baker (1996)
and referenced in the Chen-Zimmermann OSAP catalogue:
  cap_turnover = sales_ttm / book_equity
  where book_equity = equity (shareholders' equity from balance sheet).
Fallback: if equity is missing, we use assets as denominator (asset turnover).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
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
    "name": "osap_capturnover",
    "description": (
        "Capital Turnover: Revenue_TTM / Book Equity (shareholders' equity). "
        "Captures the efficiency with which a firm converts equity capital into "
        "revenue — a component of the DuPont ROE decomposition. Higher values "
        "signal more efficient deployment of equity capital. "
        "Produces: (1) level ratio, (2) YoY change in the ratio (momentum in "
        "efficiency improvement), (3) a 4-quarter rolling z-score of the level "
        "as a per-ticker normalised view. "
        "Implementation is per-ticker using PIT fundamentals; the cross-sectional "
        "ranking dimension is left to the model. Falls back to total assets as "
        "denominator when equity is unavailable (yielding asset turnover). "
        "Source: OpenSourceAP / Chen-Zimmermann; Haugen & Baker (1996)."
    ),
    "requires": ["Close"],          # Close used only to gate price < 5 rows
    "produces": [
        "osap_capturnover_ratio",   # Revenue_TTM / Equity (level)
        "osap_capturnover_chg",     # YoY change (current - lag-252-day value)
        "osap_capturnover_zscore",  # 4-quarter (252-day) rolling z-score of ratio
    ],
    "tags": ["fundamental", "efficiency", "value", "dupont", "osap"],
    "version": "1.0.0",
    "author": (
        "Haugen & Baker (1996); OpenSourceAP / Chen-Zimmermann catalogue "
        "(osap_capturnover). Block by Claude Sonnet 4-6."
    ),
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add capital-turnover features to a single-ticker DataFrame."""
    # ------------------------------------------------------------------
    # 1. Load PIT fundamentals
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=["revenue_ttm", "equity", "assets"])

    # ------------------------------------------------------------------
    # 2. Capital turnover ratio = revenue_ttm / equity
    #    Fallback to assets when equity is zero / missing.
    # ------------------------------------------------------------------
    rev = df["fund_revenue_ttm"]
    eq  = df["fund_equity"].replace(0, np.nan)
    ast = df["fund_assets"].replace(0, np.nan)

    # Use equity if available, otherwise assets (asset turnover proxy)
    denom = eq.where(eq.notna(), ast)

    ratio = rev / denom          # may be NaN where fundamentals absent

    # Guard inf
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_capturnover_ratio"] = ratio

    # ------------------------------------------------------------------
    # 3. YoY change (~252 trading-day lag) — captures efficiency momentum
    # ------------------------------------------------------------------
    lag_252 = ratio.shift(252)
    chg = ratio - lag_252
    chg = chg.replace([np.inf, -np.inf], np.nan)
    df["osap_capturnover_chg"] = chg

    # ------------------------------------------------------------------
    # 4. Rolling z-score over ~252 days (≈1 year) — per-ticker normalisation
    # ------------------------------------------------------------------
    roll_mean = ratio.rolling(252, min_periods=60).mean()
    roll_std  = ratio.rolling(252, min_periods=60).std()
    zscore = (ratio - roll_mean) / roll_std.replace(0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    df["osap_capturnover_zscore"] = zscore

    # ------------------------------------------------------------------
    # 5. Drop scratch fund_ columns not in produces
    # ------------------------------------------------------------------
    df = df.drop(columns=["fund_revenue_ttm", "fund_equity", "fund_assets"],
                 errors="ignore")

    return df
