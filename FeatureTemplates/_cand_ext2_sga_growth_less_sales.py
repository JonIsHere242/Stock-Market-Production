"""
Organizational investment outpacing revenue (ext2_sga_growth_less_sales).

YoY growth of SG&A (proxy: gross_profit_ttm - operating_income_ttm) minus
YoY growth of revenue_ttm. A positive value means org spending is growing
faster than sales — interpreted as building organizational capital ahead of
future revenue (the "excess SG&A investment" signal from the OSAP/OrgCap
literature).

Also produces a variant adding R&D to the numerator:
  (SG&A + RnD) YoY growth - revenue_ttm YoY growth.

All fundamentals are sourced via PIT _fundamentals.as_of (backward merge on
filed_date), so there is no lookahead.  YoY lags use 252 trading-day offsets
on the already-merged columns, which are themselves point-in-time safe.

Per-ticker proxy; no cross-sectional ranking.
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
    "name": "ext2_sga_growth_less_sales",
    "description": (
        "YoY growth of SG&A (gross_profit_ttm - operating_income_ttm) minus "
        "YoY growth of revenue_ttm, and same metric including RnD. "
        "Positive = org spend growing faster than revenue (organizational "
        "capital investment ahead of sales). PIT-safe via _fundamentals.as_of. "
        "Per-ticker proxy for the cross-sectional OrgCap signal. "
        "Author note: SG&A is derived as gross_profit_ttm - operating_income_ttm "
        "because direct SG&A is not in the available field list."
    ),
    "requires": [],
    "produces": [
        "ext2_sga_growth_less_sales_sga",   # SG&A YoY growth minus revenue YoY growth
        "ext2_sga_growth_less_sales_sgard",  # (SG&A + RnD) YoY growth minus revenue YoY growth
        "ext2_sga_growth_less_sales_sign",   # sign-persistent 4q smoothed version of _sga
    ],
    "tags": ["fundamentals", "organizational_capital", "sga", "growth", "quality"],
    "version": "1.0.0",
    "author": (
        "Round-3 deep exploration of rich winner vein (osap_orgcap); "
        "extends OSAP/OrgCap literature (Peters & Taylor, 2017)."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LAG = 252  # ~1 trading year


def _safe_yoy_growth(series: pd.Series, lag: int = _LAG) -> pd.Series:
    """YoY growth = (x_t - x_{t-lag}) / |x_{t-lag}|.  Returns NaN when
    denominator is zero or either value is NaN."""
    prev = series.shift(lag)
    denom = prev.abs()
    denom = denom.where(denom > 1e-12, other=np.nan)
    return (series - prev) / denom


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Fetch PIT fundamentals we need
    df = _fundamentals.as_of(
        df,
        fields=[
            "gross_profit_ttm",
            "operating_income_ttm",
            "revenue_ttm",
            "rnd_expense_ttm",
        ],
    )

    gp   = df["fund_gross_profit_ttm"]
    oi   = df["fund_operating_income_ttm"]
    rev  = df["fund_revenue_ttm"]
    rnd  = df["fund_rnd_expense_ttm"]

    # SG&A proxy = gross_profit - operating_income  (both TTM)
    # This works because: gross_profit - COGS = gross_profit,
    # and operating_income = gross_profit - SG&A - RnD - D&A (approx).
    # A more parsimonious proxy: sga_proxy = gross_profit_ttm - operating_income_ttm
    sga_proxy = gp - oi  # may be negative if oi > gp (unusual)

    # SG&A + RnD (fill RnD NaN with 0 so non-R&D companies still get a signal)
    rnd_filled = rnd.fillna(0.0)
    sgard_proxy = sga_proxy + rnd_filled

    # YoY growth series (PIT-safe: we lag the already-PIT-merged column)
    sga_g  = _safe_yoy_growth(sga_proxy)
    rev_g  = _safe_yoy_growth(rev)
    sgard_g = _safe_yoy_growth(sgard_proxy)

    # Core signal: excess org investment relative to sales growth
    df["ext2_sga_growth_less_sales_sga"]   = sga_g  - rev_g
    df["ext2_sga_growth_less_sales_sgard"] = sgard_g - rev_g

    # Smoothed / sign-persistent variant: rolling 63-day median of _sga
    # (quarterly smoothing reduces noise from lumpy filing updates)
    df["ext2_sga_growth_less_sales_sign"] = (
        df["ext2_sga_growth_less_sales_sga"]
        .rolling(63, min_periods=10)
        .median()
    )

    # Drop scratch fund_ columns
    df.drop(
        columns=[
            "fund_gross_profit_ttm",
            "fund_operating_income_ttm",
            "fund_revenue_ttm",
            "fund_rnd_expense_ttm",
        ],
        inplace=True,
    )

    return df
