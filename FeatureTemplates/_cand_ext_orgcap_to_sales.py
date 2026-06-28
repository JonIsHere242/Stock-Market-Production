"""
Organizational Capital / Sales and Productivity Ratio
------------------------------------------------------
Recomputes organizational capital (capitalized SG&A via perpetual-inventory
method, 30 %/yr depreciation) but normalizes by revenue_ttm instead of assets,
and also produces the inverse efficiency ratio (revenue per unit of org capital)
plus its YoY trend.  Orthogonal to osap_orgcap (which scales by assets).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
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
    "name": "ext_orgcap_to_sales",
    "description": (
        "Organizational capital (capitalized SG&A, perpetual-inventory 30 %/yr "
        "decay) scaled by trailing-twelve-month revenue, plus the inverse "
        "organizational-capital productivity ratio (revenue_ttm / OC) and its "
        "YoY change.  Uses PIT _fundamentals as_of to avoid lookahead.  "
        "Orthogonal to osap_orgcap (which scales OC by assets rather than "
        "revenue).  Proxy: OC reconstructed from rnd_expense_ttm as SG&A proxy "
        "when dedicated SG&A field is unavailable; coverage ~84 % (ETFs/foreign "
        "tickers return NaN rows which are expected)."
    ),
    "requires": [],          # pulls everything from PIT fundamentals
    "produces": [
        "ext_orgcap_to_sales_ratio",   # OC / revenue_ttm  (level)
        "ext_orgcap_to_sales_prod",    # revenue_ttm / OC  (efficiency level)
        "ext_orgcap_to_sales_prod_yoy",# YoY change in productivity
    ],
    "tags": ["fundamentals", "organizational_capital", "efficiency", "sga", "revenue"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner osap_orgcap; "
        "method based on Eisfeldt & Papanikolaou (2013) 'Organization Capital "
        "and the Cross-Section of Expected Returns', JF."
    ),
}

# Perpetual-inventory depreciation rate (matches standard orgcap literature)
_DELTA = 0.30
# Initial capital seed fraction (first observation = SGA / (delta + g), g=0.10)
_G = 0.10


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext_orgcap_to_sales_* columns to df (one ticker, ascending dates)."""

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(
        df,
        fields=["revenue_ttm", "rnd_expense_ttm", "cost_of_revenue_ttm"],
    )

    rev   = df["fund_revenue_ttm"].to_numpy(dtype=float)
    # Use rnd_expense_ttm as the best available SG&A-family proxy
    sga   = df["fund_rnd_expense_ttm"].to_numpy(dtype=float)

    n = len(df)

    # ------------------------------------------------------------------
    # 2. Perpetual-inventory org capital: OC_t = (1-delta)*OC_{t-1} + SGA_t
    #    Seed: OC_0 = SGA_0 / (delta + g) when first non-NaN SGA appears.
    # ------------------------------------------------------------------
    oc = np.full(n, np.nan, dtype=float)
    seeded = False
    for i in range(n):
        s = sga[i]
        if np.isnan(s):
            continue
        if not seeded:
            oc[i] = s / (_DELTA + _G)
            seeded = True
        else:
            # find last non-NaN OC
            prev_oc = np.nan
            for j in range(i - 1, -1, -1):
                if not np.isnan(oc[j]):
                    prev_oc = oc[j]
                    break
            if np.isnan(prev_oc):
                oc[i] = s / (_DELTA + _G)
            else:
                oc[i] = (1.0 - _DELTA) * prev_oc + s

    # ------------------------------------------------------------------
    # 3. Ratios
    # ------------------------------------------------------------------
    # OC / revenue_ttm
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (rev > 0) & ~np.isnan(oc) & ~np.isnan(rev),
            oc / rev,
            np.nan,
        )
        # revenue_ttm / OC  (productivity)
        prod = np.where(
            (oc > 0) & ~np.isnan(oc) & ~np.isnan(rev),
            rev / oc,
            np.nan,
        )

    # Replace inf (should not occur after np.where guards, but be safe)
    ratio = np.where(np.isinf(ratio), np.nan, ratio)
    prod  = np.where(np.isinf(prod),  np.nan, prod)

    # ------------------------------------------------------------------
    # 4. YoY change in productivity
    #    Fundamentals update quarterly; approximate "1 year ago" as 252 rows.
    #    Use pandas shift so vectorised; trailing NaN for early rows is expected.
    # ------------------------------------------------------------------
    prod_s = pd.Series(prod, index=df.index)
    prod_lag = prod_s.shift(252)
    with np.errstate(divide="ignore", invalid="ignore"):
        prod_yoy = np.where(
            ~np.isnan(prod_s.to_numpy()) & ~np.isnan(prod_lag.to_numpy()),
            prod_s.to_numpy() - prod_lag.to_numpy(),
            np.nan,
        )

    # ------------------------------------------------------------------
    # 5. Assign produced columns; drop scratch fund_* columns
    # ------------------------------------------------------------------
    df["ext_orgcap_to_sales_ratio"]    = ratio
    df["ext_orgcap_to_sales_prod"]     = prod
    df["ext_orgcap_to_sales_prod_yoy"] = prod_yoy

    # Drop fund_* scratch columns not listed in produces
    for col in ["fund_revenue_ttm", "fund_rnd_expense_ttm", "fund_cost_of_revenue_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
