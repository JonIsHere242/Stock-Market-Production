"""
Candidate feature block: ext2_return_on_intangible
Implements Return on Intangible Capital via perpetual-inventory capitalization of SG&A and R&D.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
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
    "name": "ext2_return_on_intangible",
    "description": (
        "Return on intangible capital: capitalizes SG&A (org capital, OC) at 30%/yr "
        "and R&D (RD capital, RDC) at 20%/yr via perpetual-inventory method. "
        "Produces: (1) ext2_return_on_intangible_roi = operating_income_ttm / (OC+RDC) "
        "-- intangible efficiency; (2) ext2_return_on_intangible_turnover = "
        "revenue_ttm / (OC+RDC) -- intangible asset turnover; "
        "(3) ext2_return_on_intangible_roi_trend = 252-day OLS slope of roi. "
        "Per-ticker proxy; all fundamentals are PIT via _fundamentals.as_of. "
        "Extends parent feature osap_orgcap (org-capital perpetual-inventory). "
        "~84% coverage; ETFs and foreign stocks yield NaN."
    ),
    "requires": [],
    "produces": [
        "ext2_return_on_intangible_roi",
        "ext2_return_on_intangible_turnover",
        "ext2_return_on_intangible_roi_trend",
    ],
    "tags": ["fundamentals", "intangible", "organizational_capital", "rnd_capital", "efficiency"],
    "version": "1.0.0",
    "author": "Round-3 deep exploration of a rich winner vein (osap_orgcap); spec ext2_return_on_intangible",
}


# ---------------------------------------------------------------------------
# Helper: perpetual inventory on a series of quarterly additions
# ---------------------------------------------------------------------------
def _perpetual_inventory(additions: np.ndarray, delta: float) -> np.ndarray:
    """
    Recurrence: OC_t = (1 - delta) * OC_{t-1} + add_t
    Seed: OC_0 = add_0 / (0.025 + delta)  (Gordon-growth perpetuity)
    additions: 1-D float array in chronological order, NaN treated as 0.
    Returns same-length float array (OC values).
    """
    n = len(additions)
    result = np.empty(n, dtype=np.float64)
    carry = np.float64(0.0)
    seeded = False
    for i in range(n):
        add = additions[i]
        if np.isnan(add):
            add = 0.0
        if not seeded:
            if add > 0.0:
                carry = add / (0.025 + delta)
                seeded = True
            # else carry stays 0 until we have a positive add
        carry = (1.0 - delta) * carry + add
        result[i] = carry
    return result


# ---------------------------------------------------------------------------
# Helper: rolling OLS slope (no lookahead)
# ---------------------------------------------------------------------------
def _rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope of series ~ t using only the past `window` observations.
    Returns a Series of same index; leading values are NaN.
    """
    vals = series.to_numpy(dtype=np.float64)
    n = len(vals)
    slopes = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(window, dtype=np.float64)
    x -= x.mean()
    x_ss = (x * x).sum()
    if x_ss == 0:
        return pd.Series(slopes, index=series.index)
    for i in range(window - 1, n):
        y = vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        slopes[i] = ((x * (y - y_mean)).sum()) / x_ss
    return pd.Series(slopes, index=series.index)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals; all fund_* columns added in-place (backward-safe)
    fields = [
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
        "revenue_ttm",
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # ---- Derive quarterly SG&A estimate ----
    # SGA_quarterly = (gross_profit_ttm - operating_income_ttm - rnd_expense_ttm) / 4
    # Clipped to >= 0 so capitalization is non-negative.
    gp   = df["fund_gross_profit_ttm"].to_numpy(dtype=np.float64)
    oi   = df["fund_operating_income_ttm"].to_numpy(dtype=np.float64)
    rnd  = df["fund_rnd_expense_ttm"].to_numpy(dtype=np.float64)
    rev  = df["fund_revenue_ttm"].to_numpy(dtype=np.float64)

    # rnd_expense_ttm may be reported as positive or as absolute; treat as positive cost.
    rnd_pos = np.where(np.isnan(rnd), np.nan, np.abs(rnd))

    sga_ttm = gp - oi - rnd_pos           # annual SG&A proxy
    sga_ttm = np.where(sga_ttm < 0, np.nan, sga_ttm)   # only capitalise when positive
    sga_q   = np.where(np.isnan(sga_ttm), np.nan, sga_ttm / 4.0)

    rnd_q   = np.where(np.isnan(rnd_pos), np.nan, rnd_pos / 4.0)

    # ---- Perpetual inventory ----
    # OC: org-capital, delta = 1 - 0.7^0.25 ≈ 0.08158
    delta_oc  = 1.0 - (0.7 ** 0.25)
    # RDC: R&D capital, delta = 1 - 0.8^0.25 ≈ 0.05132
    delta_rdc = 1.0 - (0.8 ** 0.25)

    oc  = _perpetual_inventory(sga_q, delta_oc)
    rdc = _perpetual_inventory(rnd_q, delta_rdc)

    total_intangible = oc + rdc

    # Guard: zero denominator -> NaN
    denom = np.where(total_intangible <= 0, np.nan, total_intangible)

    roi      = np.where(np.isnan(oi),  np.nan, oi  / denom)
    turnover = np.where(np.isnan(rev), np.nan, rev / denom)

    # Replace inf/-inf with NaN
    roi      = np.where(np.isinf(roi),      np.nan, roi)
    turnover = np.where(np.isinf(turnover), np.nan, turnover)

    # ---- 252-day rolling OLS trend of ROI ----
    roi_series = pd.Series(roi, index=df.index)
    roi_trend  = _rolling_ols_slope(roi_series, window=252)

    # Assign produced columns
    df["ext2_return_on_intangible_roi"]       = roi
    df["ext2_return_on_intangible_turnover"]  = turnover
    df["ext2_return_on_intangible_roi_trend"] = roi_trend.to_numpy()

    # Drop scratch fund_* columns not in produces
    for col in list(df.columns):
        if col.startswith("fund_"):
            df.drop(columns=[col], inplace=True)

    return df
