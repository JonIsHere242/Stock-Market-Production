"""
Net External Financing (osap_xfin)
Source: OpenSourceAP (Chen-Zimmermann); Bradshaw, Richardson, Sloan 2006.

Signal: Sum of external capital flows — equity issuance minus buybacks minus dividends
plus net long-term debt issuance — scaled by total assets.  Predicted sign: -1
(high external financing predicts lower future returns; firms raise capital before bad news).

Per-ticker proxy notes:
  - True computation needs sstk (sale of common stock), prstkc (purchase of common stock),
    dltis (LT debt issuance), dltr (LT debt reductions) — all income-statement / CF-statement
    line items not available in the PIT fundamentals store.
  - Proxy uses:
      * delta(long_term_debt)  ≈ net LT debt issuance (dltis - dltr)
      * delta(equity)          ≈ net equity issuance minus buybacks (sstk - prstkc);
        this also absorbs retained earnings, so it is noisier than sstk-prstkc.
      * dividends_paid_ttm     ≈ dv (directly available)
    All scaled by total assets (at).
  - A second column (osap_xfin_eq_chg) isolates the equity-change component only,
    which tends to carry the cleanest signal in the literature.
  - A third column (osap_xfin_debt_chg) isolates the debt-change component.
  - All changes are YoY using backward-looking rolling differences on filed-date-aligned
    fundamentals; no lookahead because as_of() uses backward merge_asof on filed_date.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_xfin",
    "description": (
        "Net External Financing (Bradshaw, Richardson, Sloan 2006 via OpenSourceAP/Chen-Zimmermann). "
        "Predicted sign -1: firms that raise more external capital tend to underperform. "
        "Per-ticker PIT proxy: uses delta(long_term_debt) as net LT debt issuance proxy, "
        "delta(equity) as net equity issuance/buyback proxy, and dividends_paid_ttm directly; "
        "all scaled by total assets.  Inherently annual-frequency accounting signal; "
        "values repeat until the next filing.  Cross-sectional ranking is needed for the "
        "original anomaly but per-ticker level is informative for time-series variation."
    ),
    "requires": [],  # fundamentals come from PIT helper, not OHLCV columns
    "produces": [
        "osap_xfin_total",    # full net-external-financing proxy / assets
        "osap_xfin_eq_chg",   # equity-change component / assets
        "osap_xfin_debt_chg", # debt-change component / assets
    ],
    "tags": ["fundamentals", "external_financing", "accounting", "osap", "bradshaw2006"],
    "version": "1.0.0",
    "author": "Bradshaw, Richardson, Sloan (2006); OpenSourceAP (Chen-Zimmermann); proxy by Claude Sonnet 4.6",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker net external financing proxy from PIT fundamentals.

    All three output columns are NaN-safe; leading NaNs when fundamentals
    are unavailable (ETFs, foreign listings, pre-filing periods) are expected.
    """
    # Attach PIT fundamentals (backward merge_asof on filed_date — no lookahead).
    df = _fundamentals.as_of(
        df,
        fields=["long_term_debt", "equity", "dividends_paid_ttm", "assets"],
    )

    # -----------------------------------------------------------------------
    # Extract raw fundamental columns (fund_* naming from as_of())
    # -----------------------------------------------------------------------
    ltd = df["fund_long_term_debt"].copy()       # long-term debt level
    eq  = df["fund_equity"].copy()               # book equity level
    div = df["fund_dividends_paid_ttm"].copy()   # dividends paid (TTM, typically negative)
    at  = df["fund_assets"].copy()               # total assets

    # Guard: assets must be positive and non-zero to use as denominator
    at_safe = at.where(at > 0, other=np.nan)

    # -----------------------------------------------------------------------
    # YoY changes — shift by 252 trading days as an annual-lag approximation.
    # Using a rolling shift avoids look-ahead because we only look backward.
    # For quarterly-filing cadence, 252 days ≈ 4 quarters back.
    # -----------------------------------------------------------------------
    LAG = 252  # ~1 calendar year of trading days

    delta_ltd = ltd - ltd.shift(LAG)   # ≈ dltis - dltr (net LT debt change)
    delta_eq  = eq  - eq.shift(LAG)    # ≈ sstk - prstkc (net equity change)

    # dividends_paid_ttm is typically reported as a negative number in CF statements;
    # we negate so that "more dividends paid" = larger positive subtraction term.
    # If the convention in the data is already a positive outflow, the sign is fine as-is.
    # We defensively take abs() so the subtraction direction is always correct.
    div_outflow = div.abs()

    # -----------------------------------------------------------------------
    # Net external financing components
    # Bradshaw (2006): XFIN = (sstk - prstkc) + (dltis - dltr) - dv
    # Proxy:           XFIN ≈ delta_eq + delta_ltd - div_outflow
    # -----------------------------------------------------------------------
    xfin_total    = (delta_eq + delta_ltd - div_outflow) / at_safe
    xfin_eq_chg   = delta_eq  / at_safe
    xfin_debt_chg = delta_ltd / at_safe

    # Guard: replace inf/-inf with NaN (should not arise given at_safe, but be safe)
    for series in (xfin_total, xfin_eq_chg, xfin_debt_chg):
        series.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["osap_xfin_total"]    = xfin_total
    df["osap_xfin_eq_chg"]   = xfin_eq_chg
    df["osap_xfin_debt_chg"] = xfin_debt_chg

    # Drop scratch fund_* columns we are NOT listing in produces
    df.drop(
        columns=["fund_long_term_debt", "fund_equity",
                 "fund_dividends_paid_ttm", "fund_assets"],
        errors="ignore",
    )

    return df
