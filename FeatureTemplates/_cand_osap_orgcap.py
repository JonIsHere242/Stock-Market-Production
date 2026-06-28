"""
osap_orgcap — Organizational Capital (Eisfeldt & Papanikolaou 2013)

Organizational capital is the stock of intangible capital embedded in key
employees and firm-specific know-how, estimated by capitalizing SG&A
expenditures with a perpetual inventory model (30 % annual depreciation rate).

Per-ticker proxy: since we receive one stock at a time (ascending Date) with
PIT fundamentals available quarterly via _fundamentals.as_of(), we:
  1. Retrieve revenue_ttm, cost_of_revenue_ttm, operating_income_ttm, and
     rnd_expense_ttm (TTM = trailing twelve months, PIT-safe).
  2. Approximate SGA_ttm = gross_profit_ttm - operating_income_ttm
     (gross_profit - EBIT ≈ SG&A + D&A + other operating).
     When rnd_expense_ttm is available we subtract it to isolate SG&A more
     closely (R&D is separately capitalized in other blocks; here we want the
     selling/org portion).
  3. Build a rolling perpetual-inventory stock via a vectorised cumulative
     decay: OC_t = sum_{s<=t} SGA_s * (1-delta)^(t-s), which for quarterly
     data reduces to the recurrence OC_t = (1-delta)*OC_{t-1} + SGA_t.
     Because delta=0.3 annualised → 0.0724 per quarter, we use that rate.
  4. Scale by book assets (assets) to get OC/Assets — the ratio used in
     factor construction.
  5. Also emit OC growth (YoY change in OC/Assets) as a dynamic variant.

Cross-sectional note: the original factor SORTS stocks on OC/Assets within
industry; here we can only compute the level per ticker and its change. The
model uses these as features and handles any cross-sectional ranking.

Coverage: ~84 % (ETFs/foreign may have no fundamentals). Leading and missing
rows are NaN — expected.
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
    "name": "osap_orgcap",
    "description": (
        "Organizational capital (OC/Assets) estimated by capitalizing SG&A "
        "expenditures via perpetual inventory model (delta=30%/yr, quarterly). "
        "SGA approximated as gross_profit_ttm - operating_income_ttm (minus "
        "rnd_expense_ttm when available). Cross-sectional sort replaced by "
        "per-ticker level and YoY growth; predicted sign: +1 (high OC/Assets "
        "earns positive abnormal return per Eisfeldt & Papanikolaou 2013)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_orgcap_ratio",   # OC / book_assets (level)
        "osap_orgcap_growth",  # YoY change in OC/Assets
        "osap_orgcap_oc_ps",   # OC per share proxy (OC / shares_outstanding)
    ],
    "tags": ["fundamental", "intangible", "organizational_capital", "osap"],
    "version": "1.0",
    "author": "Eisfeldt & Papanikolaou 2013, JF; OpenSourceAP (Chen-Zimmermann)",
}

# Annual depreciation rate from E&P 2013; convert to quarterly
_DELTA_ANNUAL = 0.30
_DELTA_QTR = 1.0 - (1.0 - _DELTA_ANNUAL) ** (1.0 / 4.0)  # ≈ 0.0861


def _build_org_capital_series(sga_series: pd.Series, delta_qtr: float) -> pd.Series:
    """Vectorised perpetual inventory accumulation over a quarterly SGA series.

    OC_t = (1 - delta) * OC_{t-1} + SGA_t
    Seed: OC_0 = SGA_0 / (g + delta) where g is mean SGA growth; we approximate
    g = 0.10 / 4 per quarter (standard 10% annual growth assumption from E&P 2013).
    NaN SGA propagates the previous OC forward (no new investment assumed).
    """
    values = sga_series.values.astype(float)
    oc = np.full(len(values), np.nan)
    decay = 1.0 - delta_qtr
    g_qtr = 0.10 / 4.0  # 10 % annual growth → quarterly

    prev = np.nan
    for i in range(len(values)):
        v = values[i]
        if np.isnan(v):
            # No new info: carry forward decayed stock if we have one
            if not np.isnan(prev):
                prev = prev * decay
                oc[i] = prev
        else:
            if np.isnan(prev):
                # Seed the stock
                denom = g_qtr + delta_qtr
                prev = v / denom if denom != 0 else np.nan
            else:
                prev = decay * prev + v
            oc[i] = prev

    return pd.Series(oc, index=sga_series.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns with NaN
    df["osap_orgcap_ratio"] = np.nan
    df["osap_orgcap_growth"] = np.nan
    df["osap_orgcap_oc_ps"] = np.nan

    if df.empty:
        return df

    # Pull PIT fundamentals (all TTM to stay quarterly / avoid point-in-time leakage)
    needed = [
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
        "assets",
        "shares_outstanding",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=needed)

    # Alias fund_ columns
    gp = df.get("fund_gross_profit_ttm", pd.Series(np.nan, index=df.index))
    oi = df.get("fund_operating_income_ttm", pd.Series(np.nan, index=df.index))
    rnd = df.get("fund_rnd_expense_ttm", pd.Series(np.nan, index=df.index))
    assets = df.get("fund_assets", pd.Series(np.nan, index=df.index))
    shares = df.get("fund_shares_outstanding", pd.Series(np.nan, index=df.index))

    # Approximate quarterly SGA:
    # gross_profit_ttm - operating_income_ttm ≈ SGA + D&A (common approximation)
    # subtract R&D when available to isolate organisational spending
    sga_annual = (gp - oi).clip(lower=0)  # keep non-negative
    rnd_clean = rnd.fillna(0.0).clip(lower=0)
    sga_annual = (sga_annual - rnd_clean).clip(lower=0)

    # Convert TTM to quarterly equivalent for inventory recurrence
    sga_qtr = sga_annual / 4.0

    # Build OC via perpetual inventory
    oc_series = _build_org_capital_series(sga_qtr, _DELTA_QTR)

    # OC / Assets
    assets_clean = assets.where(assets > 0, other=np.nan)
    oc_ratio = oc_series / assets_clean

    # YoY growth in ratio: compare to value ~252 trading days ago
    lag_252 = oc_ratio.shift(252)
    oc_growth = (oc_ratio - lag_252) / lag_252.abs().where(lag_252 != 0, other=np.nan)

    # OC per share
    shares_clean = shares.where(shares > 0, other=np.nan)
    oc_per_share = oc_series / shares_clean

    # Assign outputs — replace inf with NaN
    df["osap_orgcap_ratio"] = oc_ratio.replace([np.inf, -np.inf], np.nan)
    df["osap_orgcap_growth"] = oc_growth.replace([np.inf, -np.inf], np.nan)
    df["osap_orgcap_oc_ps"] = oc_per_share.replace([np.inf, -np.inf], np.nan)

    # Drop scratch fund_ columns not in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df.drop(columns=fund_cols, inplace=True)

    return df
