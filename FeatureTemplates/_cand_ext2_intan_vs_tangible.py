"""
Feature block: ext2_intan_vs_tangible
Intangible-heaviness vs tangible assets (org-cap + R&D cap perpetual inventory).
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
    "name": "ext2_intan_vs_tangible",
    "description": (
        "Intangible-heaviness vs tangible assets. "
        "Builds two perpetual-inventory stocks: "
        "(1) Organizational Capital (OC) capitalising SG&A at 30%/yr depreciation; "
        "(2) R&D Capital (RDC) capitalising R&D at 20%/yr depreciation. "
        "Produces: share of productive base that is intangible "
        "((OC+RDC)/((OC+RDC)+ppe_net)), its 252-day linear trend, "
        "and the intangible-investment-to-capex ratio "
        "((ΔSGA+ΔRDC_investment) / capex_ttm). "
        "Per-ticker proxy; cross-sectional rank not required. "
        "Fund coverage ~84%; ETFs/foreign yield NaN."
    ),
    "requires": [],  # fundamentals pulled internally via PIT helper
    "produces": [
        "ext2_intan_vs_tangible_intang_share",
        "ext2_intan_vs_tangible_intang_share_trend",
        "ext2_intan_vs_tangible_intang_to_capex",
    ],
    "tags": ["fundamentals", "intangibles", "org_capital", "rnd_capital", "balance_sheet"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of rich winner vein (osap_orgcap); implemented as per spec.",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEPR_OC = 0.30          # organisational capital annual depreciation rate
_DEPR_RDC = 0.20         # R&D capital annual depreciation rate
# quarterly survival fractions
_SURV_OC = (1.0 - _DEPR_OC) ** 0.25
_SURV_RDC = (1.0 - _DEPR_RDC) ** 0.25
# quarterly depreciation deltas
_DELTA_OC = 1.0 - _SURV_OC
_DELTA_RDC = 1.0 - _SURV_RDC


def _perpetual_inventory(quarterly_add: np.ndarray, delta: float, seed_denom: float) -> np.ndarray:
    """
    Run a perpetual-inventory recurrence on an array of quarterly additions.
    OC_0 = first_non_nan / (seed_denom + delta)
    OC_t = (1 - delta) * OC_{t-1} + add_t
    Returns an array of the same length; positions before the first valid
    observation are NaN.
    """
    out = np.full(len(quarterly_add), np.nan)
    stock = np.nan
    for i, add in enumerate(quarterly_add):
        if np.isnan(add):
            # keep current stock, just carry forward (no new data this period)
            if not np.isnan(stock):
                out[i] = stock
        else:
            if np.isnan(stock):
                # seed
                stock = add / (seed_denom + delta) if (seed_denom + delta) > 0 else 0.0
            else:
                stock = (1.0 - delta) * stock + add
            out[i] = stock
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intangible-heaviness features using PIT SEC fundamentals.
    """
    # --- pull PIT fundamentals ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=[
                "gross_profit_ttm",
                "operating_income_ttm",
                "rnd_expense_ttm",
                "capex_ttm",
                "ppe_net",
            ],
        )

    # --- quarterly SG&A proxy ---
    # SGA_ttm = gross_profit_ttm - operating_income_ttm - rnd_expense_ttm  (clipped >= 0)
    sga_ttm = (
        df["fund_gross_profit_ttm"].fillna(0)
        - df["fund_operating_income_ttm"].fillna(0)
        - df["fund_rnd_expense_ttm"].fillna(0)
    ).clip(lower=0)
    # replace back to NaN where all three inputs were NaN (no fundamental data)
    no_data = (
        df["fund_gross_profit_ttm"].isna()
        & df["fund_operating_income_ttm"].isna()
        & df["fund_rnd_expense_ttm"].isna()
    )
    sga_ttm = sga_ttm.where(~no_data, other=np.nan)

    sga_q = sga_ttm / 4.0   # quarterly addition to OC
    rnd_q = df["fund_rnd_expense_ttm"].where(df["fund_rnd_expense_ttm"].notna(), np.nan) / 4.0

    # --- perpetual-inventory stocks (vectorised as much as possible) ---
    # These involve a recurrence so a Python loop over rows is unavoidable;
    # but the loop is tiny-overhead: ~700 rows, O(n).
    sga_arr = sga_q.to_numpy(dtype=float)
    rnd_arr = rnd_q.to_numpy(dtype=float)

    oc_arr = _perpetual_inventory(sga_arr, delta=_DELTA_OC, seed_denom=0.025)
    rdc_arr = _perpetual_inventory(rnd_arr, delta=_DELTA_RDC, seed_denom=0.025)

    oc = pd.Series(oc_arr, index=df.index)
    rdc = pd.Series(rdc_arr, index=df.index)

    intangibles = oc + rdc
    ppe = df["fund_ppe_net"].clip(lower=0)

    # --- feature 1: intangible share of productive base ---
    total_prod_base = intangibles + ppe
    intang_share = intangibles / total_prod_base.replace(0, np.nan)
    # clip to [0, 1] (should be already, but guard corner cases)
    intang_share = intang_share.clip(lower=0.0, upper=1.0)
    df["ext2_intan_vs_tangible_intang_share"] = intang_share

    # --- feature 2: 252-day linear trend of intangible share ---
    # Use a rolling OLS slope via Var/Cov formula on the window
    # x = [0, 1, ..., 251], y = intang_share values
    window = 252
    roll = intang_share.rolling(window, min_periods=max(window // 4, 20))

    def _slope(y: np.ndarray) -> float:
        """OLS slope via Var(x)/Cov(x,y) with x = 0..n-1."""
        n = len(y)
        if n < 2:
            return np.nan
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        xv = np.arange(n, dtype=float)
        xm = xv[mask].mean()
        ym = y[mask].mean()
        ssxx = np.sum((xv[mask] - xm) ** 2)
        ssxy = np.sum((xv[mask] - xm) * (y[mask] - ym))
        if ssxx == 0:
            return np.nan
        return ssxy / ssxx

    trend = roll.apply(_slope, raw=True)
    df["ext2_intan_vs_tangible_intang_share_trend"] = trend

    # --- feature 3: intangible investment / capex ratio ---
    # intangible_investment = quarterly increments (sga_q + rnd_q) TTM ≈ sga_ttm + rnd_ttm
    intang_invest_ttm = sga_ttm.fillna(0) + df["fund_rnd_expense_ttm"].fillna(0)
    intang_invest_ttm = intang_invest_ttm.where(~no_data, other=np.nan)
    capex_ttm = df["fund_capex_ttm"].abs()  # capex is often reported negative
    capex_ttm = capex_ttm.replace(0, np.nan)
    intang_to_capex = intang_invest_ttm / capex_ttm
    # winsorise extreme ratios (e.g. near-zero capex firms)
    intang_to_capex = intang_to_capex.clip(lower=0.0, upper=50.0)
    df["ext2_intan_vs_tangible_intang_to_capex"] = intang_to_capex

    # --- drop scratch fund_ columns ---
    scratch_cols = [
        "fund_gross_profit_ttm",
        "fund_operating_income_ttm",
        "fund_rnd_expense_ttm",
        "fund_capex_ttm",
        "fund_ppe_net",
    ]
    df = df.drop(columns=[c for c in scratch_cols if c in df.columns])

    return df
