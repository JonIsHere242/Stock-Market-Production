"""
ext3_cf_earnings_gap: Cash-flow vs earnings divergence (quality).

Accrual gap = (net_income_ttm - operating_cash_flow_ttm) / assets.
High accrual gap => low earnings quality (accruals inflate earnings).
Also produces the 252-day change in accrual gap and the cash conversion
ratio (operating_cash_flow_ttm / net_income_ttm).

Per-ticker, point-in-time safe via _fundamentals.as_of (backward merge on
filed_date). ~84% coverage; ETFs/foreign lead rows will be NaN.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (by file path — never import from package)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_cf_earnings_gap",
    "description": (
        "Cash-flow vs earnings divergence (earnings quality). "
        "ext3_cf_earnings_gap_accrual = (net_income_ttm - operating_cash_flow_ttm) / assets "
        "(higher = more accruals = lower quality; typically negatively predictive). "
        "ext3_cf_earnings_gap_accrual_ch252 = 252-day change in that ratio (momentum of deterioration). "
        "ext3_cf_earnings_gap_cash_conv = operating_cash_flow_ttm / net_income_ttm (cash conversion; "
        "near 1 = high quality; extreme values winsorised). "
        "Per-ticker proxy using PIT fundamentals; cross-sectional ranking not applied."
    ),
    "requires": ["Close"],   # Close present by convention; fundamentals come from helper
    "produces": [
        "ext3_cf_earnings_gap_accrual",
        "ext3_cf_earnings_gap_accrual_ch252",
        "ext3_cf_earnings_gap_cash_conv",
    ],
    "tags": ["quality", "fundamentals", "accruals", "cash_flow", "earnings"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_orgcap); spec: ext3_cf_earnings_gap",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add accrual-gap and cash-conversion features. Returns df with new columns."""
    # Pull PIT fundamentals (backward merge on filed_date — no lookahead)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=["net_income_ttm", "operating_cash_flow_ttm", "assets"],
        )

    ni = df["fund_net_income_ttm"]
    ocf = df["fund_operating_cash_flow_ttm"]
    assets = df["fund_assets"]

    # ------------------------------------------------------------------
    # 1. Accrual gap = (net_income_ttm - operating_cash_flow_ttm) / assets
    # ------------------------------------------------------------------
    denom_assets = assets.replace(0, np.nan)
    accrual = (ni - ocf) / denom_assets
    # Guard infinities (shouldn't occur after nan-replace but be safe)
    accrual = accrual.replace([np.inf, -np.inf], np.nan)
    df["ext3_cf_earnings_gap_accrual"] = accrual

    # ------------------------------------------------------------------
    # 2. 252-day change in accrual gap (momentum of quality deterioration)
    #    Use .diff(252) on the daily series — fundamentals update ~quarterly
    #    so this captures roughly 4 quarters of drift.
    # ------------------------------------------------------------------
    accrual_ch252 = accrual.diff(252)
    accrual_ch252 = accrual_ch252.replace([np.inf, -np.inf], np.nan)
    df["ext3_cf_earnings_gap_accrual_ch252"] = accrual_ch252

    # ------------------------------------------------------------------
    # 3. Cash conversion ratio = operating_cash_flow_ttm / net_income_ttm
    #    Values near 1 => high quality. Winsorise to [-10, 10] to avoid
    #    huge values when net_income_ttm is near zero.
    # ------------------------------------------------------------------
    denom_ni = ni.replace(0, np.nan)
    cash_conv = ocf / denom_ni
    cash_conv = cash_conv.replace([np.inf, -np.inf], np.nan)
    # Soft winsorise (clip, not drop) so ranking is stable
    cash_conv = cash_conv.clip(-10.0, 10.0)
    df["ext3_cf_earnings_gap_cash_conv"] = cash_conv

    # ------------------------------------------------------------------
    # Drop scratch fund_ columns we are NOT listing in produces
    # ------------------------------------------------------------------
    for col in ["fund_net_income_ttm", "fund_operating_cash_flow_ttm", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
