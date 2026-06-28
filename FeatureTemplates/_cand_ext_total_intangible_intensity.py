"""
_cand_ext_total_intangible_intensity.py

Total intangible investment intensity & trend.

Per-ticker proxy: intangible investment is approximated as
  SG&A + R&D  where  SG&A = gross_profit_ttm - operating_income_ttm (implied SG&A)
Three produced signals:
  ext_total_intangible_intensity_rev   -- intangible_invest / revenue_ttm  (intensity vs sales)
  ext_total_intangible_intensity_yoy   -- YoY change in the revenue-intensity ratio
  ext_total_intangible_intensity_asset -- intangible_invest / total assets
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ── load PIT fundamentals helper ──────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "ext_total_intangible_intensity",
    "description": (
        "Total intangible investment intensity & trend (per-ticker, PIT-safe). "
        "Intangible investment = implied SG&A (gross_profit_ttm - operating_income_ttm) "
        "+ rnd_expense_ttm. "
        "Produces: (1) intangible invest / revenue_ttm (intensity vs. sales), "
        "(2) YoY change in that ratio (trend/momentum of intangible commitment), "
        "(3) intangible invest / total assets (balance-sheet intensity). "
        "Extends the osap_orgcap organisational-capital family on a DIFFERENT axis: "
        "SG&A+R&D share of sales/assets vs. the capitalised-cost depreciation approach."
    ),
    "requires": [],
    "produces": [
        "ext_total_intangible_intensity_rev",
        "ext_total_intangible_intensity_yoy",
        "ext_total_intangible_intensity_asset",
    ],
    "tags": ["fundamentals", "intangibles", "rnd", "sga", "quality"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner osap_orgcap; "
        "spec SOURCE: Extension of osap_orgcap (project feature-discovery pipeline)"
    ),
}

# approximate number of trading days in a year for YoY rolling shift
_APPROX_YEAR_ROWS = 252


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── 1. pull PIT fundamentals ──────────────────────────────────────────────
    fields = [
        "gross_profit_ttm",
        "operating_income_ttm",
        "rnd_expense_ttm",
        "revenue_ttm",
        "assets",
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # ── 2. derive intangible investment (all in currency units, TTM) ──────────
    # implied SG&A = gross_profit_ttm - operating_income_ttm
    # rnd_expense_ttm is typically reported as a positive number (cost)
    sga = df["fund_gross_profit_ttm"] - df["fund_operating_income_ttm"]
    rnd = df["fund_rnd_expense_ttm"].abs()          # guard sign convention

    intangible_invest = sga + rnd                   # may be NaN where coverage absent

    # ── 3. intensity vs revenue ───────────────────────────────────────────────
    rev = df["fund_revenue_ttm"].replace(0, np.nan)
    intensity_rev = intangible_invest / rev

    # clip to guard extreme outliers (e.g. near-zero or negative revenue)
    intensity_rev = intensity_rev.where(
        (intensity_rev > -5) & (intensity_rev < 5), other=np.nan
    )

    # ── 4. YoY change in intensity_rev ───────────────────────────────────────
    # shift by ~252 rows (approx 1 year of daily bars per stock)
    prior_year = intensity_rev.shift(_APPROX_YEAR_ROWS)
    yoy_change = intensity_rev - prior_year         # positive = firm investing more in intangibles

    # ── 5. intensity vs assets ────────────────────────────────────────────────
    assets = df["fund_assets"].replace(0, np.nan)
    intensity_asset = intangible_invest / assets
    intensity_asset = intensity_asset.where(
        (intensity_asset > -5) & (intensity_asset < 5), other=np.nan
    )

    # ── 6. assign produced columns ────────────────────────────────────────────
    df["ext_total_intangible_intensity_rev"]   = intensity_rev.replace([np.inf, -np.inf], np.nan)
    df["ext_total_intangible_intensity_yoy"]   = yoy_change.replace([np.inf, -np.inf], np.nan)
    df["ext_total_intangible_intensity_asset"] = intensity_asset.replace([np.inf, -np.inf], np.nan)

    # ── 7. drop scratch fund_* columns not in produces ───────────────────────
    scratch_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=scratch_cols)

    return df
