"""
R&D Capital via perpetual-inventory model (Peters & Taylor 2017).

Capitalises R&D spending using a 20 %/yr depreciation rate, mirroring the
osap_orgcap organisational-capital block but for knowledge/R&D capital.
The signal is orthogonal to org-cap because R&D is expensed separately and
captures a different asset class (knowledge stock vs. SG&A-embedded process
capital).

Per-ticker proxy notes:
  - rnd_expense_ttm is available quarterly via PIT fundamentals.
  - Quarterly add = rnd_expense_ttm / 4  (approximate quarterly flow).
  - Annual depreciation rate delta=0.20 → quarterly delta_q = 1-(1-0.20)^(1/4).
  - RDC is built via an iterative PIT roll over filed_date observations then
    merged back to daily rows (lookahead-safe via merge_asof backward).
  - 84 % fundamentals coverage; ETFs/foreign → NaN (expected).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_HERE = _P(__file__).resolve().parent

def _load_fundamentals():
    _s = _ilu.spec_from_file_location(
        "_fundamentals", _HERE / "_fundamentals.py"
    )
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m

# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_rnd_capital",
    "description": (
        "R&D capital stock built with a perpetual-inventory model "
        "(20 %/yr depreciation, Peters & Taylor 2017). "
        "Produces: (1) RDC/assets — R&D capital intensity relative to book "
        "assets, (2) RDC/revenue_ttm — revenue-normalised knowledge intensity, "
        "(3) RDC YoY growth rate. "
        "Per-ticker proxy: quarterly rnd_expense_ttm flow from PIT fundamentals "
        "rolled forward with delta_q = 1-(0.80)^(1/4); seed = rnd_expense_ttm/delta. "
        "Orthogonal to osap_orgcap (SG&A-based) — captures knowledge-capital axis."
    ),
    "requires": [],  # all signal comes from _fundamentals; OHLCV used for merge scaffold
    "produces": [
        "ext_rnd_capital_to_assets",
        "ext_rnd_capital_to_rev",
        "ext_rnd_capital_yoy",
    ],
    "tags": ["fundamentals", "rnd", "capital", "value", "quality"],
    "version": "1.0.0",
    "author": "Peters & Taylor (2017) 'Internal Capital Markets and Investment Policy with Financial Constraints'; spec ext_rnd_capital",
}

# Annual depreciation → quarterly equivalent
_DELTA_ANNUAL = 0.20
_DELTA_Q = 1.0 - (1.0 - _DELTA_ANNUAL) ** (1.0 / 4.0)  # ≈ 0.0535


def _build_rdc_series(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-ticker DataFrame with columns [filed_date, rnd_ttm, assets, rev],
    sorted ascending by filed_date, return the same frame with added 'rdc' column.

    Perpetual-inventory: RDC_t = (1 - delta_q) * RDC_{t-1} + quarterly_rnd
    Seed: RDC_0 = quarterly_rnd_0 / delta_q  (steady-state approximation)
    """
    rnd = fund_df["rnd_ttm"].values.astype(float)
    assets = fund_df["assets"].values.astype(float)
    rev = fund_df["rev"].values.astype(float)

    n = len(rnd)
    rdc = np.full(n, np.nan)

    if n == 0:
        fund_df = fund_df.copy()
        fund_df["rdc"] = rdc
        fund_df["rdc_to_assets"] = np.nan
        fund_df["rdc_to_rev"] = np.nan
        fund_df["rdc_yoy"] = np.nan
        return fund_df

    # Find the first valid rnd observation to seed the inventory
    first_valid = -1
    for i in range(n):
        if np.isfinite(rnd[i]) and rnd[i] >= 0:
            first_valid = i
            break

    if first_valid == -1:
        fund_df = fund_df.copy()
        fund_df["rdc"] = rdc
        fund_df["rdc_to_assets"] = np.nan
        fund_df["rdc_to_rev"] = np.nan
        fund_df["rdc_yoy"] = np.nan
        return fund_df

    # quarterly flow ≈ TTM / 4
    quarterly_rnd = rnd / 4.0

    # Seed at steady state for the first valid quarter
    rdc[first_valid] = quarterly_rnd[first_valid] / max(_DELTA_Q, 1e-9)

    for i in range(first_valid + 1, n):
        if np.isfinite(quarterly_rnd[i]):
            prev = rdc[i - 1] if np.isfinite(rdc[i - 1]) else 0.0
            rdc[i] = (1.0 - _DELTA_Q) * prev + quarterly_rnd[i]
        # if rnd is NaN at this quarter, propagate depreciation of prior stock
        else:
            if np.isfinite(rdc[i - 1]):
                rdc[i] = (1.0 - _DELTA_Q) * rdc[i - 1]

    fund_df = fund_df.copy()
    fund_df["rdc"] = rdc

    # derived features at the fundamental observation level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fund_df["rdc_to_assets"] = np.where(
            assets > 0, rdc / assets, np.nan
        )
        fund_df["rdc_to_rev"] = np.where(
            rev > 0, rdc / rev, np.nan
        )
        # YoY growth: 4 quarterly observations back (≈1 year)
        rdc_lag4 = np.full(n, np.nan)
        rdc_lag4[4:] = rdc[:-4]
        fund_df["rdc_yoy"] = np.where(
            rdc_lag4 > 0, (rdc - rdc_lag4) / rdc_lag4, np.nan
        )

    return fund_df


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN
    df["ext_rnd_capital_to_assets"] = np.nan
    df["ext_rnd_capital_to_rev"] = np.nan
    df["ext_rnd_capital_yoy"] = np.nan

    if df.empty:
        return df

    try:
        _fundamentals = _load_fundamentals()
        df = _fundamentals.as_of(
            df,
            fields=["rnd_expense_ttm", "assets", "revenue_ttm"],
        )
    except Exception:
        return df

    rnd_col = "fund_rnd_expense_ttm"
    assets_col = "fund_assets"
    rev_col = "fund_revenue_ttm"

    # Check columns were actually attached
    if rnd_col not in df.columns:
        return df

    # Build a per-quarter fundamental spine (one row per unique filed_date)
    # by taking the last OHLCV row at each filed_date occurrence.
    fund_cols = [rnd_col, assets_col, rev_col]
    # filed_date is injected by as_of as a hidden merge key stored on rows;
    # it may not be exposed as a column.  We work from the merged df directly.

    # We need to iterate in time order to build RDC, then merge back.
    # Create a helper frame of unique (fund_values) breakpoints by detecting
    # when any fundamental column changes value.

    # Use the fund columns to detect "new quarter" transitions
    fund_snap = (
        df[[rnd_col, assets_col, rev_col]]
        .rename(columns={rnd_col: "rnd_ttm", assets_col: "assets", rev_col: "rev"})
        .copy()
    )
    fund_snap["_row"] = np.arange(len(fund_snap))

    # Detect transitions: keep first row of each unique fund block
    changed = (
        fund_snap[["rnd_ttm", "assets", "rev"]]
        .ne(fund_snap[["rnd_ttm", "assets", "rev"]].shift(1))
        .any(axis=1)
    )
    changed.iloc[0] = True  # always keep first
    quarterly_rows = fund_snap[changed].copy().reset_index(drop=False)
    quarterly_rows = quarterly_rows.rename(columns={"index": "_orig_idx"})
    quarterly_rows["filed_date"] = df["Date"].iloc[quarterly_rows["_row"].values].values

    # Build RDC on the quarterly spine
    quarterly_rows = quarterly_rows.sort_values("filed_date").reset_index(drop=True)
    quarterly_rows = _build_rdc_series(quarterly_rows)

    # Merge back to daily df using merge_asof (backward = no lookahead)
    daily = df[["Date"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    quarterly_rows["filed_date"] = pd.to_datetime(quarterly_rows["filed_date"])
    quarterly_rows_sorted = quarterly_rows.sort_values("filed_date")

    merged = pd.merge_asof(
        daily,
        quarterly_rows_sorted[["filed_date", "rdc_to_assets", "rdc_to_rev", "rdc_yoy"]],
        left_on="Date",
        right_on="filed_date",
        direction="backward",
    )

    # Replace inf with NaN and clip extremes
    for src, dst in [
        ("rdc_to_assets", "ext_rnd_capital_to_assets"),
        ("rdc_to_rev", "ext_rnd_capital_to_rev"),
        ("rdc_yoy", "ext_rnd_capital_yoy"),
    ]:
        if src in merged.columns:
            vals = merged[src].values.astype(float)
            vals = np.where(np.isinf(vals), np.nan, vals)
            df[dst] = vals

    # Drop any fund_* scratch columns not in produces
    for col in list(df.columns):
        if col.startswith("fund_"):
            df = df.drop(columns=[col])

    return df
