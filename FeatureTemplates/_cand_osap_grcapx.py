"""
osap_grcapx — Growth in Capital Expenditures (gr_capx)

Signal: Year-over-year percentage growth in capital expenditures scaled by lagged assets.
High capex growth → over-investment → negative future returns (anomaly from Anderson & Garcia-Feijoo 2006;
also tabulated in Hou, Xue & Zhang OSAP library as "grcapx").

Per-ticker PIT implementation using SEC fundamentals (capex_ttm, assets).
Cross-sectional ranking is not done here (per-block rules); the raw level is returned
along with a 2-quarter momentum of the ratio and a price-normalised proxy for
model diversity.
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
    "name": "osap_grcapx",
    "description": (
        "Growth in Capital Expenditures (gr_capx) — year-over-year change in "
        "capital expenditures (TTM) scaled by lagged total assets. High capex "
        "growth predicts lower returns (over-investment anomaly). "
        "Implemented as a per-ticker PIT signal using SEC fundamentals; "
        "cross-sectional ranking is not applied (per-block constraint). "
        "Produces: (1) osap_grcapx_ratio = capex_ttm / assets (capex intensity); "
        "(2) osap_grcapx_yoy = YoY growth in capex_ttm / assets_lag (main signal); "
        "(3) osap_grcapx_accel = recent change in the ratio (2-period delta of ratio) "
        "as a capex-acceleration proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_grcapx_ratio",   # capex_ttm / assets  (capex intensity level)
        "osap_grcapx_yoy",     # (capex_ttm_t - capex_ttm_t-4q) / assets_t-4q  (main OSAP signal)
        "osap_grcapx_accel",   # change in ratio over ~2 quarters (acceleration)
    ],
    "tags": ["fundamentals", "investment", "capex", "osap", "anomaly"],
    "version": "1.0.0",
    "author": "Anderson & Garcia-Feijoo (2006); Hou, Xue & Zhang OSAP library (grcapx). Implementation by Claude.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute capex-growth features using PIT SEC fundamentals."""

    # Pull point-in-time fundamentals (backward merge on filed_date — no lookahead)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["capex_ttm", "assets"])

    capex = df["fund_capex_ttm"]   # trailing twelve months capex (positive = spending)
    assets = df["fund_assets"]     # total assets

    # --- Feature 1: capex intensity (ratio) ---
    # capex_ttm / assets  — guard zero/missing assets
    safe_assets = assets.replace(0, np.nan)
    ratio = capex / safe_assets
    df["osap_grcapx_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)

    # --- Feature 2: YoY growth in capex intensity (main OSAP signal) ---
    # Fundamentals arrive ~quarterly via filed_date; shift by ~252 trading days ≈ 1 year.
    # We use the ratio series (already PIT) and measure YoY change.
    # shift(252) on a daily series approximates "same quarter one year ago".
    ratio_lag_1y = ratio.shift(252)
    safe_lag_1y = ratio_lag_1y.replace(0, np.nan)

    # Growth = (ratio_now - ratio_1y_ago) / |ratio_1y_ago|   (signed pct change)
    yoy = (ratio - ratio_lag_1y) / safe_lag_1y.abs()
    df["osap_grcapx_yoy"] = yoy.replace([np.inf, -np.inf], np.nan)

    # --- Feature 3: acceleration — change over ~2 quarters (≈126 trading days) ---
    ratio_lag_2q = ratio.shift(126)
    safe_lag_2q = ratio_lag_2q.replace(0, np.nan)
    accel = (ratio - ratio_lag_2q) / safe_lag_2q.abs()
    df["osap_grcapx_accel"] = accel.replace([np.inf, -np.inf], np.nan)

    # Drop scratch fund_ columns not in produces
    df = df.drop(columns=["fund_capex_ttm", "fund_assets"], errors="ignore")

    return df
