"""
osap_grcapx3y — Change in capex (three years)
Source: OpenSourceAP (Chen-Zimmermann); Anderson and Garcia-Feijoo 2006.

Signal: capex / sum(capex_{t-1}, capex_{t-2}, capex_{t-3})  (predicted sign: -1, i.e. high capex growth → lower future returns).
Per-ticker proxy: uses PIT fundamentals (capex_ttm), with ppe_net as fallback when capex is unavailable.
Prior-year capex levels are approximated by shifting the merged fundamental series by ~252, ~504, ~756 trading days
so that past-filed values are used (fully lookahead-safe via _fundamentals.as_of).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
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
    "name": "osap_grcapx3y",
    "description": (
        "Per-ticker capex growth ratio: capex_ttm divided by the sum of capex_ttm values "
        "from approximately one, two, and three years prior (approximated by shifting the "
        "PIT-merged fundamental series by ~252, ~504, ~756 trading days). Falls back to "
        "ppe_net when capex_ttm is entirely missing. Predicted sign is -1 (high investment "
        "growth → lower future returns, Anderson and Garcia-Feijoo 2006 via "
        "Chen-Zimmermann OpenSourceAP). Cross-sectional ranking replaced by per-ticker "
        "time-series ratio — captures the same investment-acceleration signal."
    ),
    "requires": [],
    "produces": [
        "osap_grcapx3y_ratio",   # capex / (capx_1y + capx_2y + capx_3y)
        "osap_grcapx3y_yoy",     # yoy change: capex / capex_1y - 1
        "osap_grcapx3y_accel",   # acceleration: yoy change vs prior yoy change
    ],
    "tags": ["fundamentals", "investment", "capex", "accounting", "openSourceAP"],
    "version": "1.0",
    "author": "Anderson and Garcia-Feijoo 2006; OpenSourceAP / Chen-Zimmermann; block by Claude",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
_YEAR_DAYS = 252   # approximate trading days per year


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Pull PIT fundamentals (capex_ttm + ppe_net fallback) ----------
    df = _fundamentals.as_of(df, fields=["capex_ttm", "ppe_net"])

    # Determine which series to use as the capex proxy
    # If capex_ttm is entirely NaN for this ticker, fall back to ppe_net
    capex_series = df["fund_capex_ttm"].copy()
    use_fallback = capex_series.isna().all()
    if use_fallback:
        capex_series = df["fund_ppe_net"].copy()

    # --- 2. Build lagged capex levels (per-ticker time-series shifts) ------
    # shift by integer row counts; safe because df is already sorted ascending
    capx_1y = capex_series.shift(_YEAR_DAYS)
    capx_2y = capex_series.shift(2 * _YEAR_DAYS)
    capx_3y = capex_series.shift(3 * _YEAR_DAYS)

    # --- 3. Three-year ratio: capex / (sum of prior three annual levels) ---
    denom_3y = capx_1y + capx_2y + capx_3y
    # Guard against zero / negative denominators (ppe_net can be near-zero)
    denom_3y_safe = denom_3y.where(denom_3y.abs() > 1e-9)
    ratio = capex_series / denom_3y_safe
    # Replace inf/-inf with NaN
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # --- 4. YoY change: capex / capex_1y - 1 -----------------------------
    capx_1y_safe = capx_1y.where(capx_1y.abs() > 1e-9)
    yoy = (capex_series / capx_1y_safe) - 1.0
    yoy = yoy.replace([np.inf, -np.inf], np.nan)

    # --- 5. Acceleration: this year's yoy vs prior year's yoy -------------
    # prior yoy: capx_1y / capx_2y - 1
    capx_2y_safe = capx_2y.where(capx_2y.abs() > 1e-9)
    yoy_prior = (capx_1y / capx_2y_safe) - 1.0
    yoy_prior = yoy_prior.replace([np.inf, -np.inf], np.nan)
    accel = yoy - yoy_prior

    # --- 6. Assign produced columns ---------------------------------------
    df["osap_grcapx3y_ratio"] = ratio
    df["osap_grcapx3y_yoy"] = yoy
    df["osap_grcapx3y_accel"] = accel

    # Drop scratch fundamental columns (not in produces)
    df = df.drop(columns=["fund_capex_ttm", "fund_ppe_net"], errors="ignore")

    return df
