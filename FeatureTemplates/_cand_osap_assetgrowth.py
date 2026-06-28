"""
osap_assetgrowth — Asset Growth (Cooper, Gulen & Schill 2008)

Annual growth rate of total assets (at), sourced from PIT SEC fundamentals.
High asset growth predicts lower future returns (predicted sign: -1).
Proxy notes: per-ticker only; cross-sectional rank done downstream by the
framework's --add_xs_features pass.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (by file path — never `from FeatureTemplates`)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA: dict = {
    "name": "osap_assetgrowth",
    "description": (
        "Asset Growth (Cooper, Gulen & Schill 2008; OpenSourceAP / Chen-Zimmermann). "
        "Computes the year-over-year growth rate of total assets from PIT SEC filings "
        "(as_of backward merge on filed_date — no lookahead). "
        "Predicted sign is -1 (high asset growth → lower subsequent returns). "
        "Per-ticker implementation; cross-sectional ranking is applied downstream. "
        "Produces: level growth rate, a 2-quarter smoothed variant, and a "
        "trailing acceleration (change in growth rate) for the dynamic/slope signal."
    ),
    "requires": [],          # no OHLCV dependency — fundamentals only
    "produces": [
        "osap_assetgrowth_yoy",     # raw YoY asset growth rate (primary signal)
        "osap_assetgrowth_smooth",  # 2-obs smoothed (reduce filing noise)
        "osap_assetgrowth_accel",   # change in growth rate (acceleration)
    ],
    "tags": ["fundamentals", "investment", "asset_growth", "osap"],
    "version": "1.0.0",
    "author": "Cooper, Gulen & Schill (2008); OpenSourceAP Chen-Zimmermann dataset",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add asset-growth features to a single-ticker OHLCV dataframe."""
    # Pull PIT total assets (backward merge on filed_date)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["assets"])

    # fund_assets: point-in-time total assets as publicly known on each date
    a = df["fund_assets"]

    # YoY growth: (a_t - a_{t-252}) / |a_{t-252}|
    # Use ~252 trading-day lag as proxy for annual; division guarded vs zero.
    lag_annual = a.shift(252)
    denom = lag_annual.abs().replace(0, np.nan)
    yoy = (a - lag_annual) / denom

    # Clip extreme outliers (mergers/spin-offs can create >10× spikes)
    yoy = yoy.clip(-5.0, 5.0)

    # 2-observation rolling mean (smooths back-to-back filing noise)
    smooth = yoy.rolling(window=2, min_periods=1).mean()

    # Acceleration: change in YoY growth vs one year ago
    lag_yoy = yoy.shift(252)
    accel = yoy - lag_yoy

    df["osap_assetgrowth_yoy"] = yoy
    df["osap_assetgrowth_smooth"] = smooth
    df["osap_assetgrowth_accel"] = accel

    # Drop scratch fundamental column not listed in produces
    df.drop(columns=["fund_assets"], inplace=True)

    return df
