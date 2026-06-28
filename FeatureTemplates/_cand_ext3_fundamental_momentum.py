"""
Fundamental momentum (improving profitability) — ext3_fundamental_momentum

Captures the trend in ROE and net margin (252-day changes) and their composite
z-score over a 504-day window. Positive values indicate improving profitability,
which predicts forward returns per fundamental-momentum literature.

Per-ticker PIT implementation via _fundamentals.as_of (backward merge on filed_date).
Coverage ~84%; ETFs and foreign stocks will be NaN.
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
    "name": "ext3_fundamental_momentum",
    "description": (
        "Fundamental momentum via improving profitability: "
        "252-day change in ROE, 252-day change in net_margin, and a composite "
        "z-score (average of both z-scored over a 504-day rolling window). "
        "Uses point-in-time SEC fundamentals (filed_date backward merge). "
        "Coverage ~84%; ETFs/foreign rows are NaN."
    ),
    "requires": [],  # all inputs come from PIT fundamentals, not OHLCV columns
    "produces": [
        "ext3_fundamental_momentum_roe_chg",
        "ext3_fundamental_momentum_margin_chg",
        "ext3_fundamental_momentum_composite_z",
    ],
    "tags": ["fundamentals", "momentum", "profitability", "quality"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_orgcap); spec ext3_fundamental_momentum",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOW_CHG = 252   # approx 1-year change in fundamental
_WINDOW_Z = 504     # approx 2-year rolling z-score window


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with guarded std (returns NaN when std == 0 or too few obs)."""
    mu = s.rolling(window, min_periods=max(window // 4, 21)).mean()
    sd = s.rolling(window, min_periods=max(window // 4, 21)).std(ddof=1)
    sd = sd.replace(0.0, np.nan)
    return (s - mu) / sd


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add three fundamental-momentum columns to df (one stock at a time)."""

    # Pull PIT fundamentals: ROE and net_margin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["roe", "net_margin"])

    # --- 252-day change in ROE ---
    roe = df["fund_roe"].astype(float)
    roe_chg = roe - roe.shift(_WINDOW_CHG)
    df["ext3_fundamental_momentum_roe_chg"] = roe_chg

    # --- 252-day change in net_margin ---
    margin = df["fund_net_margin"].astype(float)
    margin_chg = margin - margin.shift(_WINDOW_CHG)
    df["ext3_fundamental_momentum_margin_chg"] = margin_chg

    # --- Composite z-score: average z of (roe_chg, margin_chg) over 504d ---
    z_roe = _rolling_zscore(roe_chg, _WINDOW_Z)
    z_margin = _rolling_zscore(margin_chg, _WINDOW_Z)

    # Average the two z-scores; if one is NaN the other alone carries the signal
    composite = np.where(
        z_roe.isna() & z_margin.isna(),
        np.nan,
        np.nanmean(np.column_stack([z_roe.values, z_margin.values]), axis=1),
    )
    df["ext3_fundamental_momentum_composite_z"] = composite

    # Drop scratch fund_ columns we are NOT producing
    df = df.drop(columns=["fund_roe", "fund_net_margin"], errors="ignore")

    return df
