"""
osap_fgr5yrlag — 5-Year Lagged Sales Growth Rate (per-ticker proxy)

Economic signal: Firms with high past long-run sales growth (glamour stocks)
tend to be overpriced and subsequently underperform (contrarian, sign = -1).
This is the OSAP "fgr5yrlag" anomaly from LaPorta (1996) / Lakonishok, Shleifer
& Vishny (1994), also catalogued in Chen & Zimmermann OpenSourceAP.

Cross-sectional note: The canonical signal ranks stocks by their 5-year lagged
CAGR of sales (revenue). This per-ticker implementation computes the same
economic quantity — the 5-year compound annual revenue growth rate lagged by
one fiscal year — using PIT fundamentals (revenue_ttm), so there is no leakage.
The cross-sectional ranking is left to the downstream model (XGBoost).
"""

from __future__ import annotations
import importlib.util as _ilu
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
    "name": "osap_fgr5yrlag",
    "description": (
        "5-year lagged sales (revenue) CAGR — the 'glamour vs value' growth signal. "
        "Uses PIT revenue_ttm to compute a rolling 5-year compound annual growth rate, "
        "then lags that estimate by ~1 year (252 trading days) to mimic the fiscal-year "
        "lag in the OSAP definition. High values indicate glamour / past-winner stocks "
        "that tend to underperform (predicted sign = -1). Inherently cross-sectional in "
        "the original paper; this is the faithful per-ticker proxy using PIT fundamentals."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_fgr5yrlag_cagr5y",       # rolling 5-yr revenue CAGR (current)
        "osap_fgr5yrlag_cagr5y_lag1y",  # same, lagged ~1yr (the OSAP lag)
        "osap_fgr5yrlag_cagr_chg",      # change: current minus lagged (momentum of growth)
    ],
    "tags": ["fundamental", "growth", "contrarian", "osap", "accounting"],
    "version": "1.0",
    "author": (
        "LaPorta 1996; Lakonishok, Shleifer & Vishny 1994; "
        "Chen & Zimmermann OpenSourceAP (osap_fgr5yrlag). "
        "Per-ticker proxy implementation."
    ),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DAYS_PER_YEAR = 252          # trading days
_WINDOW_5Y = 5 * _DAYS_PER_YEAR   # 1260 trading days back for 5-yr CAGR
_LAG_1Y = _DAYS_PER_YEAR          # 252-day lag to replicate fiscal-year lag
_MIN_OBS = 3 * _DAYS_PER_YEAR     # need at least 3 years of data to be meaningful


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 5-year lagged revenue CAGR for one ticker.

    Steps:
    1. Attach PIT revenue_ttm via backward merge on filed_date.
    2. Compute rolling 5yr CAGR using revenue from now vs revenue from 5yr ago.
       CAGR = (rev_now / rev_5ya)^(1/5) - 1  (annualised)
    3. Lag by 252 trading days to mirror the fiscal-year lag in the OSAP spec.
    4. Compute change (current CAGR minus lagged CAGR) as a momentum-of-growth signal.
    """
    # --- 1. Attach PIT fundamentals ----------------------------------------
    df = _fundamentals.as_of(df, fields=["revenue_ttm"])

    rev = df["fund_revenue_ttm"].copy()

    # Guard: revenue must be positive to compute meaningful growth rates
    rev = rev.where(rev > 0, other=np.nan)

    n = len(df)

    # --- 2. Rolling 5-year CAGR --------------------------------------------
    # rev_5ya[i] = revenue_ttm value ~5 years (1260 bars) ago
    # We use .shift(1260) which is a pure lag — no lookahead.
    rev_5ya = rev.shift(_WINDOW_5Y)

    # CAGR = (rev_now / rev_5ya)^(1/5) - 1
    ratio = rev / rev_5ya
    ratio = ratio.where(ratio > 0, other=np.nan)  # guard against negative / zero
    cagr5y = ratio ** (1.0 / 5.0) - 1.0

    # Clamp extreme values (e.g. mergers / data errors): cap at ±5 (500% per year)
    cagr5y = cagr5y.clip(-5.0, 5.0)

    # Require minimum history: NaN out until we have _MIN_OBS bars of revenue data
    rev_valid_count = rev.notna().cumsum()
    cagr5y = cagr5y.where(rev_valid_count >= _MIN_OBS, other=np.nan)

    # --- 3. Lagged CAGR (the OSAP "fgr5yrlag") ----------------------------
    cagr5y_lag1y = cagr5y.shift(_LAG_1Y)

    # --- 4. Change in growth rate ------------------------------------------
    cagr_chg = cagr5y - cagr5y_lag1y

    # --- 5. Assign columns -------------------------------------------------
    df["osap_fgr5yrlag_cagr5y"] = cagr5y
    df["osap_fgr5yrlag_cagr5y_lag1y"] = cagr5y_lag1y
    df["osap_fgr5yrlag_cagr_chg"] = cagr_chg

    # Drop scratch fundamentals column not in produces
    df = df.drop(columns=["fund_revenue_ttm"], errors="ignore")

    return df
