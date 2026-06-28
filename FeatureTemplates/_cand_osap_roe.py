"""
osap_roe — Return on Equity (ROE) feature block.

Source: Jensen, Kelly & Pedersen (2023) "Is There a Replication Crisis in Finance?"
        Open Source Asset Pricing (OSAP) project — characteristic `roe`.
        Original construction: net_income_ttm / book_equity (shareholders' equity),
        using point-in-time (PIT) filed data, updated at each fiscal quarter.

Per-ticker proxy note:
  The OSAP signal is cross-sectionally ranked each month. Per-ticker we cannot rank
  across the universe, so we expose:
    osap_roe_level  — PIT ROE (net_income_ttm / equity), updated on each filing date.
    osap_roe_chg    — change in ROE versus 4 quarters prior (252 trading-day lag on
                      the filed series), capturing profitability momentum.
    osap_roe_zscore — 252-day rolling z-score of osap_roe_level (captures deviation
                      from the stock's own historical mean — a within-ticker signal).
  Missing equity / negative equity rows are left NaN (no fill).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
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
    "name": "osap_roe",
    "description": (
        "Return on Equity (ROE) from OSAP — net_income_ttm / shareholders_equity "
        "using point-in-time filed fundamentals. Produces a level, a 4-quarter "
        "change (profitability momentum), and a 252-day rolling z-score. "
        "Cannot cross-sectionally rank per-ticker; z-score is within-ticker only."
    ),
    "requires": [],
    "produces": ["osap_roe_level", "osap_roe_chg", "osap_roe_zscore"],
    "tags": ["fundamental", "profitability", "osap", "roe", "pit"],
    "version": "1.0.0",
    "author": "Jensen, Kelly & Pedersen (2023) OSAP — per-ticker proxy impl.",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_roe_level, osap_roe_chg, osap_roe_zscore to df."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Pull PIT fundamentals: net income TTM + shareholders' equity
        df = _fundamentals.as_of(df, fields=["net_income_ttm", "equity"])

    ni = df["fund_net_income_ttm"]
    eq = df["fund_equity"].replace(0, np.nan)  # guard divide-by-zero

    # --- Level: ROE = net_income_ttm / equity ---------------------------
    roe_level = ni / eq
    # Winsorise extreme values (e.g., tiny near-zero equity makes ROE blow up)
    p01 = roe_level.quantile(0.01)
    p99 = roe_level.quantile(0.99)
    roe_level = roe_level.clip(lower=p01, upper=p99)

    df["osap_roe_level"] = roe_level

    # --- Change: ROE now vs ~4 quarters ago (252 trading days) ----------
    # Use shift on the filed-updated series (already PIT-safe; shift is backward)
    roe_lag = roe_level.shift(252)
    df["osap_roe_chg"] = roe_level - roe_lag

    # --- Rolling z-score (252-day window, min 63 obs) -------------------
    roll_mean = roe_level.rolling(252, min_periods=63).mean()
    roll_std = roe_level.rolling(252, min_periods=63).std()
    roll_std = roll_std.replace(0, np.nan)
    df["osap_roe_zscore"] = (roe_level - roll_mean) / roll_std

    # Drop scratch fund_ columns we are not listing in produces
    df = df.drop(columns=["fund_net_income_ttm", "fund_equity"], errors="ignore")

    return df
