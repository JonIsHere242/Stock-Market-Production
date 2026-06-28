"""
Composite Debt Issuance (osap_compositedebtissuance)

Per-ticker PIT-fundamentals implementation of Lyandres, Sun and Zhang (2008)
composite debt issuance factor. The original is cross-sectional; this block
computes the same quantity per-ticker using point-in-time SEC fundamentals.

Signal: log(total_debt_t) - log(total_debt_{t-5yr})
  Negative predicted sign: firms that issued more debt underperform.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT-fundamentals helper (loaded by file path, never from FeatureTemplates)
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_compositedebtissuance",
    "description": (
        "Composite debt issuance (Lyandres, Sun and Zhang 2008 via "
        "OpenSourceAP / Chen-Zimmermann). "
        "Original: log(dltt+dlc)_t - log(dltt+dlc)_{t-5yr} (cross-sectional). "
        "Per-ticker PIT proxy: uses `total_debt` from SEC fundamentals "
        "(backward-merged filed_date, no lookahead). "
        "Level = log-debt-change over ~5yr filing horizon; "
        "slope = 1yr rate-of-change of log total_debt; "
        "accel = difference of two consecutive 1yr slopes (acceleration). "
        "Predicted sign: -1 (heavy issuers underperform)."
    ),
    "requires": [],  # uses PIT fundamentals, not OHLCV directly
    "produces": [
        "osap_compositedebtissuance_5yr",   # main: log debt change over 5 yrs
        "osap_compositedebtissuance_1yr",   # slope: log debt change over 1 yr
        "osap_compositedebtissuance_accel", # accel: 2nd derivative of log debt
    ],
    "tags": ["fundamentals", "external_financing", "debt", "issuance", "osap"],
    "version": "1.0.0",
    "author": "Lyandres, Sun and Zhang 2008; OpenSourceAP (Chen-Zimmermann); impl. by Claude",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach composite debt issuance features to a single-ticker DataFrame.

    Steps:
      1. Pull PIT `total_debt` from SEC fundamentals via as_of().
      2. Compute log(total_debt); guard zero/negative values.
      3. 5-yr change  = log_debt_t - log_debt_{t-5yr}  (using shift on
         the merged series, aligned by filing cadence via merge_asof).
      4. 1-yr slope   = log_debt_t - log_debt_{t-1yr}.
      5. Acceleration = 1yr_slope_t - 1yr_slope_{t-1yr}.
    """
    # -- 1. Pull fundamentals -------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["total_debt"])

    col = "fund_total_debt"

    # Guard: must have the column (coverage ~84%; ETFs/foreign return NaN)
    if col not in df.columns:
        df["osap_compositedebtissuance_5yr"] = np.nan
        df["osap_compositedebtissuance_1yr"] = np.nan
        df["osap_compositedebtissuance_accel"] = np.nan
        return df

    debt = df[col].copy()

    # -- 2. Log of total_debt (guard non-positive) ----------------------------
    debt_pos = debt.where(debt > 0, other=np.nan)
    log_debt = np.log(debt_pos)

    # -- 3. 5-year change (approx 252*5 = 1260 trading days) -----------------
    # The SEC data is filed quarterly/annually; shifts of N rows correspond to
    # N filings-worth of data already merged onto trading days via merge_asof.
    # We shift by calendar-approximate trading days to stay lookahead-safe.
    # 252 * 5 = 1260 rows back for ~5-year window.
    DAYS_1YR = 252
    DAYS_5YR = 252 * 5

    log_debt_5yr_ago = log_debt.shift(DAYS_5YR)
    log_debt_1yr_ago = log_debt.shift(DAYS_1YR)

    df["osap_compositedebtissuance_5yr"] = log_debt - log_debt_5yr_ago

    slope_now = log_debt - log_debt_1yr_ago
    df["osap_compositedebtissuance_1yr"] = slope_now

    slope_1yr_ago = slope_now.shift(DAYS_1YR)
    df["osap_compositedebtissuance_accel"] = slope_now - slope_1yr_ago

    # -- 4. Drop scratch fundamentals column ----------------------------------
    df = df.drop(columns=[col])

    return df
