"""
osap_shareiss5y — Share Issuance (5-year)
Per-ticker proxy for Daniel & Titman (2006) 5-year share issuance anomaly.

Original: 5-year growth in split-adjusted shares outstanding (shrout/cfacshr).
Negative predicted sign: high issuance -> negative forward returns.

Implementation notes:
- Uses PIT fundamentals `shares_outstanding` (backward-safe via filed_date).
- Computes trailing 5-year (≈252*5=1260 trading-day) log-growth in shares.
- Degraded gracefully: if fundamentals unavailable (ETFs, foreign), produces NaN.
- Inherently per-ticker; no cross-sectional ranking here.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_shareiss5y",
    "description": (
        "5-year share issuance signal (Daniel & Titman 2006 / OpenSourceAP Chen-Zimmermann). "
        "Measures 5-year log-growth in shares outstanding from PIT SEC fundamentals. "
        "Predicted sign: -1 (heavy issuance forecasts negative returns). "
        "Per-ticker proxy — no cross-sectional ranking. "
        "Three columns: level log-growth over 5yr, a 1-yr log-change (short issuance pulse), "
        "and a recency ratio (5yr growth minus 1yr growth, capturing the multi-year trend "
        "excluding the most recent year)."
    ),
    "requires": [],
    "produces": [
        "osap_shareiss5y_5y",      # 5-year log-growth in shares outstanding (primary signal)
        "osap_shareiss5y_1y",      # 1-year log-growth in shares outstanding
        "osap_shareiss5y_lt",      # long-term issuance: 5y minus 1y (4-yr window excl. latest yr)
    ],
    "tags": ["external_financing", "issuance", "fundamentals", "accounting", "daniel_titman"],
    "version": "1.0",
    "author": "Daniel and Titman (2006) via OpenSourceAP (Chen & Zimmermann); block by Claude",
}

# Approximate trading days for windows
_DAYS_1Y = 252
_DAYS_5Y = 252 * 5   # 1260


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-year share issuance features using PIT fundamentals.

    Parameters
    ----------
    df : pd.DataFrame
        Single-ticker OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Input frame with three new columns appended.
    """
    # Initialise output columns as NaN
    df = df.copy()
    df["osap_shareiss5y_5y"] = np.nan
    df["osap_shareiss5y_1y"] = np.nan
    df["osap_shareiss5y_lt"] = np.nan

    # Pull PIT shares_outstanding
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = _fundamentals.as_of(df, fields=["shares_outstanding"])
        except Exception:
            return df

    so_col = "fund_shares_outstanding"
    if so_col not in df.columns:
        return df

    # Convert to float, guard against zeros
    shares = df[so_col].astype(float).replace(0, np.nan)

    # Log of shares outstanding
    log_shares = np.log(shares)

    # Rolling windows: trailing 5y and 1y
    # We want log(shares_t) - log(shares_{t-N}) using the most-recent non-NaN
    # approach: rolling min_periods=1 keeps intermediate values but we need
    # the value N trading days ago. Use shift-based approach on the
    # forward-filled fundamentals series to capture the PIT view.
    # Forward-fill within the compute window is safe here because fundamentals
    # are already PIT-stamped by _fundamentals.as_of (filed_date backward merge).
    log_shares_ffill = log_shares.ffill()

    shares_5y_ago = log_shares_ffill.shift(_DAYS_5Y)
    shares_1y_ago = log_shares_ffill.shift(_DAYS_1Y)

    current_log = log_shares_ffill

    growth_5y = current_log - shares_5y_ago   # log-growth over 5 years
    growth_1y = current_log - shares_1y_ago   # log-growth over 1 year
    growth_lt = growth_5y - growth_1y          # 4-year portion excl. latest year

    # Replace inf/-inf with NaN
    df["osap_shareiss5y_5y"] = growth_5y.replace([np.inf, -np.inf], np.nan)
    df["osap_shareiss5y_1y"] = growth_1y.replace([np.inf, -np.inf], np.nan)
    df["osap_shareiss5y_lt"] = growth_lt.replace([np.inf, -np.inf], np.nan)

    # Drop scratch fundamental column
    df.drop(columns=[so_col], inplace=True)

    return df
