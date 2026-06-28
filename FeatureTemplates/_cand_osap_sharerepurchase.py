"""
osap_sharerepurchase — Share repurchase payout indicator.

Spec: OpenSourceAP (Chen-Zimmermann); original signal: Ikenberry, Lakonishok, Vermaelen 1995.
Original definition: binary 1 if prstkc > 0 (purchase of common stock in cash flow statement).

Per-ticker proxy: prstkc is not available in the PIT fundamentals catalog.
We approximate share-repurchase activity using two signals:
  1. osap_sharerepurchase_flag  — binary: shares_outstanding declined YoY (shrinkage = buyback proxy).
  2. osap_sharerepurchase_shrink — continuous YoY fractional decline in shares outstanding
       (positive = fewer shares, i.e. buyback occurred; negative = dilution).
  3. osap_sharerepurchase_trend — 2-period momentum of the shrinkage rate (acceleration).

When shares_outstanding is missing (ETFs, foreign), all outputs are NaN.
Cross-sectional ranking is NOT applied; this is a per-ticker signal.
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
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_sharerepurchase",
    "description": (
        "Per-ticker share-repurchase indicator based on YoY decline in shares outstanding "
        "(proxy for prstkc > 0 from the cash-flow statement). "
        "Original signal is cross-sectional binary (1 if prstkc > 0) from Ikenberry, "
        "Lakonishok & Vermaelen 1995 / OpenSourceAP (Chen-Zimmermann). "
        "Here we use PIT shares_outstanding: shrinkage = buyback proxy. "
        "osap_sharerepurchase_flag is the binary equivalent; "
        "osap_sharerepurchase_shrink is the continuous YoY rate; "
        "osap_sharerepurchase_trend is 2-period acceleration of that rate."
    ),
    "requires": [],
    "produces": [
        "osap_sharerepurchase_flag",
        "osap_sharerepurchase_shrink",
        "osap_sharerepurchase_trend",
    ],
    "tags": ["payout", "fundamentals", "repurchase", "shares_outstanding", "osap"],
    "version": "1.0",
    "author": "Ikenberry, Lakonishok, Vermaelen (1995); OpenSourceAP Chen-Zimmermann. Block by Claude.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add share-repurchase proxy columns to a single-ticker OHLCV frame."""

    # Pull PIT shares_outstanding; degrade gracefully if missing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _fundamentals.as_of(df, fields=["shares_outstanding"])
        has_fund = "fund_shares_outstanding" in df.columns
    except Exception:
        has_fund = False

    if not has_fund:
        df["osap_sharerepurchase_flag"] = np.nan
        df["osap_sharerepurchase_shrink"] = np.nan
        df["osap_sharerepurchase_trend"] = np.nan
        return df

    so = df["fund_shares_outstanding"].astype(float)

    # YoY shrinkage: how much fewer shares vs ~252 trading-days ago (1 year lag).
    # Using shift(252) on the PIT-merged column gives a backward-looking, leakage-free estimate.
    so_lag1 = so.shift(252)
    so_lag2 = so.shift(504)   # 2-year lag for trend

    # Continuous shrinkage rate: positive means fewer shares (buyback), negative = dilution
    # Guard against zero denominator
    denom1 = so_lag1.replace(0, np.nan)
    denom2 = so_lag2.replace(0, np.nan)

    shrink = (so_lag1 - so) / denom1          # positive = share count fell
    shrink_prev = (so_lag2 - so_lag1) / denom2  # prior-year shrinkage

    # Binary flag: 1 if share count declined (proxy for prstkc > 0)
    flag = (shrink > 0).astype(float)
    flag[shrink.isna()] = np.nan

    # 2-period trend: acceleration of shrinkage (positive = buyback activity increasing)
    trend = shrink - shrink_prev

    # Replace any inf that may have leaked through
    shrink = shrink.replace([np.inf, -np.inf], np.nan)
    trend = trend.replace([np.inf, -np.inf], np.nan)

    df["osap_sharerepurchase_flag"] = flag
    df["osap_sharerepurchase_shrink"] = shrink
    df["osap_sharerepurchase_trend"] = trend

    # Drop scratch fundamental column not in produces
    df = df.drop(columns=["fund_shares_outstanding"], errors="ignore")

    return df
