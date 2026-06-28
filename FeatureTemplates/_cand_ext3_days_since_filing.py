"""
Post-filing drift window (days since last filing) — ext3_days_since_filing

Detects new SEC filings as a change in PIT fundamentals (net_income_ttm,
revenue_ttm, assets) and produces a PEAD-style drift clock:
  - ext3_days_since_filing_days   : calendar days since the last detected filing
  - ext3_days_since_filing_norm   : days_since / 90.0 (normalised, bounded 0-1)
  - ext3_days_since_filing_drift  : 5-day return * I(days_since in [1,60])
                                    (captures post-earnings announcement drift window)

Coverage ~84 % (ETFs / foreign -> NaN). Per-ticker proxy; causal / no-lookahead.
SOURCE: Round-4 expansion (osap_orgcap), extends osap_orgcap signal axis.
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
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_days_since_filing",
    "description": (
        "Post-filing drift window (PEAD drift clock). Detects new SEC filings via "
        "changes in PIT net_income_ttm, revenue_ttm, or assets from "
        "_fundamentals.as_of (filed_date, lookahead-safe). Produces "
        "(1) calendar days since last detected filing change, "
        "(2) that value normalised by 90 days, and "
        "(3) 5-day return gated to the 1-60 day post-filing window as a "
        "per-ticker PEAD (post-earnings announcement drift) proxy. "
        "Coverage ~84%; ETFs and foreign tickers produce NaN. "
        "Per-ticker, no cross-sectional look; causal, no lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_days_since_filing_days",
        "ext3_days_since_filing_norm",
        "ext3_days_since_filing_drift",
    ],
    "tags": ["fundamentals", "pead", "filing", "event", "drift"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_orgcap); spec by project team",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add PEAD drift-clock features derived from PIT fundamentals filing changes."""

    nan_col = pd.Series(np.nan, index=df.index)

    # Attach PIT fundamentals (backward merge on filed_date — no lookahead)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = _fundamentals.as_of(
                df, fields=["net_income_ttm", "revenue_ttm", "assets"]
            )
    except Exception:
        # Fundamentals unavailable — emit NaN columns and return
        df["ext3_days_since_filing_days"] = nan_col
        df["ext3_days_since_filing_norm"] = nan_col
        df["ext3_days_since_filing_drift"] = nan_col
        return df

    fund_cols = ["fund_net_income_ttm", "fund_revenue_ttm", "fund_assets"]
    present = [c for c in fund_cols if c in df.columns]

    if not present:
        df["ext3_days_since_filing_days"] = nan_col
        df["ext3_days_since_filing_norm"] = nan_col
        df["ext3_days_since_filing_drift"] = nan_col
        # drop any fund_ scratch columns that were added
        for c in fund_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        return df

    # -----------------------------------------------------------------------
    # Detect filing events: any change in the merged fundamental values
    # A "change" = the backward-merged value differs from the previous row's
    # merged value. Because as_of does a backward merge on filed_date, each
    # change in these columns corresponds to a new public filing.
    # -----------------------------------------------------------------------
    # Build a boolean series: True on rows where any of the fund cols changed
    changed = pd.Series(False, index=df.index)
    for col in present:
        series = df[col]
        # .diff() on the fund value: non-zero (and not first-bar NaN) = new filing
        delta = series.diff()
        # Also flag the very first non-NaN row per ticker as a "new" observation
        first_valid = series.first_valid_index()
        is_new = delta.abs() > 0
        if first_valid is not None:
            is_new.loc[first_valid] = True
        changed = changed | is_new

    # -----------------------------------------------------------------------
    # days_since_filing: calendar days elapsed since the last detected change
    # Use Date column (already in df as object or datetime-like)
    # -----------------------------------------------------------------------
    try:
        dates = pd.to_datetime(df["Date"])
    except Exception:
        dates = df.index

    # Build days_since array
    days_since = np.full(len(df), np.nan)
    last_filing_date = None

    for i in range(len(df)):
        if changed.iloc[i]:
            last_filing_date = dates.iloc[i]
        if last_filing_date is not None:
            delta_days = (dates.iloc[i] - last_filing_date).days
            days_since[i] = float(delta_days)

    days_since_s = pd.Series(days_since, index=df.index)

    # -----------------------------------------------------------------------
    # Feature 1: raw days since last filing
    # -----------------------------------------------------------------------
    df["ext3_days_since_filing_days"] = days_since_s

    # -----------------------------------------------------------------------
    # Feature 2: normalised (0-1 scale over a 90-day window)
    # -----------------------------------------------------------------------
    df["ext3_days_since_filing_norm"] = (days_since_s / 90.0).clip(upper=1.0)

    # -----------------------------------------------------------------------
    # Feature 3: PEAD drift proxy
    #   5-day forward return (using past 5 days' log return, causal)
    #   gated to window [1, 60] days post-filing.
    # Note: we use the 5-day LAGGED return ending at t (past), not future.
    # This is causal: ret_5d[t] = log(Close[t] / Close[t-5]).
    # Within the PEAD window, this captures momentum that has already
    # materialised — a measure of how much drift has occurred so far.
    # -----------------------------------------------------------------------
    log_ret_5d = np.log(
        df["Close"] / df["Close"].shift(5).replace(0, np.nan)
    )

    in_pead_window = (days_since_s >= 1) & (days_since_s <= 60)
    drift = log_ret_5d.where(in_pead_window, other=np.nan)
    df["ext3_days_since_filing_drift"] = drift

    # -----------------------------------------------------------------------
    # Drop scratch fund_ columns (not listed in produces)
    # -----------------------------------------------------------------------
    for c in fund_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df
