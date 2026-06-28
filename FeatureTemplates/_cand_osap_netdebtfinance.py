"""
_cand_osap_netdebtfinance.py  --  Net debt financing anomaly (per-ticker PIT proxy)

Captures the external-financing / debt-issuance anomaly:
  companies that raise large amounts of net new debt tend to underperform.

ACADEMIC SIGNAL:
  Net Debt Financing (NDF) = (change in total debt) / avg total assets
  Predicted sign: NEGATIVE -- heavy debt issuers earn lower future returns
  (Bradshaw, Richardson & Sloan 2006; Spiess & Affleck-Graves 1999)

PER-TICKER PROXY (this block):
  Uses PIT SEC fundamentals (total_debt, assets) merged backward on filed_date.
  1. osap_netdebtfinance_lvl  -- NDF level: (total_debt_t - total_debt_t-1) / avg(assets)
     Negative expected predictive sign (high = more debt issued = lower fwd return).
  2. osap_netdebtfinance_accel -- acceleration: rolling 2-period change in NDF level
     (captures intensification of debt issuance).
  3. osap_netdebtfinance_ded -- debt-to-equity deviation: current D/E vs its own
     trailing 4-quarter rolling mean, normalized; captures departure from leverage norm.

Cross-sectional ranking is NOT implemented here (requires multiple tickers simultaneously).
All signals are per-ticker time-series only.
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
    "name": "osap_netdebtfinance",
    "description": (
        "Net debt financing anomaly (per-ticker PIT proxy). "
        "Measures year-over-year change in total debt scaled by average total assets, "
        "plus an acceleration variant and a debt-to-equity deviation from rolling norm. "
        "Negative predicted sign: heavy net debt issuance -> lower forward returns. "
        "Cross-sectional ranking not implemented; per-ticker time-series proxy only. "
        "Source: OpenSourceAP (Chen-Zimmermann), Bradshaw/Richardson/Sloan 2006, "
        "Spiess/Affleck-Graves 1999."
    ),
    "requires": [],  # fundamentals loaded internally; no raw OHLCV columns needed
    "produces": [
        "osap_netdebtfinance_lvl",
        "osap_netdebtfinance_accel",
        "osap_netdebtfinance_ded",
    ],
    "tags": ["fundamental", "financing", "leverage", "debt", "osap"],
    "version": "1.0.0",
    "author": "OpenSourceAP (Chen-Zimmermann); Bradshaw, Richardson & Sloan 2006; Spiess & Affleck-Graves 1999",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add net debt financing features to a single-ticker OHLCV frame."""

    n = len(df)

    # Default: all NaN
    df["osap_netdebtfinance_lvl"] = np.nan
    df["osap_netdebtfinance_accel"] = np.nan
    df["osap_netdebtfinance_ded"] = np.nan

    if n == 0:
        return df

    # -----------------------------------------------------------------------
    # Load PIT fundamentals
    # -----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = _fundamentals.as_of(df, fields=["total_debt", "assets", "debt_to_equity"])
        except Exception:
            # No coverage for this ticker; leave columns NaN
            return df

    # -----------------------------------------------------------------------
    # Feature 1: NDF level
    #   NDF_t = (total_debt_t - total_debt_{t-1}) / ((assets_t + assets_{t-1}) / 2)
    #
    # fundamentals are PIT-stamped: the same filing value is repeated until the next filing.
    # We use .diff() on the filed-date-merged series -- consecutive rows with the same
    # filing value will give diff=0 (no new debt), which is correct (no filing event).
    # An actual new filing will produce a nonzero diff only when a new filing is merged in.
    # -----------------------------------------------------------------------
    td = df["fund_total_debt"]
    at = df["fund_assets"]

    td_prev = td.shift(1)
    at_prev = at.shift(1)

    avg_assets = (at + at_prev) / 2.0
    # Guard: avg_assets == 0 -> NaN
    avg_assets_safe = avg_assets.where(avg_assets != 0, np.nan)

    ndf_lvl = (td - td_prev) / avg_assets_safe
    # Clip extremes to suppress outlier noise from data gaps
    ndf_lvl = ndf_lvl.clip(lower=-5.0, upper=5.0)
    df["osap_netdebtfinance_lvl"] = ndf_lvl

    # -----------------------------------------------------------------------
    # Feature 2: NDF acceleration
    #   How much faster/slower is the company taking on debt vs the prior period?
    #   Rolling 2-period change in NDF level (diff of diff).
    #   We use a 3-row rolling window to get a stable finite-difference.
    # -----------------------------------------------------------------------
    ndf_accel = ndf_lvl.diff(1)
    ndf_accel = ndf_accel.clip(lower=-5.0, upper=5.0)
    df["osap_netdebtfinance_accel"] = ndf_accel

    # -----------------------------------------------------------------------
    # Feature 3: Debt-to-equity deviation from rolling norm
    #   de = fund_debt_to_equity
    #   rolling mean and std over a 252-day trailing window
    #   ded = (de - rolling_mean) / rolling_std   [z-score vs own history]
    #   Captures leverage departures regardless of absolute level.
    # -----------------------------------------------------------------------
    de = df["fund_debt_to_equity"]
    roll_window = 252

    # Only compute where we have enough history; pandas rolling handles NaN gracefully
    de_roll_mean = de.rolling(window=roll_window, min_periods=60).mean()
    de_roll_std = de.rolling(window=roll_window, min_periods=60).std()

    # Guard zero std
    de_roll_std_safe = de_roll_std.where(de_roll_std > 0, np.nan)
    ded = (de - de_roll_mean) / de_roll_std_safe
    ded = ded.clip(lower=-5.0, upper=5.0)
    df["osap_netdebtfinance_ded"] = ded

    # -----------------------------------------------------------------------
    # Drop scratch fund_ columns that are NOT in produces
    # -----------------------------------------------------------------------
    for col in ["fund_total_debt", "fund_assets", "fund_debt_to_equity"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
