"""
osap_am — Total Assets to Market
Source: OpenSourceAP (Chen-Zimmermann); Fama and French (1992)

Per-ticker implementation using PIT SEC fundamentals (as_of) + daily Close×shares_outstanding
as market-cap proxy. Predicted sign: +1 (long high asset/market ratio = value tilt).
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
    "name": "osap_am",
    "description": (
        "Total Assets to Market (A/M): PIT total assets divided by market value of equity "
        "(Close × shares_outstanding). A value/leverage signal: high A/M firms are asset-heavy "
        "relative to their market cap. Predicted sign +1 (long high). "
        "Per-ticker — not cross-sectional; level and 6-month z-score variants produced. "
        "Source: OpenSourceAP (Chen-Zimmermann), Fama & French 1992."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_am_ratio",      # Total assets / market cap (PIT)
        "osap_am_log",        # log(1 + ratio) — compresses right skew
        "osap_am_z126",       # 126-day rolling z-score of the log ratio (trend)
    ],
    "tags": ["valuation", "fundamentals", "accounting", "fama-french", "asset-intensity"],
    "version": "1.0.0",
    "author": "Fama and French (1992) via OpenSourceAP (Chen-Zimmermann)",
}


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach osap_am_ratio, osap_am_log, osap_am_z126 to df.

    Market cap proxy = Close × fund_shares_outstanding (PIT from SEC).
    If shares data is unavailable (ETFs / foreign), all three columns are NaN.
    """
    # --- pull PIT fundamentals (assets + shares) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["assets", "shares_outstanding"])

    # fund_assets: total assets in USD (same units as shares × price => need to
    # ensure consistent scale; assets from SEC are typically in thousands or units
    # depending on filer -- as_of returns the raw value from companyfacts)
    assets = df["fund_assets"]                      # PIT total assets
    shares = df["fund_shares_outstanding"]          # PIT diluted shares outstanding

    # Market cap: price × shares (keep in same scale as assets)
    # SEC assets are in USD (not thousands for most modern filers via companyfacts)
    mktcap = df["Close"] * shares
    mktcap = mktcap.replace(0, np.nan)

    # Core ratio: assets / market_cap
    ratio = assets / mktcap
    # Guard: inf can appear if mktcap is 0 (already replaced), but belt+suspenders
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    df["osap_am_ratio"] = ratio

    # Log-transform to compress right skew (Fama-French common practice)
    log_ratio = np.log1p(ratio.clip(lower=0))
    log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan)
    df["osap_am_log"] = log_ratio

    # 126-day rolling z-score of log ratio (captures drift / re-rating signal)
    roll_mean = log_ratio.rolling(126, min_periods=63).mean()
    roll_std  = log_ratio.rolling(126, min_periods=63).std(ddof=1)
    roll_std  = roll_std.replace(0, np.nan)
    df["osap_am_z126"] = (log_ratio - roll_mean) / roll_std

    # Drop scratch fund_* columns not in produces
    df = df.drop(columns=["fund_assets", "fund_shares_outstanding"], errors="ignore")

    return df
