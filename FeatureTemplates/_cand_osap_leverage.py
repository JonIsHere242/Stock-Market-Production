"""
Candidate feature block: osap_leverage
Market Leverage = Total Liabilities / Market Value of Equity (Bhandari 1988).
Per-ticker proxy: uses PIT fundamentals (liabilities, shares_outstanding) merged
backward on filed_date; market cap computed as Close * fund_shares_outstanding.
Cross-sectional ranking is inherently impossible per-ticker, so this captures
the absolute leverage level and its recent change as the economic signal.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper (by file path -- do NOT import as package)
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
    "name": "osap_leverage",
    "description": (
        "Market leverage: total liabilities divided by market value of equity "
        "(Close * shares_outstanding), using point-in-time SEC fundamentals so "
        "there is no lookahead. Produces: (1) osap_leverage_ratio -- the raw "
        "leverage ratio; (2) osap_leverage_chg -- quarter-over-quarter change in "
        "the ratio (rolling 63-day diff, ~1 fiscal quarter) as a momentum/trend "
        "signal; (3) osap_leverage_log -- log(1 + ratio) to compress right tail. "
        "Per-ticker proxy: cross-sectional ranking not available in this context; "
        "the level captures the same Bhandari (1988) economic signal. "
        "Coverage ~84% (ETFs/foreign will be NaN)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_leverage_ratio",
        "osap_leverage_chg",
        "osap_leverage_log",
    ],
    "tags": ["leverage", "fundamentals", "accounting", "osap"],
    "version": "1.0",
    "author": "Bhandari (1988) via OpenSourceAP / Chen-Zimmermann",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals -- backward merge on filed_date (no lookahead)
    df = _fundamentals.as_of(df, fields=["liabilities", "shares_outstanding"])

    # Market value of equity: price * shares (shares in actual count, not millions)
    mve = df["Close"] * df["fund_shares_outstanding"]
    # Guard: replace zero/negative MVE with NaN
    mve = mve.where(mve > 0, np.nan)

    total_liab = df["fund_liabilities"]
    # Some sources store liabilities in thousands; as_of returns raw EDGAR units
    # (USD). No unit conversion needed -- ratio is dimensionless.

    # (1) Raw leverage ratio
    ratio = total_liab / mve
    # Clip extreme outliers to avoid infs from near-zero float MVE
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_leverage_ratio"] = ratio

    # (2) Rolling 63-day (approx 1 quarter) change in ratio -- captures
    #     whether leverage is rising or falling (momentum signal)
    df["osap_leverage_chg"] = ratio.diff(63)

    # (3) Log-compressed ratio: log(1 + max(ratio, 0)) -- handles NaN gracefully
    pos_ratio = ratio.clip(lower=0)
    df["osap_leverage_log"] = np.log1p(pos_ratio)

    # Drop scratch fundamentals columns we are NOT producing
    df = df.drop(columns=["fund_liabilities", "fund_shares_outstanding"], errors="ignore")

    return df
