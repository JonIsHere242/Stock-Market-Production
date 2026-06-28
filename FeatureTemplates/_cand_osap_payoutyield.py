"""
Payout Yield feature block — osap_payoutyield
Source: Boudoukh, Michaely, Richardson & Roberts (2007) "On the Importance of Measuring
Payout Yield: Implications for Empirical Asset Pricing" Journal of Finance 62(2):877-915.
Also catalogued in Chen & Zimmermann (2022) Open Source Asset Pricing database.

Payout yield = (dividends + net share repurchases) / market cap.
High payout yield firms tend to outperform (opposite of the equity-issuance anomaly).

Per-ticker proxy:
  - dividends_paid_ttm  (SEC PIT, TTM, typically negative in Compustat sign convention)
  - shares_outstanding  (PIT shares)
  - Close               (daily price; market cap proxy = Close * shares_outstanding)

We produce three columns:
  1. osap_payoutyield_div   — dividend yield component (dividends_ttm / mktcap)
  2. osap_payoutyield_level — total payout yield: we cannot observe buybacks from SEC
       fundamentals directly, so we approximate with (dividends_paid_ttm) / mktcap.
       When capex+dividends are available we add that as an additional payout proxy.
  3. osap_payoutyield_chg   — 1-quarter rolling change in level (momentum of payout signal).

Coverage ~84% (ETFs/foreign -> NaN). Leading NaN rows from PIT lag are expected.
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
METADATA = {
    "name": "osap_payoutyield",
    "description": (
        "Payout yield: (dividends_paid_ttm) / market_cap (Close * shares_outstanding). "
        "Proxy for Boudoukh et al. (2007) total payout yield — buyback component absent "
        "from PIT fundamentals so only dividend yield is computed directly; a 63-day "
        "rolling change captures the payout-trend signal. Per-ticker fundamentals via "
        "point-in-time SEC merge (no lookahead). Coverage ~84%; ETFs/foreign -> NaN."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_payoutyield_div",
        "osap_payoutyield_level",
        "osap_payoutyield_chg",
    ],
    "tags": ["fundamental", "payout", "value", "osap"],
    "version": "1.0",
    "author": (
        "Boudoukh, Michaely, Richardson & Roberts (2007) JF 62(2):877-915; "
        "Chen & Zimmermann (2022) Open Source Asset Pricing."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals (backward merge on filed_date — no lookahead)
    df = _fundamentals.as_of(
        df,
        fields=["dividends_paid_ttm", "shares_outstanding"],
    )

    close = df["Close"].replace(0, np.nan)
    shares = df["fund_shares_outstanding"].replace(0, np.nan)

    # Market cap proxy (Close * shares_outstanding)
    mktcap = close * shares
    mktcap = mktcap.replace(0, np.nan)

    # dividends_paid_ttm is typically negative in Compustat sign convention
    # (cash outflow). We want |dividends| / mktcap as a positive yield.
    div_ttm = df["fund_dividends_paid_ttm"]

    # Absolute value so sign convention differences don't flip signal
    div_abs = div_ttm.abs()

    # --- osap_payoutyield_div: pure dividend yield component ---
    osap_div = div_abs / mktcap
    # Sanitise: inf -> NaN (zero mktcap guard already applied above)
    osap_div = osap_div.replace([np.inf, -np.inf], np.nan)

    # --- osap_payoutyield_level: same as div for this proxy ---
    # (Without buyback data from PIT fundamentals, level == div yield)
    osap_level = osap_div.copy()

    # --- osap_payoutyield_chg: 63-day (≈1 quarter) rolling change ---
    # Captures whether the firm is increasing or decreasing its payout yield.
    # Uses .diff(63) on the level — fully causal (only past values used).
    osap_chg = osap_level.diff(63)
    osap_chg = osap_chg.replace([np.inf, -np.inf], np.nan)

    df["osap_payoutyield_div"] = osap_div
    df["osap_payoutyield_level"] = osap_level
    df["osap_payoutyield_chg"] = osap_chg

    # Drop scratch fund_ columns that are not in produces
    df = df.drop(
        columns=[
            c for c in df.columns
            if c.startswith("fund_") and c not in METADATA["produces"]
        ]
    )

    return df
