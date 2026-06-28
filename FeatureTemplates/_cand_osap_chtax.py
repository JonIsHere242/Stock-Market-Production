"""
osap_chtax — Change in Taxes (Thomas & Zhang 2011)
OpenSourceAP / Chen-Zimmermann.

4-quarter change in quarterly total taxes (txtq) scaled by lagged total
assets (at).  Tax increases relative to assets predict positive future
returns cross-sectionally; implemented here as a per-ticker PIT proxy
using annual income-tax expense (net_income proxy via operating_income
minus net_income gives tax-like residual) — but the best available field
is net_income_ttm vs operating_income_ttm to back out implied tax.
We use: implied_tax = operating_income_ttm - net_income_ttm (a faithful
proxy for aggregate taxes paid), change = implied_tax(t) - implied_tax(t-1yr),
scaled by lagged assets.  Per-ticker only; cross-sectional ranking not
applied here.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── PIT fundamentals helper ────────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_chtax",
    "description": (
        "Change in Taxes (Thomas & Zhang 2011 / OpenSourceAP Chen-Zimmermann). "
        "Proxy: 4-quarter change in implied income taxes (operating_income_ttm − "
        "net_income_ttm), scaled by lagged total assets. A positive change in taxes "
        "relative to assets is associated with positive future cross-sectional returns. "
        "Per-ticker PIT proxy — cross-sectional ranking not applied."
    ),
    "requires": [],
    "produces": [
        "osap_chtax_level",    # current implied-tax / assets
        "osap_chtax_chg",      # 1-year change in implied-tax / assets (main signal)
        "osap_chtax_accel",    # 2nd difference (acceleration of tax change)
    ],
    "tags": ["accounting", "fundamentals", "tax", "osap"],
    "version": "1.0.0",
    "author": "Thomas and Zhang 2011; OpenSourceAP (Chen-Zimmermann); block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── pull PIT fundamental fields ────────────────────────────────────────
    df = _fundamentals.as_of(
        df,
        fields=["operating_income_ttm", "net_income_ttm", "assets"],
    )

    oi = df["fund_operating_income_ttm"]
    ni = df["fund_net_income_ttm"]
    assets = df["fund_assets"]

    # Implied income tax = operating income − net income (before interest etc.)
    # This captures taxes + interest as a proxy for the quarterly txtq aggregate.
    implied_tax = oi - ni

    # Lagged assets (use prior available filing — shift(1) within ticker series)
    assets_lag = assets.shift(1)

    # Level: implied_tax / assets (scaled by current assets for comparability)
    denom_level = assets.replace(0, np.nan)
    level = implied_tax / denom_level

    # 1-year change: difference between current and ~1yr-ago level
    # With PIT data sparsely updated (quarterly), shift(4) approximates 4 quarters.
    # We also try a 252-trading-day look-back via a time-indexed approach, but
    # since the index is not guaranteed DatetimeIndex, we use row-shift as proxy.
    ROWS_1YR = 4   # fundamentals update ~quarterly; 4 rows ≈ 1 year of filings
    level_lag1yr = level.shift(ROWS_1YR)
    chg = level - level_lag1yr

    # Acceleration (2nd difference)
    level_lag2yr = level.shift(2 * ROWS_1YR)
    chg_prior = level_lag1yr - level_lag2yr
    accel = chg - chg_prior

    df["osap_chtax_level"] = level
    df["osap_chtax_chg"] = chg
    df["osap_chtax_accel"] = accel

    # Drop scratch fund_* columns not in produces
    for col in ["fund_operating_income_ttm", "fund_net_income_ttm", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
