"""
Candidate block: osap_realestate
Spec: OpenSourceAP (Chen-Zimmermann) / Tuzel 2010
Signal: Real estate holdings (asset composition factor, predicted sign +1 long high).

Original method: (fatb+fatl)/ppegt or (ppenb+ppenl)/ppent, industry-adjusted at 2-digit SIC.
Per-ticker proxy: ppe_net / assets as an approximation of fixed/real-asset intensity,
since fine-grained Compustat components (fatb, fatl, ppegt, ppenb, ppenl) are not
available in the fundamentals helper. The SIC industry-mean adjustment is inherently
cross-sectional and therefore implemented here as a rolling per-ticker deviation from
a 4-quarter trailing own-history median (captures changes in the firm's own real
estate intensity over time). This is a PROXY -- it will not replicate the original
cross-sectional sort exactly.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_realestate",
    "description": (
        "Per-ticker proxy for Tuzel (2010) real-estate holdings factor. "
        "Original: (fatb+fatl)/ppegt [or (ppenb+ppenl)/ppent], industry-adjusted at 2-digit SIC. "
        "Proxy here uses ppe_net/assets (net PP&E intensity) from PIT fundamentals "
        "because granular Compustat components are unavailable. "
        "Industry adjustment is replaced by rolling 4-quarter own-median deviation "
        "since cross-sectional SIC means require multi-stock data. "
        "Produces: level ratio, rolling deviation from own 4Q median, "
        "and 2-quarter change (momentum of real-asset intensity)."
    ),
    "requires": [],
    "produces": [
        "osap_realestate_ratio",       # ppe_net / assets  (level)
        "osap_realestate_dev",         # deviation from own 4Q rolling median
        "osap_realestate_chg2q",       # 2-quarter change in ratio
    ],
    "tags": ["fundamentals", "asset_composition", "real_estate", "osap"],
    "version": "1.0",
    "author": "Tuzel 2010 / OpenSourceAP Chen-Zimmermann; per-ticker OHLCV+fundamentals proxy by codegen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real-estate-intensity proxy using PIT fundamentals.
    All produced columns are NaN for rows without fundamentals coverage (~16% of universe).
    """
    # Pull PIT fundamentals: net PP&E and total assets
    df = _fundamentals.as_of(df, fields=["ppe_net", "assets"])

    ppe = df["fund_ppe_net"]
    assets = df["fund_assets"]

    # Level ratio: net PP&E / total assets  (guard div-by-zero)
    ratio = np.where(
        (assets.notna()) & (ppe.notna()) & (assets != 0),
        ppe.values / assets.values,
        np.nan,
    )
    ratio = pd.Series(ratio, index=df.index)
    # Clamp to [0, 1] -- ratio above 1 or below 0 indicates data anomaly
    ratio = ratio.clip(0.0, 1.0)
    df["osap_realestate_ratio"] = ratio

    # Rolling deviation from own 4-quarter trailing median
    # Fundamentals update ~quarterly; 63 trading days ~ 1 quarter; 252 ~ 4 quarters
    WINDOW_4Q = 252
    rolling_med = ratio.rolling(window=WINDOW_4Q, min_periods=63).median()
    df["osap_realestate_dev"] = ratio - rolling_med

    # 2-quarter change: ratio(t) - ratio(t - 126 trading days)
    WINDOW_2Q = 126
    df["osap_realestate_chg2q"] = ratio - ratio.shift(WINDOW_2Q)

    # Drop scratch fund_* columns not listed in produces
    df.drop(columns=["fund_ppe_net", "fund_assets"], inplace=True)

    return df
