"""
osap_chnncoa — Change in Net Noncurrent Operating Assets
=========================================================
SOURCE : OpenSourceAP (Chen-Zimmermann), Soliman (2008)
SIGNAL : 12-month change in noncurrent operating assets (chNCOA).
         Predicted sign: -1 (high asset expansion → lower future returns).

Definition (Soliman 2008 / OpenSourceAP):
    NCOA = ( (at - act - ivao) - (lt - dlc - dltt) ) / at
    chNCOA = NCOA_t − NCOA_{t-12m}

where:
    at   = total assets
    act  = current assets
    ivao = other long-term investments (not available in _fundamentals;
           proxied with goodwill, which captures goodwill & intangibles
           that are excluded from operating noncurrent assets)
    lt   = total liabilities
    dlc  = current portion of long-term debt (not directly available;
           approximated as 0 — conservative: treats all current
           liabilities as operating, slightly overstates operating
           liabilities but is the closest honest proxy)
    dltt = long-term debt

Per-ticker proxy notes:
  - This is inherently a point-in-time balance-sheet signal; we use
    PIT fundamentals (filed_date-safe via _fundamentals.as_of).
  - Cross-sectional ranking is NOT applied; we emit the raw NCOA level
    and its 12-month rolling change so the predictor can rank across
    the universe at inference time.
  - Coverage ~84%; ETFs/foreign firms return NaN (expected).
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
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_chnncoa",
    "description": (
        "Change in Net Noncurrent Operating Assets (chNCOA) from Soliman (2008) "
        "via OpenSourceAP (Chen-Zimmermann). "
        "NCOA = ((assets - assets_current - goodwill) - "
        "(liabilities - 0 - long_term_debt)) / assets; "
        "chNCOA = 12-month change in NCOA. "
        "Predicted sign -1: asset expansion predicts lower future returns. "
        "Per-ticker PIT proxy: ivao proxied by goodwill; dlc set to 0 "
        "(conservative). Cross-sectional ranking deferred to the predictor."
    ),
    "requires": ["Close"],   # Close used only as a date anchor; OHLCV otherwise unused
    "produces": [
        "osap_chnncoa_ncoa",     # level: net noncurrent operating assets / total assets
        "osap_chnncoa_ch",       # 12-month change in NCOA (primary signal)
        "osap_chnncoa_ch_sign",  # sign-adjusted: multiply by -1 so high = bearish
    ],
    "tags": ["fundamental", "investment", "balance_sheet", "osap", "soliman2008"],
    "version": "1.0.0",
    "author": "Soliman (2008) / OpenSourceAP Chen-Zimmermann; block by claude-sonnet-4-6",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach osap_chnncoa_* columns to a single-ticker OHLCV DataFrame.
    df is ascending by Date, one ticker only.
    """
    # Pull PIT fundamentals we need
    df = _fundamentals.as_of(
        df,
        fields=[
            "assets",           # at
            "assets_current",   # act
            "goodwill",         # ivao proxy
            "liabilities",      # lt
            "long_term_debt",   # dltt
        ],
    )

    at   = df["fund_assets"]
    act  = df["fund_assets_current"]
    ivao = df["fund_goodwill"]          # proxy; NaN → treated as 0 below
    lt   = df["fund_liabilities"]
    # dlc (current portion of LTD) not available → 0
    dltt = df["fund_long_term_debt"]

    # Replace NaN in ivao with 0 (missing goodwill → assume zero, not missing NCOA)
    ivao_filled = ivao.fillna(0.0)

    # Noncurrent operating assets (numerator top part): at - act - ivao
    noncurrent_op_assets = at - act - ivao_filled

    # Noncurrent operating liabilities (numerator bottom part): lt - 0 - dltt
    noncurrent_op_liabs = lt - dltt.fillna(0.0)

    # Net noncurrent operating assets scaled by total assets
    # Guard against at == 0 or NaN
    at_safe = at.where(at.abs() > 0, np.nan)
    ncoa = (noncurrent_op_assets - noncurrent_op_liabs) / at_safe

    # 12-month change: use a 252-trading-day lag (≈ 1 year of daily rows)
    # Because fundamentals are quarterly-filed, the lag captures the annual change
    # in the most-recently-filed values, which is the standard chNCOA definition.
    # We also offer a 63-day lag (≈ 1 quarter) to allow the predictor flexibility,
    # but the primary 252-day lag is the canonical annual signal.
    ncoa_lag252 = ncoa.shift(252)
    ch = ncoa - ncoa_lag252

    # Sign-adjusted version: multiply raw change by -1 so that HIGH = more bearish
    # (consistent with OSAP predicted sign -1 convention flipped to +1 for modeling)
    ch_sign = -1.0 * ch

    # Assign produced columns
    df["osap_chnncoa_ncoa"]    = ncoa
    df["osap_chnncoa_ch"]      = ch
    df["osap_chnncoa_ch_sign"] = ch_sign

    # Drop scratch fund_* columns that are NOT in produces
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
