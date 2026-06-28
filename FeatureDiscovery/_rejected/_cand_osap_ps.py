"""
osap_ps — Sales-to-Price (Price-to-Sales inverse) value factor.

SOURCE: OpenSourceAP (Chen-Zimmermann); original reference: Barbee, Mukherji & Raines 1996
        "Do Sales–Price and Debt–Equity Explain Stock Returns Better than Book-Market
        and Firm Size?" FAJ.

ECONOMIC SIGNAL: High revenue relative to market-cap ("cheap on sales") predicts
positive future returns (value/growth mispricing). Implemented as S/P = revenue_ttm
per share / close price, i.e. the inverse of the popular P/S multiple.  A rising S/P
trend (momentum in cheapness) is also captured as a secondary feature.

NOTE: The factor is inherently cross-sectional in the original paper (NYSE universe
rank). Here it is computed as a per-ticker level so the model can rank cross-sectionally
at inference time. No cross-sectional ranking is applied inside this block.
"""

from __future__ import annotations
import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper (load by path, no package import)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)


METADATA: dict = {
    "name": "osap_ps",
    "description": (
        "Sales-to-Price (S/P) value factor from OpenSourceAP (Chen-Zimmermann). "
        "Measures TTM revenue per share relative to current share price. "
        "High S/P = cheap on sales = positive predicted return. "
        "Secondary feature: 252-day rolling z-score of S/P captures trend in "
        "relative valuation (cheapening momentum). "
        "Factor is per-ticker; cross-sectional ranking is done at inference time. "
        "Original paper: Barbee, Mukherji & Raines 1996 FAJ."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_ps_ratio",       # TTM revenue per share / close  (S/P level)
        "osap_ps_log",         # log(S/P)  — more stationary, used in original paper
        "osap_ps_zscore_252",  # 252-day rolling z-score of log(S/P) — trend/momentum
    ],
    "tags": ["valuation", "fundamental", "sales", "accounting", "osap"],
    "version": "1.0",
    "author": "osap_ps spec — OpenSourceAP (Chen-Zimmermann); Barbee, Mukherji & Raines 1996",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_ps_ratio, osap_ps_log, osap_ps_zscore_252 to df (one ticker, ascending)."""

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals: revenue_ttm + shares_outstanding
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["revenue_ttm", "sales_per_share"])

    # Prefer sales_per_share when available; fall back to revenue_ttm / shares_outstanding
    # sales_per_share = revenue_ttm / diluted shares (already per share from Compustat)
    # revenue_ttm is in millions of dollars in most providers — we use sales_per_share
    # which is already on a per-share basis, making it directly comparable to Close.

    sps = df["fund_sales_per_share"].copy()

    # If sales_per_share is missing but revenue_ttm is present we cannot divide without
    # shares_outstanding (not in our fundamentals list as a per-share helper), so we
    # rely solely on sales_per_share.  Where both are NaN the produced columns are NaN,
    # which is expected for ETFs / foreign names (~16% of universe).

    close = df["Close"]

    # ------------------------------------------------------------------
    # 2. S/P ratio  (sales per share / price)
    # ------------------------------------------------------------------
    # Guard: price must be positive; sps must be positive (negative sales unusual)
    price_safe = close.where(close > 0, np.nan)
    sps_safe = sps.where(sps > 0, np.nan)

    sp_ratio = sps_safe / price_safe          # S/P level
    sp_log = np.log(sp_ratio)                 # log(S/P) — more Gaussian, matches paper

    # ------------------------------------------------------------------
    # 3. Rolling 252-day z-score of log(S/P)
    # ------------------------------------------------------------------
    roll_mean = sp_log.rolling(252, min_periods=63).mean()
    roll_std = sp_log.rolling(252, min_periods=63).std()
    sp_zscore = (sp_log - roll_mean) / roll_std.where(roll_std > 0, np.nan)

    # ------------------------------------------------------------------
    # 4. Assign produced columns; drop scratch fund_ columns
    # ------------------------------------------------------------------
    df["osap_ps_ratio"] = sp_ratio
    df["osap_ps_log"] = sp_log
    df["osap_ps_zscore_252"] = sp_zscore

    # Drop fund_ scratch columns — not in produces
    for col in ["fund_revenue_ttm", "fund_sales_per_share"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
