"""
osap_rd: R&D expense over market cap (Chen-Zimmermann OpenSourceAP).
Chan, Lakonishok and Sougiannis (2001) -- high R&D/MktCap predicts positive future returns.

Per-ticker implementation: PIT R&D expense (TTM) / (Close * shares_outstanding).
Cross-sectional rank is inherently multi-stock, but the ratio itself is computed
faithfully per-ticker from PIT fundamentals -- a level signal capturing the same
economic substance (R&D intensity relative to market value).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load PIT fundamentals helper ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_rd",
    "description": (
        "R&D expense over market cap (TTM R&D / (Close * shares_outstanding)). "
        "Per-ticker PIT implementation of Chan, Lakonishok & Sougiannis (2001): "
        "stocks with higher R&D intensity relative to market value tend to earn "
        "positive future abnormal returns. Cross-sectional rank not available "
        "per-ticker -- the raw ratio captures the same economic signal."
        " Proxy uses fund_rnd_expense_ttm and fund_shares_outstanding."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_rd_ratio",      # TTM R&D / market cap (level)
        "osap_rd_chg_63d",    # 63-trading-day change in the ratio (momentum of signal)
    ],
    "tags": ["fundamentals", "rd", "innovation", "value", "accounting"],
    "version": "1.0",
    "author": "Chan, Lakonishok and Sougiannis (2001); OpenSourceAP (Chen-Zimmermann)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Fetch PIT fundamentals (backward merge on filed_date -- no lookahead)
    df = _fundamentals.as_of(df, fields=["rnd_expense_ttm", "shares_outstanding"])

    rnd = df["fund_rnd_expense_ttm"]           # TTM R&D expense (USD)
    shares = df["fund_shares_outstanding"]     # shares outstanding

    # Market cap = Close * shares_outstanding
    mktcap = df["Close"] * shares
    # Guard against zero / negative market cap
    mktcap_safe = mktcap.where(mktcap > 0, np.nan)

    # Core ratio: R&D / market cap  (0 when R&D is NaN or zero is fine -- stays 0)
    # But if rnd itself is NaN (no filing yet) -> NaN propagates naturally
    ratio = rnd / mktcap_safe

    # Replace inf/-inf just in case
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df["osap_rd_ratio"] = ratio

    # 63-trading-day (≈1 quarter) change in ratio: captures rising/falling R&D intensity
    df["osap_rd_chg_63d"] = ratio - ratio.shift(63)

    # Drop scratch fund_* columns not in produces
    df = df.drop(columns=["fund_rnd_expense_ttm", "fund_shares_outstanding"], errors="ignore")

    return df
