"""
osap_accrualsbm — Book-to-market and accruals combined signal
Source: OpenSourceAP (Chen-Zimmermann), Bartov and Kim (2004)

Economic signal: stocks with HIGH accruals AND LOW book-to-market are expected
to underperform (long-short: +1 = low accruals + high BM; short = high accruals
+ low BM).

Per-ticker proxy (cross-sectional quintile ranking is not feasible in a
single-stock compute() call):
  - accruals = (change in non-cash working capital) / total assets
      = ((delta current assets - delta cash) - (delta current liabilities
         - delta short-term debt)) / assets
      Approximated with PIT fundamentals: assets_current, cash, liabilities_current,
      total_debt, assets.
  - book_to_market = book_value_per_share / Close  (using fund_book_value_per_share)
  - book equity guard: exclude (NaN out) rows where equity <= 0.

Produces:
  osap_accrualsbm_accruals    — rolling accrual ratio (level)
  osap_accrualsbm_bm          — book-to-market ratio (level; higher = cheaper)
  osap_accrualsbm_score       — composite score: bm_zscore - accruals_zscore
                                (higher = more long-signal-like; per-ticker
                                trailing 252-day z-score to make it stationary)

NOTE: The original signal is a cross-sectional binary quintile variable. This
implementation captures the same economic ingredients per-ticker using rolling
z-scores. Cross-sectional quintile membership (needed for the exact binary
variable) cannot be determined inside a single stock's compute() call.
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
    "name": "osap_accrualsbm",
    "description": (
        "Per-ticker proxy for Bartov & Kim (2004) book-to-market + accruals "
        "combined signal (OpenSourceAP / Chen-Zimmermann). Original signal is "
        "a cross-sectional binary quintile variable; this block implements it "
        "as per-ticker accrual ratio, BM ratio, and a trailing z-score composite "
        "(bm_z - accruals_z). Rows with non-positive book equity are set to NaN. "
        "Predicted sign: +1 (long high BM + low accruals)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_accrualsbm_accruals",
        "osap_accrualsbm_bm",
        "osap_accrualsbm_score",
    ],
    "tags": ["valuation", "accounting", "accruals", "book_to_market", "osap"],
    "version": "1.0.0",
    "author": "Bartov and Kim (2004); OpenSourceAP Chen-Zimmermann; block by claude-sonnet-4-6",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals needed
    df = _fundamentals.as_of(
        df,
        fields=[
            "assets_current",
            "cash",
            "liabilities_current",
            "total_debt",
            "assets",
            "equity",
            "book_value_per_share",
        ],
    )

    # -----------------------------------------------------------------------
    # Guard: set everything to NaN where equity <= 0 (book equity negative)
    # -----------------------------------------------------------------------
    eq = df["fund_equity"]
    bad_equity = eq.isna() | (eq <= 0.0)

    # -----------------------------------------------------------------------
    # Accruals = change in (non-cash working capital) / total assets
    # Non-cash working capital = (assets_current - cash) - (liabilities_current - total_debt)
    # -----------------------------------------------------------------------
    ac = df["fund_assets_current"].fillna(np.nan)
    cash = df["fund_cash"].fillna(np.nan)
    lc = df["fund_liabilities_current"].fillna(np.nan)
    td = df["fund_total_debt"].fillna(np.nan)
    assets = df["fund_assets"].fillna(np.nan)

    # non-cash WC level
    noncash_wc = (ac - cash) - (lc - td)
    # change (period-over-period in the backward-merged PIT data)
    delta_wc = noncash_wc.diff(1)
    # scale by average assets to reduce size bias
    avg_assets = (assets + assets.shift(1)) / 2.0
    accruals = delta_wc / avg_assets.replace(0.0, np.nan)
    accruals = accruals.replace([np.inf, -np.inf], np.nan)
    accruals = accruals.where(~bad_equity, np.nan)

    # -----------------------------------------------------------------------
    # Book-to-market ratio: book_value_per_share / Close
    # -----------------------------------------------------------------------
    bvps = df["fund_book_value_per_share"].fillna(np.nan)
    close = df["Close"].replace(0.0, np.nan)
    bm = bvps / close
    bm = bm.replace([np.inf, -np.inf], np.nan)
    bm = bm.where(~bad_equity, np.nan)

    # -----------------------------------------------------------------------
    # Composite score: trailing 252-day z-score of BM minus z-score of accruals
    # Higher score = higher BM (value) AND lower accruals (quality) → long signal
    # -----------------------------------------------------------------------
    window = 252

    def rolling_zscore(s: pd.Series, w: int) -> pd.Series:
        mu = s.rolling(w, min_periods=max(20, w // 4)).mean()
        sd = s.rolling(w, min_periods=max(20, w // 4)).std(ddof=1)
        sd = sd.replace(0.0, np.nan)
        return (s - mu) / sd

    bm_z = rolling_zscore(bm, window)
    acc_z = rolling_zscore(accruals, window)
    score = bm_z - acc_z
    score = score.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------------------------
    # Assign to df and drop scratch fund_ columns not in produces
    # -----------------------------------------------------------------------
    df["osap_accrualsbm_accruals"] = accruals
    df["osap_accrualsbm_bm"] = bm
    df["osap_accrualsbm_score"] = score

    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
