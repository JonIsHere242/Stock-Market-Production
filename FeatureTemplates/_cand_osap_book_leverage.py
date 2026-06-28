"""
Book Leverage (annual) — Fama & French 1992 / OpenSourceAP (Chen-Zimmermann)

Book leverage = Total Assets / Book Equity (+ deferred taxes proxy).
Predicted sign: -1 (higher leverage → lower future returns, cross-sectionally).

Per-ticker implementation using PIT fundamentals via _fundamentals.as_of().
Cross-sectional ranking is not done here; the raw ratio and its change are emitted
so the model can learn the monotone signal within and across stocks.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load PIT fundamentals helper ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_book_leverage",
    "description": (
        "Book Leverage: total assets divided by book equity (shareholders' equity; "
        "fallback to assets minus total liabilities when equity is missing). "
        "Fama & French (1992) factor; higher leverage predicts lower returns (sign=-1). "
        "Implemented as a per-ticker PIT ratio from SEC fundamentals — inherently an "
        "annual accounting signal updated at each filing date. "
        "Produces: level ratio, its trailing 4-quarter change (momentum of leverage), "
        "and a z-score of the ratio over the stock's own recent history."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_book_leverage_ratio",   # assets / book_equity (level)
        "osap_book_leverage_chg",     # change vs ~1yr ago (leverage momentum)
        "osap_book_leverage_zscore",  # z-score of ratio over trailing 252-day window
    ],
    "tags": ["leverage", "fundamentals", "accounting", "fama_french", "annual"],
    "version": "1.0.0",
    "author": "Fama and French 1992 / OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute book-leverage features per ticker using PIT SEC fundamentals.

    Returns df with three new columns added.
    """
    # Pull PIT fundamentals — backward merge_asof on filed_date is handled inside
    needed = ["assets", "equity", "liabilities"]
    df = _fundamentals.as_of(df, fields=needed)

    # --- Book equity: use equity (seq) if available; fallback = assets - liabilities ---
    # fund_equity maps to shareholders' equity (seq in Compustat convention).
    # fund_liabilities maps to total liabilities (lt).
    # fund_assets = total assets (at).
    book_eq = df["fund_equity"].copy()

    # Fallback: assets - liabilities when equity is NaN
    fallback_eq = df["fund_assets"] - df["fund_liabilities"]
    book_eq = book_eq.where(book_eq.notna() & (book_eq != 0), other=fallback_eq)

    # Guard: if still zero/negative equity, set NaN (leverage ratio undefined / extreme)
    book_eq = book_eq.where(book_eq.abs() > 0, other=np.nan)

    # --- Level: book leverage ratio ---
    lev = df["fund_assets"] / book_eq
    # Cap extreme ratios (>100x assets vs equity is typically distressed noise)
    lev = lev.where(lev.between(-100, 100), other=np.nan)
    df["osap_book_leverage_ratio"] = lev

    # --- Dynamic: change in leverage over ~252 trading days (~1 fiscal year) ---
    lev_shift = lev.shift(252)
    lev_chg = lev - lev_shift
    df["osap_book_leverage_chg"] = lev_chg

    # --- Z-score of leverage ratio within own trailing 252-day window ---
    roll_mean = lev.rolling(252, min_periods=60).mean()
    roll_std = lev.rolling(252, min_periods=60).std(ddof=1)
    roll_std = roll_std.replace(0, np.nan)
    lev_z = (lev - roll_mean) / roll_std
    # Clip z-score to avoid extreme outliers polluting tree splits
    lev_z = lev_z.clip(-5, 5)
    df["osap_book_leverage_zscore"] = lev_z

    # Drop scratch fund_ columns we are not producing
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
