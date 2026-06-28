"""
osap_bm — Book-to-Market ratio (Stattman 1980) via OpenSourceAP (Chen-Zimmermann).

Per-ticker proxy: log(book_value_per_share / Close) is equivalent to
log(book_equity / market_equity) when shares cancel, and is fully
lookahead-safe because we pull book_value_per_share via PIT as_of().

Cross-sectional note: the original signal (Stattman 1980) is ranked
cross-sectionally. Here we compute the raw per-ticker log ratio
(level + 52-week rolling z-score) so the upstream predictor can
still apply XS-ranking at inference time.
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
    "name": "osap_bm",
    "description": (
        "Book-to-Market ratio (Stattman 1980) via OpenSourceAP (Chen-Zimmermann). "
        "Computes log(book_value_per_share / Close) per ticker using point-in-time "
        "fundamentals so there is no lookahead. The raw log ratio (level) is paired "
        "with a 252-day rolling z-score to capture regime shifts. Because the original "
        "signal is inherently cross-sectional (rank BM across universe), these per-ticker "
        "values serve as inputs for XS-ranking at prediction time."
    ),
    "requires": ["Close"],
    "produces": ["osap_bm_log", "osap_bm_zscore"],
    "tags": ["valuation", "fundamental", "book_to_market", "osap"],
    "version": "1.0",
    "author": "Stattman 1980 / OpenSourceAP Chen-Zimmermann; impl proxy by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT book value per share (filed_date matched, backward)
    df = _fundamentals.as_of(df, fields=["book_value_per_share"])

    bvps = df["fund_book_value_per_share"]
    close = df["Close"]

    # Avoid divide-by-zero / non-positive prices or book values
    denom = close.replace(0, np.nan)
    ratio = bvps / denom  # book equity per share / price per share

    # Only take log where ratio is strictly positive (negative BV -> NaN)
    log_bm = np.where(ratio > 0, np.log(ratio), np.nan)
    log_bm = pd.Series(log_bm, index=df.index)

    df["osap_bm_log"] = log_bm

    # 252-day rolling z-score of the log B/M (captures mean-reversion signal)
    roll = log_bm.rolling(252, min_periods=63)
    df["osap_bm_zscore"] = (log_bm - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)

    # Drop scratch fundamental column
    df.drop(columns=["fund_book_value_per_share"], inplace=True)

    return df
