"""
fvn_valuation_zscores.py  --  OWN-HISTORY standardization of valuation yields (the EDGE).

A raw valuation yield (E/P, S/P, FCF/P, ...) is dominated by cross-sectional industry level: a bank
and a software firm have permanently different book yields, so the raw level is mostly a sector
dummy the model already proxies. The orthogonal signal is "cheap vs its OWN history": how far is THIS
stock's current yield from where IT has traded over the last ~6-12 months. That removes the static
per-name level and surfaces genuine re-rating / de-rating.

This block consumes the raw yields from fvn_valuation_yields and, for each, computes a trailing
z-score vs its own 120d and 250d distribution:

    z = (yield_t - rolling_mean) / rolling_std        (clipped, std==0 -> NaN)

Higher positive z = the stock is unusually CHEAP relative to its own recent norm (high yield),
a value/mean-reversion tilt orthogonal to plain price momentum.

Windows: 120d (~6mo) and 250d (~1yr). min_periods kept well below the window (history is ~645 rows)
so columns are not all-NaN. Forward-fill on the underlying yield is NOT applied here -- the as-of
merge already step-holds each filing, so the yield only moves on a new price or a new filing, which
is exactly the series we want to standardize.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The raw yields produced by fvn_valuation_yields.py. Listing them in `requires` makes the
# framework run that block first.
_YIELDS = [
    "fvn_earnings_yield",
    "fvn_sales_yield",
    "fvn_book_yield",
    "fvn_fcf_yield",
    "fvn_cfo_yield",
    "fvn_dividend_yield",
]

_WINDOWS = [120, 250]
_MIN_FRAC = 0.5          # min_periods = half the window
_Z_CLIP = 6.0

# Short tag per yield for compact column names.
_TAG = {
    "fvn_earnings_yield": "ey",
    "fvn_sales_yield":    "sy",
    "fvn_book_yield":     "by",
    "fvn_fcf_yield":      "fy",
    "fvn_cfo_yield":      "cy",
    "fvn_dividend_yield": "dy",
}

_PRODUCES = [f"fvn_{_TAG[y]}_z_{w}" for y in _YIELDS for w in _WINDOWS]

METADATA = {
    "name":        "fvn_valuation_zscores",
    "description": "Trailing 120d/250d z-scores of each valuation yield vs its OWN history "
                   "(cheap-vs-own-history), orthogonal to static cross-sectional valuation level.",
    "requires":    list(_YIELDS),
    "produces":    list(_PRODUCES),
    "tags":        ["fundamentals", "valuation", "zscore", "mean_reversion"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    for y in _YIELDS:
        if y in df.columns:
            s = pd.to_numeric(df[y], errors="coerce")
        else:
            # Defensive: producer skipped (no SEC coverage) -> emit all-NaN columns to honour contract.
            s = pd.Series(np.nan, index=df.index, dtype="float64")
        tag = _TAG[y]
        for w in _WINDOWS:
            mp = max(2, int(w * _MIN_FRAC))
            mean = s.rolling(w, min_periods=mp).mean()
            std = s.rolling(w, min_periods=mp).std()
            std = std.where(std > 0)  # avoid divide-by-zero -> NaN
            z = (s - mean) / std
            z = z.replace([np.inf, -np.inf], np.nan).clip(-_Z_CLIP, _Z_CLIP)
            df[f"fvn_{tag}_z_{w}"] = z
    return df
