"""
ext4_relative_strength_multi
Multi-benchmark relative strength vs QQQ, IWM, and SPY over a 63-day window,
plus the worst-case (minimum) relative strength across all three benchmarks.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never via package import)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_relative_strength_multi",
    "description": (
        "Per-ticker 63-day cumulative return minus each of QQQ, IWM, and SPY "
        "cumulative returns over the same window (backward-merged, no lookahead). "
        "Also computes the min across the three benchmarks as a worst-case relative "
        "strength signal. Cross-sectional ranking is not possible here, so this is a "
        "faithful per-ticker proxy: positive values mean the stock outpaced that "
        "benchmark over the trailing quarter."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_relative_strength_multi_vs_qqq",
        "ext4_relative_strength_multi_vs_iwm",
        "ext4_relative_strength_multi_vs_spy",
        "ext4_relative_strength_multi_worst",
    ],
    "tags": ["relative_strength", "momentum", "benchmark", "multi_benchmark"],
    "version": "1.0.0",
    "author": "Round-5 expansion spec (ext3_relative_strength); implemented per spec.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOW = 63  # ~1 quarter of trading days


def _cum_return_series(close: pd.Series) -> pd.Series:
    """63-day cumulative return at each bar (no lookahead)."""
    shifted = close.shift(_WINDOW)
    return (close - shifted) / shifted.replace(0, np.nan)


def _index_cum_return(symbol: str, df_dates: pd.Series) -> pd.Series:
    """
    Fetch index close, align to df dates via merge_asof backward,
    then compute 63-day cumulative return aligned to df index.
    Returns a Series aligned to df_dates.index, or all-NaN on failure.
    """
    try:
        idx_close = _indexes.index_close(symbol)
        if idx_close is None or idx_close.empty:
            return pd.Series(np.nan, index=df_dates.index)

        # Build a small DataFrame for merge_asof
        idx_df = idx_close.rename("_idx_close").reset_index()  # columns: Date, _idx_close
        idx_df.columns = ["Date", "_idx_close"]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])

        stock_dates = pd.DataFrame({"Date": pd.to_datetime(df_dates.values)}, index=df_dates.index)
        merged = pd.merge_asof(
            stock_dates.sort_values("Date"),
            idx_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        merged = merged.set_index(stock_dates.sort_values("Date").index)
        # Restore original order
        merged = merged.reindex(df_dates.index)

        idx_aligned = merged["_idx_close"]

        # Compute 63-day cumulative return on the aligned series
        shifted = idx_aligned.shift(_WINDOW)
        cum_ret = (idx_aligned - shifted) / shifted.replace(0, np.nan)
        cum_ret.index = df_dates.index
        return cum_ret

    except Exception:
        return pd.Series(np.nan, index=df_dates.index)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    dates = df["Date"]

    # Stock's own 63-day cumulative return
    stock_ret = _cum_return_series(close)
    stock_ret.index = df.index

    # Benchmark cumulative returns
    qqq_ret = _index_cum_return("QQQ", dates)
    iwm_ret = _index_cum_return("IWM", dates)
    spy_ret = _index_cum_return("SPY", dates)

    # Relative strength = stock - benchmark (excess return over trailing 63d)
    vs_qqq = stock_ret - qqq_ret
    vs_iwm = stock_ret - iwm_ret
    vs_spy = stock_ret - spy_ret

    # Replace inf with NaN (guard against 0-price edge cases)
    vs_qqq = vs_qqq.replace([np.inf, -np.inf], np.nan)
    vs_iwm = vs_iwm.replace([np.inf, -np.inf], np.nan)
    vs_spy = vs_spy.replace([np.inf, -np.inf], np.nan)

    # Worst-case relative strength (min across the three)
    combined = pd.concat([vs_qqq, vs_iwm, vs_spy], axis=1)
    worst = combined.min(axis=1)  # NaN-propagating: if all NaN -> NaN

    df["ext4_relative_strength_multi_vs_qqq"] = vs_qqq.values
    df["ext4_relative_strength_multi_vs_iwm"] = vs_iwm.values
    df["ext4_relative_strength_multi_vs_spy"] = vs_spy.values
    df["ext4_relative_strength_multi_worst"] = worst.values

    return df
