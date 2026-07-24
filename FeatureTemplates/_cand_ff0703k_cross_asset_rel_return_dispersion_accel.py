"""
_cand_ff0703k_cross_asset_rel_return_dispersion_accel.py

Second-order dynamics of idiosyncratic-vs-macro spread.

S_t = mean_i(|rel_i|) where rel_i = CR20 - CR20_i is the stock-minus-benchmark
20-trading-day cumulative relative return for i in {SPY, QQQ, IWM, DIA}
(index prices joined via merge_asof-backward, lookahead-safe).

  LEVEL    ff0703k_cross_asset_rel_return_dispersion_accel_level = S_t
  DYNAMIC  ff0703k_cross_asset_rel_return_dispersion_accel_accel
           = S_t - 2*S_{t-10} + S_{t-20}   (acceleration of the divergence)

Faithful implementation notes: the spec's cross-sectional basket {SPY,QQQ,IWM,DIA}
maps directly onto the four cached index parquets exposed by `_indexes.py`, so no
proxy substitution is needed here. Each index's own 20d cumulative return is
computed on the index's native calendar and joined onto the stock's Date axis via
a backward merge_asof (same-day-or-prior index bar), which is lookahead-safe and
degrades gracefully (NaN) for any index whose data is missing or stale.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff0703k_cross_asset_rel_return_dispersion_accel",
    "description": (
        "Acceleration of the stock's 20d cumulative-return dispersion vs a 4-index "
        "macro basket (SPY/QQQ/IWM/DIA). S_t = mean_i(|CR20_stock - CR20_index_i|); "
        "dynamic column is the discrete second difference S_t - 2*S_{t-10} + S_{t-20}, "
        "i.e. whether idiosyncratic-vs-macro divergence is widening/narrowing faster or "
        "slower over time. Faithful per-ticker implementation using the cached index "
        "panel (backward merge_asof, lookahead-safe)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703k_cross_asset_rel_return_dispersion_accel_level",
        "ff0703k_cross_asset_rel_return_dispersion_accel_accel",
    ],
    "tags": ["cross_asset", "dispersion", "second_order", "macro_relative"],
    "version": "1.0",
    "author": (
        "feature-factory codegen; faithful implementation of spec ff0703k "
        "(basket = SPY,QQQ,IWM,DIA via _indexes.py cache, backward-merge_asof join)"
    ),
}

_BASKET = ["SPY", "QQQ", "IWM", "DIA"]
_WIN = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    level_col = "ff0703k_cross_asset_rel_return_dispersion_accel_level"
    accel_col = "ff0703k_cross_asset_rel_return_dispersion_accel_accel"

    df[level_col] = np.nan
    df[accel_col] = np.nan

    n = len(df)
    if n < _WIN + 1 or "Close" not in df.columns:
        return df

    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else None
    if dates is None:
        return df

    close = df["Close"].astype("float64")
    stock_cr20 = close / close.shift(_WIN) - 1.0
    stock_cr20 = stock_cr20.replace([np.inf, -np.inf], np.nan)

    work = pd.DataFrame({"Date": dates.values, "stock_cr20": stock_cr20.values})
    work = work.sort_values("Date")

    rel_frames = []
    for sym in _BASKET:
        idx_close = _indexes.index_close(sym)
        if idx_close is None or idx_close.empty:
            continue
        idx_close = idx_close.sort_index()
        idx_cr20 = idx_close / idx_close.shift(_WIN) - 1.0
        idx_cr20 = idx_cr20.replace([np.inf, -np.inf], np.nan)
        idx_df = idx_cr20.reset_index()
        idx_df.columns = ["Date", f"idx_cr20_{sym}"]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])
        idx_df = idx_df.sort_values("Date")

        merged = pd.merge_asof(
            work[["Date"]], idx_df, on="Date", direction="backward"
        )
        rel = work["stock_cr20"].values - merged[f"idx_cr20_{sym}"].values
        rel_frames.append(np.abs(rel))

    if not rel_frames:
        return df

    rel_stack = np.vstack(rel_frames)  # shape (n_indices, n_rows)
    with np.errstate(invalid="ignore"):
        s_t = np.nanmean(rel_stack, axis=0)
    # if ALL indices were NaN for a row, nanmean gives nan already (with a warning
    # suppressed above); require at least one finite value explicitly for safety
    valid_count = np.sum(~np.isnan(rel_stack), axis=0)
    s_t = np.where(valid_count > 0, s_t, np.nan)

    s_series = pd.Series(s_t, index=work.index)
    # work's index is inherited from df's original positional index (only its row
    # ORDER was sorted by Date); use .loc to scatter values back to original positions.
    s_aligned = pd.Series(np.nan, index=df.index)
    s_aligned.loc[work.index] = s_series.values

    s_t_minus10 = s_aligned.shift(10)
    s_t_minus20 = s_aligned.shift(20)

    accel = s_aligned - 2.0 * s_t_minus10 + s_t_minus20

    # guard: require the anchor window itself to have >= 20 rows of history
    row_pos = np.arange(n)
    insufficient_history = row_pos < (2 * _WIN)

    level_out = s_aligned.copy()
    accel_out = accel.copy()
    level_out[insufficient_history] = np.nan
    accel_out[insufficient_history] = np.nan

    df[level_col] = level_out.values
    df[accel_col] = accel_out.values

    return df
