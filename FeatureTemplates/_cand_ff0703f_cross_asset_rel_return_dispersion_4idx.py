"""
_cand_ff0703f_cross_asset_rel_return_dispersion_4idx.py

SPEC ID: ff0703f_cross_asset_rel_return_dispersion_4idx
VEIN: cross_asset

METHOD
------
For each of SPY, QQQ, IWM, DIA (merge_asof-backward aligned to this ticker's Date), compute
the stock's 20-day relative return vs that index:
    rr_k = (Close_t / Close_{t-20}) - (Idx_k,t / Idx_k,t-20)
LEVEL:   rel_disp_4idx        = cross-index population std of the available rr_k values at each bar.
                                 High dispersion => the stock's outperformance is index-specific
                                 (idiosyncratic / size-or-style driven). Low dispersion => a
                                 uniform macro move that shows up the same way vs every benchmark.
DYNAMIC: rel_disp_4idx_slope  = 10-day change of rel_disp_4idx (value now minus value 10 bars ago).

Guard: computed over whichever indexes are available (>=2 of the 4); if fewer than 2 indexes
have data at a bar, that bar is NaN. This is causal (only past closes, backward merge_asof) and
orthogonal to single-benchmark relative-strength features because it encodes cross-benchmark
AGREEMENT/DISAGREEMENT rather than the level of relative strength against one index.

Faithfulness note: implemented exactly as specified using the four index parquet files via the
shared `_indexes` helper. No deviation from the spec was required.
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
    "name": "ff0703f_cross_asset_rel_return_dispersion_4idx",
    "description": (
        "Cross-index dispersion of the stock's 20-day relative return vs SPY/QQQ/IWM/DIA "
        "(population std of the four benchmark-relative returns, computed over whichever "
        "indexes are available, min 2); plus its 10-day change. High dispersion flags "
        "idiosyncratic/style-specific outperformance vs a uniform macro move. Faithful "
        "per-ticker implementation of the spec via merge_asof-backward index alignment."
    ),
    "requires": ["Close"],
    "produces": ["ff0703f_rel_disp_4idx", "ff0703f_rel_disp_4idx_slope"],
    "tags": ["cross_asset", "dispersion", "relative_strength", "index"],
    "version": "1.0",
    "author": "feature-factory (auto-generated, faithful to spec)",
}

_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
_WIN = 20
_SLOPE_LAG = 10


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df["ff0703f_rel_disp_4idx"] = np.nan
    df["ff0703f_rel_disp_4idx_slope"] = np.nan

    if n == 0 or "Close" not in df.columns:
        return df

    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else None
    if dates is None:
        return df

    close = pd.to_numeric(df["Close"], errors="coerce")

    # Stock's own 20-day return, causal (uses only current + past 20 bars).
    close_lag = close.shift(_WIN)
    stock_ret20 = np.where(close_lag != 0, close / close_lag - 1.0, np.nan)
    stock_ret20 = pd.Series(stock_ret20, index=df.index)

    # Build a small helper frame with Date for merge_asof alignment.
    left = pd.DataFrame({"Date": dates}).reset_index(names="_orig_idx")
    left = left.sort_values("Date")

    rr_cols = []
    for sym in _SYMBOLS:
        idx_close = _indexes.index_close(sym)
        if idx_close is None or len(idx_close) == 0:
            continue

        idx_frame = idx_close.rename("idx_close").reset_index()
        idx_frame.columns = ["Date", "idx_close"]
        idx_frame["Date"] = pd.to_datetime(idx_frame["Date"])
        idx_frame = idx_frame.sort_values("Date")

        merged = pd.merge_asof(left, idx_frame, on="Date", direction="backward")
        merged = merged.sort_values("_orig_idx")
        idx_aligned = pd.Series(merged["idx_close"].values, index=merged["_orig_idx"].values)
        idx_aligned = idx_aligned.reindex(df.index)

        idx_lag = idx_aligned.shift(_WIN)
        idx_ret20 = np.where(idx_lag.to_numpy() != 0, idx_aligned.to_numpy() / idx_lag.to_numpy() - 1.0, np.nan)
        idx_ret20 = pd.Series(idx_ret20, index=df.index)

        rr_k = stock_ret20 - idx_ret20
        rr_cols.append(rr_k)

    if len(rr_cols) < 2:
        return df

    rr_mat = np.column_stack([c.to_numpy() for c in rr_cols])  # shape (n, k)
    valid_mask = ~np.isnan(rr_mat)
    valid_count = valid_mask.sum(axis=1)

    with np.errstate(invalid="ignore"):
        masked = np.where(valid_mask, rr_mat, np.nan)
        mean_vals = np.nanmean(masked, axis=1)
        # population std over available values
        sq_dev = (masked - mean_vals[:, None]) ** 2
        var_vals = np.nanmean(sq_dev, axis=1)
        std_vals = np.sqrt(var_vals)

    std_vals = np.where(valid_count >= 2, std_vals, np.nan)

    rel_disp = pd.Series(std_vals, index=df.index)
    df["ff0703f_rel_disp_4idx"] = rel_disp
    df["ff0703f_rel_disp_4idx_slope"] = rel_disp - rel_disp.shift(_SLOPE_LAG)

    return df
