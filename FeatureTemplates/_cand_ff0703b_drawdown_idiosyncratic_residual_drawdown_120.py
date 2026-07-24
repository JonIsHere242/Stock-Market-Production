"""
_cand_ff0703b_drawdown_idiosyncratic_residual_drawdown_120.py -- Idiosyncratic (SPY-residual) drawdown.

METHOD (per spec ff0703b_drawdown_idiosyncratic_residual_drawdown_120):
Roll a 120-day window. Inside each window, OLS-regress the stock's daily return on SPY's
daily return (r_t = alpha + beta*spy_t + e_t). Build the residual "idiosyncratic" equity
curve cumprod(1+e_t) over the window and take its max-drawdown magnitude. This isolates
drawdown risk that is NOT explained by market beta -- distinct from a plain beta-adjusted
drawdown (drawdown_beta), because here the *equity path itself* is reconstructed purely
from the regression residuals, not from beta-scaled returns.

Faithfulness note: true rolling per-bar OLS + per-bar residual-equity max-drawdown is
implemented fully vectorised via sliding_window_view + closed-form OLS (sums of
squares/cross-products), never a per-row Python loop. Bars with either return missing
(NaN) inside a window are neutralised (residual set to 0, i.e. a no-op day in the
residual equity path) and excluded from the beta/alpha sums, matching the spec's
"NaN if n<40" rule on the count of *valid* (non-NaN, paired) observations in the window.

Produces:
  - ff0703b_drawdown_idiosyncratic_residual_drawdown_120: max drawdown magnitude (0..1) of
    the 120d residual equity curve.
  - ff0703b_drawdown_idiosyncratic_residual_drawdown_recent_ratio: share of that drawdown
    depth attributable to the most recent 20 trading days of the same window (recency /
    dynamics of the idiosyncratic drawdown -- is it fresh or old damage).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

_SPECID = "ff0703b_drawdown_idiosyncratic_residual_drawdown_120"

METADATA = {
    "name": "ff0703b_drawdown_idiosyncratic_residual_drawdown_120",
    "description": (
        "Max drawdown magnitude of the SPY-residual (idiosyncratic) equity curve over a "
        "rolling 120d window: regress daily returns on SPY daily returns via closed-form "
        "OLS (per-window, fully vectorised), build cumprod(1+residual) equity, take its "
        "max drawdown. Distinct from beta-scaled drawdown -- this measures stock-specific "
        "damage net of market beta. Faithful vectorised proxy of the spec (per-ticker, "
        "OHLCV+SPY only; no cross-sectional or training dependency needed)."
    ),
    "requires": ["Close"],
    "produces": [
        f"{_SPECID}",
        f"{_SPECID}_recent_ratio",
    ],
    "tags": ["drawdown", "idiosyncratic", "residual", "beta", "risk"],
    "version": "1.0",
    "author": "feature-factory (ff0703b codegen)",
}

_WINDOW = 120
_RECENT = 20
_MIN_N = 40


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    col_main = f"{_SPECID}"
    col_recent = f"{_SPECID}_recent_ratio"

    df[col_main] = np.nan
    df[col_recent] = np.nan

    if n < _WINDOW:
        return df

    close = df["Close"].astype(float)
    ret = close.pct_change()

    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = pd.DataFrame({"Date": pd.DatetimeIndex(spy_close.index), "spy_close": spy_close.values})
    spy_df = spy_df.sort_values("Date")

    left = df[["Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"])
    left["_orig_order"] = np.arange(n)
    left = left.sort_values("Date")

    merged = pd.merge_asof(left, spy_df, on="Date", direction="backward")
    merged = merged.sort_values("_orig_order")
    spy_close_aligned = merged["spy_close"].to_numpy(dtype=float)
    spy_ret = pd.Series(spy_close_aligned).pct_change().to_numpy(dtype=float)

    ret_arr = ret.to_numpy(dtype=float)

    valid_mask = np.isfinite(ret_arr) & np.isfinite(spy_ret)
    ret_f = np.where(valid_mask, ret_arr, 0.0)
    spy_f = np.where(valid_mask, spy_ret, 0.0)

    count_w = n - _WINDOW + 1
    # sliding windows: shape (count_w, _WINDOW)
    ret_w = np.lib.stride_tricks.sliding_window_view(ret_f, _WINDOW)
    spy_w = np.lib.stride_tricks.sliding_window_view(spy_f, _WINDOW)
    valid_w = np.lib.stride_tricks.sliding_window_view(valid_mask, _WINDOW)

    counts = valid_w.sum(axis=1).astype(float)
    counts_safe = np.where(counts > 0, counts, np.nan)

    sum_x = spy_w.sum(axis=1)
    sum_y = ret_w.sum(axis=1)
    sum_xx = (spy_w * spy_w).sum(axis=1)
    sum_xy = (spy_w * ret_w).sum(axis=1)

    mean_x = sum_x / counts_safe
    mean_y = sum_y / counts_safe
    var_x = sum_xx / counts_safe - mean_x * mean_x
    cov_xy = sum_xy / counts_safe - mean_x * mean_y

    var_x_safe = np.where((counts >= _MIN_N) & (var_x > 0), var_x, np.nan)
    beta = cov_xy / var_x_safe
    alpha = mean_y - beta * mean_x

    # residual for every day in the window; invalid (NaN-source) days -> residual 0 (no-op)
    resid_w = ret_w - (alpha[:, None] + beta[:, None] * spy_w)
    resid_w = np.where(valid_w, resid_w, 0.0)
    resid_w = np.where(np.isfinite(resid_w), resid_w, 0.0)

    equity = np.cumprod(1.0 + resid_w, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    running_max_safe = np.where(running_max > 0, running_max, np.nan)
    dd = (running_max_safe - equity) / running_max_safe
    dd = np.where(np.isfinite(dd), dd, 0.0)

    max_dd_full = dd.max(axis=1)
    max_dd_recent = dd[:, -_RECENT:].max(axis=1)

    valid_row = (counts >= _MIN_N) & np.isfinite(beta) & np.isfinite(var_x_safe)

    max_dd_full = np.clip(max_dd_full, 0.0, 1.0)
    max_dd_recent = np.clip(max_dd_recent, 0.0, 1.0)

    max_dd_full = np.where(valid_row, max_dd_full, np.nan)
    dd_full_safe = np.where((max_dd_full > 0) & np.isfinite(max_dd_full), max_dd_full, np.nan)
    recent_ratio = np.where(np.isfinite(dd_full_safe), max_dd_recent / dd_full_safe, np.nan)
    recent_ratio = np.clip(recent_ratio, 0.0, 1.0)
    recent_ratio = np.where(valid_row, recent_ratio, np.nan)

    out_main = np.full(n, np.nan)
    out_recent = np.full(n, np.nan)
    out_main[_WINDOW - 1:] = max_dd_full
    out_recent[_WINDOW - 1:] = recent_ratio

    df[col_main] = out_main
    df[col_recent] = out_recent

    return df
