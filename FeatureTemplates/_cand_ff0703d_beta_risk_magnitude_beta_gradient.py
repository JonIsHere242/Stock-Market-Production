"""
Magnitude-conditional beta gradient feature block.

Tests whether a stock's market sensitivity (beta vs SPY) amplifies on
large-magnitude market-move days relative to small-magnitude days.

Over a trailing 120-day window, splits days into terciles by |SPY return|
(small / mid / large). Computes beta = cov(r, m) / var(m) separately within
the large-|m| tercile and the small-|m| tercile (NaN if a tercile has < 8
valid obs or var(m) == 0). Emits:
  - LEVEL:      beta on large-|m| days
  - ASYMMETRY:  large-tercile beta minus small-tercile beta (co-movement
                amplification with move magnitude)
  - SIGNED:     the same asymmetry restricted to down-market days (m < 0)
  - DYNAMIC:    change in the asymmetry vs the window ending 20 days ago

Per-ticker OHLCV proxy using SPY via the _indexes helper. Computed on a
fixed-from-series-start stride (i % 5 == 0) and forward-filled for
causality/perf; all inputs to each window are strictly past-and-current.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, per contract)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
_PFX = "ff0703d_beta_risk_magnitude_beta_gradient"

METADATA = {
    "name": "ff0703d_beta_risk_magnitude_beta_gradient",
    "description": (
        "Magnitude-conditional beta gradient vs SPY. Trailing 120-day window; "
        "splits days into terciles by |SPY daily return| and computes "
        "beta=cov(r,m)/var(m) within the large-|m| tercile and small-|m| tercile "
        "separately (NaN if a tercile has <8 obs or var(m)==0). Produces the "
        "large-move beta level, the large-minus-small asymmetry (amplification "
        "of co-movement with move size), a down-market-only signed variant, and "
        "the 20-day change in the asymmetry. Fixed-from-start stride (i%5==0), "
        "forward-filled. Per-ticker OHLCV proxy using SPY via _indexes."
    ),
    "requires": ["Close"],
    "produces": [
        f"{_PFX}_large_beta",
        f"{_PFX}_gradient",
        f"{_PFX}_gradient_down",
        f"{_PFX}_gradient_chg",
    ],
    "tags": ["beta", "risk", "market", "asymmetry", "spy", "magnitude"],
    "version": "1.0.0",
    "author": (
        "ff0703d beta_risk vein spec — faithful per-ticker implementation "
        "(tercile-conditional beta split by |SPY move| magnitude)."
    ),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120
_MIN_OBS = 40
_TERCILE_MIN = 8
_STRIDE = 5
_CHG_LAG = 20


def _tercile_beta(m: np.ndarray, r: np.ndarray, mask: np.ndarray) -> float:
    """cov(r,m)/var(m) restricted to mask; NaN if too few obs or var(m)==0."""
    if mask.sum() < _TERCILE_MIN:
        return np.nan
    mm = m[mask]
    rr = r[mask]
    if len(mm) < 2:
        return np.nan
    varm = np.var(mm, ddof=1)
    if not np.isfinite(varm) or varm == 0:
        return np.nan
    covm = np.cov(rr, mm, ddof=1)[0, 1]
    if not np.isfinite(covm):
        return np.nan
    beta = covm / varm
    if not np.isfinite(beta):
        return np.nan
    return beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    large_col = f"{_PFX}_large_beta"
    grad_col = f"{_PFX}_gradient"
    grad_down_col = f"{_PFX}_gradient_down"
    grad_chg_col = f"{_PFX}_gradient_chg"

    df[large_col] = np.nan
    df[grad_col] = np.nan
    df[grad_down_col] = np.nan
    df[grad_chg_col] = np.nan

    if n < _WINDOW + 2:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY close and align to this stock's dates (backward, no lookahead)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    stock_df = df[["Date"]].copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    merged = pd.merge_asof(
        stock_df.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    merged = merged.set_index(stock_df.sort_values("Date").index)
    merged = merged.reindex(df.index)

    spy_c = merged["spy_close"].values

    # ------------------------------------------------------------------
    # Daily returns
    # ------------------------------------------------------------------
    close_arr = df["Close"].values.astype(float)

    stock_ret = np.full(n, np.nan)
    stock_ret[1:] = np.where(
        close_arr[:-1] != 0,
        (close_arr[1:] - close_arr[:-1]) / close_arr[:-1],
        np.nan,
    )

    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = np.where(
        spy_c[:-1] != 0,
        (spy_c[1:] - spy_c[:-1]) / spy_c[:-1],
        np.nan,
    )

    # ------------------------------------------------------------------
    # Fixed-from-start stride grid: only compute on i % STRIDE == 0
    # ------------------------------------------------------------------
    large_arr = np.full(n, np.nan)
    grad_arr = np.full(n, np.nan)
    grad_down_arr = np.full(n, np.nan)

    for t in range(_WINDOW - 1, n):
        if t % _STRIDE != 0:
            continue

        w_spy = spy_ret[t - _WINDOW + 1 : t + 1]
        w_stk = stock_ret[t - _WINDOW + 1 : t + 1]

        valid = ~(np.isnan(w_spy) | np.isnan(w_stk))
        ws = w_spy[valid]
        wk = w_stk[valid]

        if len(ws) < _MIN_OBS:
            continue

        abs_m = np.abs(ws)
        q1, q2 = np.percentile(abs_m, [100.0 / 3.0, 200.0 / 3.0])

        small_mask = abs_m <= q1
        large_mask = abs_m >= q2

        beta_large = _tercile_beta(ws, wk, large_mask)
        beta_small = _tercile_beta(ws, wk, small_mask)

        large_arr[t] = beta_large
        if np.isfinite(beta_large) and np.isfinite(beta_small):
            grad_arr[t] = beta_large - beta_small

        down_mask = ws < 0
        beta_large_down = _tercile_beta(ws, wk, large_mask & down_mask)
        beta_small_down = _tercile_beta(ws, wk, small_mask & down_mask)
        if np.isfinite(beta_large_down) and np.isfinite(beta_small_down):
            grad_down_arr[t] = beta_large_down - beta_small_down

    # ------------------------------------------------------------------
    # Forward-fill the stride grid (causal: only uses past computed values)
    # ------------------------------------------------------------------
    large_filled = pd.Series(large_arr).ffill().values
    grad_filled = pd.Series(grad_arr).ffill().values
    grad_down_filled = pd.Series(grad_down_arr).ffill().values

    # Dynamic: change vs the window ending _CHG_LAG days ago (causal shift)
    grad_chg = grad_filled - pd.Series(grad_filled).shift(_CHG_LAG).values

    # ------------------------------------------------------------------
    # Guard infs and assign
    # ------------------------------------------------------------------
    large_filled = np.where(np.isinf(large_filled), np.nan, large_filled)
    grad_filled = np.where(np.isinf(grad_filled), np.nan, grad_filled)
    grad_down_filled = np.where(np.isinf(grad_down_filled), np.nan, grad_down_filled)
    grad_chg = np.where(np.isinf(grad_chg), np.nan, grad_chg)

    df[large_col] = large_filled
    df[grad_col] = grad_filled
    df[grad_down_col] = grad_down_filled
    df[grad_chg_col] = grad_chg

    return df
