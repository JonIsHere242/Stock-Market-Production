"""
ff0703b_freq_stability_allan_vix_resid_80
Allan deviation of VIX-conditioned idiosyncratic returns over a trailing 80-day window.

Method (per spec ff0703b_freq_stability_allan_vix_resid_80):
  1. On a causal stride (i % 5 == 0, measured from series start, forward-filled between
     strides), take the trailing 80 daily log-returns r and the contemporaneous trailing
     log-changes of VIX (vx) from the shared `_indexes` helper (backward merge_asof, so
     VIX-as-of-day is never a future value).
  2. Regress r on vx (OLS with intercept) over that 80-obs window; residual e = the
     part of the stock's return NOT explained by same-day VIX moves (idiosyncratic,
     VIX-conditioned).
  3. Standard (non-overlapping block) Allan deviation of e at taus {1,2,4,8}:
         sigma_A^2(tau) = 0.5 * mean( (ybar_{k+1} - ybar_k)^2 )
     where ybar_k is the mean of block k of length tau.
  4. Output = log(Allan deviation at tau=2).
  NaN if fewer than 60 valid paired (r, vx) observations in the window, or VIX missing.

This is distinct from allan_idio_resid (which residualizes on SPY index returns,
not VIX). Here the conditioning variable is VIX changes, so the residual captures
idiosyncratic-return stability net of volatility-regime co-movement rather than
net of market beta. Per-ticker causal proxy; vectorised except for the small stride
loop (bars processed = n/5, work per bar = O(window)).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# --- load _indexes helper (by file path, per contract) ---
_spec = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name": "ff0703b_freq_stability_allan_vix_resid_80",
    "description": (
        "Log Allan deviation (tau=2) of residuals from a trailing-80-day causal OLS of "
        "daily log-returns on same-day log-changes in VIX. Captures the frequency-domain "
        "stability of the stock's idiosyncratic (VIX-conditioned) return component -- "
        "distinct from allan_idio_resid, which conditions on SPY returns rather than VIX. "
        "Computed on a stride=5 grid from the series start and forward-filled "
        "(causal-striding). NaN if <60 valid paired obs in the window or VIX unavailable. "
        "Per-ticker proxy using the shared _indexes VIX loader; no cross-sectional data."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703b_freq_stability_allan_vix_resid_80_level",
    ],
    "tags": ["frequency", "allan_deviation", "vix", "residual", "stability", "orthogonal"],
    "version": "1.0.0",
    "author": "feature-factory ff0703b",
}

_WINDOW = 80
_STRIDE = 5
_TAUS = [1, 2, 4, 8]
_TAU_TARGET = 2
_MIN_OBS = 60


def _standard_allan_dev(arr: np.ndarray, tau: int) -> float:
    """
    Standard (non-overlapping-block) Allan deviation at the given tau, per spec:
        sigma_A^2(tau) = 0.5 * mean( (ybar_{k+1} - ybar_k)^2 )
    where ybar_k is the mean of consecutive non-overlapping blocks of length tau.
    Requires at least 2 complete blocks. Returns np.nan on failure.
    """
    n_complete = len(arr) // tau
    if n_complete < 2:
        return np.nan
    trimmed = arr[: n_complete * tau]
    block_means = trimmed.reshape(n_complete, tau).mean(axis=1)
    d = np.diff(block_means)
    if len(d) == 0:
        return np.nan
    val = 0.5 * np.mean(d * d)
    if not np.isfinite(val) or val < 0:
        return np.nan
    return float(np.sqrt(val))


def _window_ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    OLS of y on x (with intercept) over the full passed-in window (already trailing
    and causal by construction of the caller). Returns residual array same length,
    NaN where y or x is NaN. Pairwise-complete: rows with NaN in either series are
    excluded from the fit but residuals array keeps original positions (NaN there).
    """
    n = len(y)
    resid = np.full(n, np.nan)
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < _MIN_OBS:
        return resid

    yv = y[valid]
    xv = x[valid]
    m = len(yv)
    sx = xv.sum()
    sy = yv.sum()
    sxx = (xv * xv).sum()
    sxy = (xv * yv).sum()
    denom = m * sxx - sx * sx
    if denom == 0.0 or not np.isfinite(denom):
        return resid
    beta = (m * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / m

    resid[valid] = yv - alpha - beta * xv
    return resid


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_level = "ff0703b_freq_stability_allan_vix_resid_80_level"

    df[col_level] = np.nan

    n = len(df)
    if n < _WINDOW:
        return df

    # --- VIX daily close, aligned via backward merge_asof (lookahead-safe) ---
    try:
        vix_df = _indexes.vix_daily_close()
    except Exception:
        vix_df = None

    if vix_df is None or vix_df.empty:
        return df

    df_work = df[["Date"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df = vix_df.sort_values("Date")

    merged = pd.merge_asof(
        df_work.sort_values("Date"),
        vix_df,
        on="Date",
        direction="backward",
    )
    merged = merged.set_index(df_work.sort_values("Date").index)
    merged = merged.reindex(df.index)

    vix_close = merged["vix_close"].values.astype(float)

    # --- Ticker log-returns ---
    close = df["Close"].values.astype(float)
    ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close[1:] / close[:-1]
        ret[1:] = np.where(ratio > 0, np.log(ratio), np.nan)

    # --- VIX log-changes (contemporaneous with ret) ---
    vx = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        vratio = vix_close[1:] / vix_close[:-1]
        vx[1:] = np.where(vratio > 0, np.log(vratio), np.nan)

    if np.sum(np.isfinite(vx)) < _MIN_OBS:
        return df

    level_arr = np.full(n, np.nan)
    last_level = np.nan

    for i in range(n):
        if i % _STRIDE == 0 and i >= _WINDOW - 1:
            i0 = i - _WINDOW + 1
            r_win = ret[i0 : i + 1]
            vx_win = vx[i0 : i + 1]

            valid_count = np.sum(np.isfinite(r_win) & np.isfinite(vx_win))
            if valid_count >= _MIN_OBS:
                resid = _window_ols_residuals(r_win, vx_win)
                resid_valid = resid[np.isfinite(resid)]
                if len(resid_valid) >= _MIN_OBS:
                    adev = _standard_allan_dev(resid_valid, _TAU_TARGET)
                    if np.isfinite(adev) and adev > 0:
                        last_level = np.log(adev)
                    else:
                        last_level = np.nan
                else:
                    last_level = np.nan
            else:
                last_level = np.nan

        level_arr[i] = last_level

    df[col_level] = level_arr

    return df
