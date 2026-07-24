"""
Quantile Beta Fan feature block.

Over a rolling 120-day window of daily returns, fits pinball-loss (quantile)
regressions of the stock's daily return on SPY's daily return at
tau = 0.10, 0.50, 0.90, via Iteratively Reweighted Least Squares (IRLS) on a
2-parameter (intercept, slope) weighted-least-squares problem -- a standard,
fully vectorisable per-window solve for quantile regression (equivalent to the
pinball-loss LP solution at convergence).

Produces:
  - ff0703c_beta_risk_quantile_beta_fan_level  : beta at tau=0.50 (median
      comovement -- robust to outliers vs OLS beta).
  - ff0703c_beta_risk_quantile_beta_fan_asym   : beta_0.90 - beta_0.10 (fan
      width -- how much co-movement slope differs between the stock's best
      and worst days relative to SPY).
  - ff0703c_beta_risk_quantile_beta_fan_dyn    : 60-day change in fan width
      (is the asymmetry widening or narrowing).

Per-ticker proxy using SPY as the market factor via the _indexes helper.
Computed on a fixed-from-start stride-5 grid (IRLS solves are not free) and
forward-filled -- causal, no lookahead. Returns NaN wherever the solver does
not converge or market variance in the window is ~0.
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
METADATA = {
    "name": "ff0703c_beta_risk_quantile_beta_fan",
    "description": (
        "Rolling 120-day pinball-loss (quantile) regression of stock return on "
        "SPY return at tau=0.10/0.50/0.90, solved via IRLS weighted least "
        "squares on a fixed-from-start stride-5 grid, forward-filled. LEVEL = "
        "median beta (tau=0.50, robust comovement). ASYM = beta_0.90 - "
        "beta_0.10 (fan width: does the stock's slope to SPY differ on its "
        "best vs worst relative days). DYN = 60-day change in fan width. "
        "Per-ticker proxy vs SPY (true method is cross-sectional/market-model; "
        "this captures the same tau-conditional-comovement signal). NaN when "
        "the IRLS solve fails to converge or market variance in-window is 0."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703c_beta_risk_quantile_beta_fan_level",
        "ff0703c_beta_risk_quantile_beta_fan_asym",
        "ff0703c_beta_risk_quantile_beta_fan_dyn",
    ],
    "tags": ["beta_risk", "beta", "quantile", "market", "spy", "asymmetry", "ff0703c"],
    "version": "1.0.0",
    "author": (
        "Batch 0703c spec ff0703c_beta_risk_quantile_beta_fan -- faithful "
        "per-ticker IRLS pinball-regression proxy vs SPY."
    ),
}

_WINDOW = 120
_STRIDE = 5
_TAUS = (0.10, 0.50, 0.90)
_MIN_VALID = 60
_DYN_LAG = 60
_MAX_ITER = 15
_EPS = 1e-6


def _fit_quantile_beta(m: np.ndarray, r: np.ndarray, tau: float) -> float:
    """IRLS pinball-loss weighted least squares for 1 predictor + intercept."""
    n = len(m)
    if n < _MIN_VALID:
        return np.nan
    mv = np.var(m)
    if not np.isfinite(mv) or mv <= 1e-12:
        return np.nan

    # OLS warm start
    m_mean = m.mean()
    r_mean = r.mean()
    denom0 = np.sum((m - m_mean) ** 2)
    if denom0 <= 1e-12:
        return np.nan
    b = np.sum((m - m_mean) * (r - r_mean)) / denom0
    a = r_mean - b * m_mean

    for _ in range(_MAX_ITER):
        resid = r - a - b * m
        absr = np.abs(resid)
        absr = np.where(absr < _EPS, _EPS, absr)
        w = np.where(resid >= 0, tau, 1.0 - tau) / absr

        sw = np.sum(w)
        swm = np.sum(w * m)
        swm2 = np.sum(w * m * m)
        swr = np.sum(w * r)
        swmr = np.sum(w * m * r)

        denom = sw * swm2 - swm * swm
        if not np.isfinite(denom) or abs(denom) <= 1e-12:
            return np.nan

        b_new = (sw * swmr - swm * swr) / denom
        a_new = (swr - b_new * swm) / sw

        if not (np.isfinite(a_new) and np.isfinite(b_new)):
            return np.nan

        if abs(b_new - b) < 1e-8 and abs(a_new - a) < 1e-8:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(b)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    level_col = "ff0703c_beta_risk_quantile_beta_fan_level"
    asym_col = "ff0703c_beta_risk_quantile_beta_fan_asym"
    dyn_col = "ff0703c_beta_risk_quantile_beta_fan_dyn"

    df[level_col] = np.nan
    df[asym_col] = np.nan
    df[dyn_col] = np.nan

    if n < _WINDOW + 2:
        return df

    # ------------------------------------------------------------------
    # SPY close aligned to this stock's dates (backward asof merge)
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
    # Fixed-from-start stride grid: solve IRLS quantile betas at each grid pt
    # ------------------------------------------------------------------
    level_out = np.full(n, np.nan)
    asym_out = np.full(n, np.nan)

    for t in range(_WINDOW - 1, n):
        if t % _STRIDE != 0:
            continue

        w_spy = spy_ret[t - _WINDOW + 1 : t + 1]
        w_stk = stock_ret[t - _WINDOW + 1 : t + 1]

        valid = ~(np.isnan(w_spy) | np.isnan(w_stk))
        ws = w_spy[valid]
        wk = w_stk[valid]

        if len(ws) < _MIN_VALID:
            continue

        b10 = _fit_quantile_beta(ws, wk, 0.10)
        b50 = _fit_quantile_beta(ws, wk, 0.50)
        b90 = _fit_quantile_beta(ws, wk, 0.90)

        if not (np.isfinite(b10) and np.isfinite(b50) and np.isfinite(b90)):
            continue

        level_out[t] = b50
        asym_out[t] = b90 - b10

    # Forward-fill the grid results (causal -- only uses past computed points)
    level_s = pd.Series(level_out)
    asym_s = pd.Series(asym_out)
    level_ff = level_s.ffill().values
    asym_ff = asym_s.ffill().values

    # ------------------------------------------------------------------
    # Dynamic: 60-trading-day change of the (forward-filled) fan width
    # ------------------------------------------------------------------
    dyn_out = np.full(n, np.nan)
    if n > _DYN_LAG:
        prev = asym_ff[: n - _DYN_LAG]
        curr = asym_ff[_DYN_LAG:]
        diff = curr - prev
        diff = np.where(np.isnan(prev) | np.isnan(curr), np.nan, diff)
        dyn_out[_DYN_LAG:] = diff

    level_ff = np.where(np.isinf(level_ff), np.nan, level_ff)
    asym_ff = np.where(np.isinf(asym_ff), np.nan, asym_ff)
    dyn_out = np.where(np.isinf(dyn_out), np.nan, dyn_out)

    df[level_col] = level_ff
    df[asym_col] = asym_ff
    df[dyn_col] = dyn_out

    return df
