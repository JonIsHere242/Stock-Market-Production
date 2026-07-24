"""
Cross-index downside expectile-beta dispersion.

Method (per spec ff0703c_beta_risk_cross_index_downside_expectile_dispersion):
  For each of the four broad indexes (SPY, QQQ, IWM, DIA), compute the
  tau=0.10 downside expectile beta of the stock's daily return on the
  index's daily return, over a rolling 120-day window, fit by IRLS
  (asymmetric least squares).  Also compute the analogous tau=0.90 upside
  expectile beta for the asymmetry column.

  LEVEL      = mean of the four downside (tau=0.10) expectile betas
               -> broad crash-sensitivity level.
  DISPERSION = std across the four downside expectile betas
               -> how differently the stock crash-co-loads on large-cap
                  (SPY) vs tech (QQQ) vs small-cap (IWM) vs Dow (DIA);
                  a style/factor-tilt-in-the-tail axis.
  ASYMMETRY  = DISPERSION (tau=0.10) minus the analogous upside dispersion
               (std across the four tau=0.90 expectile betas)
               -> whether the stock's cross-index style tilt is a
                  downside-specific ("crash beta drift") phenomenon or
                  symmetric.

  Expensive IRLS fits are only evaluated on a fixed-from-start grid
  (i % 5 == 0, counted from row 0) and forward-filled -- this satisfies
  the causal-striding requirement (the grid never re-anchors to the last
  row, so it is stable under truncation).

Proxy notes: expectile regression is implemented via a small, fully
vectorised-per-window IRLS (asymmetric weighted least squares) with a
fixed iteration count -- this is the standard consistent estimator for
expectile beta and is faithful to the spec. Indexes unavailable on disk
degrade to NaN gracefully (per-index), the level/dispersion/asymmetry
columns then fall back to whatever subset of indexes is available (NaN
if fewer than 2 indexes have valid data, since dispersion needs >=2).
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path – required by sandbox rules)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
_PFX = "ff0703c_beta_risk_cross_index_downside_expectile_dispersion"

COL_LEVEL = f"{_PFX}_level"
COL_DISP = f"{_PFX}_dispersion"
COL_ASYM = f"{_PFX}_asymmetry"

METADATA = {
    "name": "ff0703c_beta_risk_cross_index_downside_expectile_dispersion",
    "description": (
        "Rolling 120-day tau=0.10 downside expectile beta (IRLS asymmetric "
        "least squares) of stock return on each of SPY/QQQ/IWM/DIA index "
        "returns. LEVEL = mean of the four downside expectile betas (broad "
        "crash sensitivity). DISPERSION = std across the four downside "
        "betas (style/factor-tilt-in-the-tail axis -- large-cap vs tech vs "
        "small-cap vs Dow). ASYMMETRY = downside dispersion minus the "
        "analogous upside (tau=0.90) dispersion. Evaluated on a "
        "fixed-from-start stride-5 grid and forward-filled for causal "
        "stability under truncation. Per-ticker proxy; faithful IRLS "
        "expectile-regression implementation of the asymmetric-least-"
        "squares beta definition."
    ),
    "requires": ["Close"],
    "produces": [COL_LEVEL, COL_DISP, COL_ASYM],
    "tags": ["beta", "expectile", "downside_risk", "dispersion", "cross_index", "rolling"],
    "version": "1.0.0",
    "author": (
        "feature-factory ff0703c (proxy: IRLS asymmetric-least-squares "
        "expectile beta on stride-5 causal grid, forward-filled)"
    ),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_WINDOW = 120
_STRIDE = 5
_MIN_OBS = 60          # minimum finite (x,y) pairs required in the window
_N_ITER = 8            # IRLS iterations
_TAU_DOWN = 0.10
_TAU_UP = 0.90
_INDEX_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")


def _irls_expectile_beta(x: np.ndarray, y: np.ndarray, tau: float, n_iter: int) -> float:
    """
    Asymmetric-least-squares (expectile) beta of y on x via IRLS.
    Weight w_i = tau if residual_i >= 0 else (1 - tau); refit weighted OLS
    with intercept each iteration. Returns np.nan on a degenerate window.
    """
    n = x.shape[0]
    if n < _MIN_OBS:
        return np.nan
    if np.std(x) < 1e-12:
        return np.nan

    w = np.full(n, 0.5)
    beta = np.nan
    for _ in range(n_iter):
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 1e-10:
            return np.nan
        xw = np.sum(w * x) / sw
        yw = np.sum(w * y) / sw
        dx = x - xw
        dy = y - yw
        varw = np.sum(w * dx * dx)
        if not np.isfinite(varw) or varw < 1e-14:
            return np.nan
        beta = float(np.sum(w * dx * dy) / varw)
        alpha = yw - beta * xw
        resid = y - (alpha + beta * x)
        w = np.where(resid >= 0.0, tau, 1.0 - tau)

    if not np.isfinite(beta):
        return np.nan
    return beta


def _rolling_expectile_beta_grid(
    stock_ret: np.ndarray,
    idx_ret: np.ndarray,
    window: int,
    stride: int,
    tau: float,
    min_obs: int,
    n_iter: int,
) -> np.ndarray:
    """
    Evaluate the IRLS expectile beta only on a fixed-from-start grid
    (i % stride == 0), forward-filled elsewhere. Causal: window at row i
    uses only rows [i-window+1, i].
    """
    n = stock_ret.shape[0]
    out = np.full(n, np.nan)

    for i in range(window - 1, n):
        if i % stride != 0:
            continue
        x_win = idx_ret[i - window + 1 : i + 1]
        y_win = stock_ret[i - window + 1 : i + 1]
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if mask.sum() < min_obs:
            continue
        out[i] = _irls_expectile_beta(x_win[mask], y_win[mask], tau, n_iter)

    return pd.Series(out).ffill().to_numpy()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns on every code path.
    df[COL_LEVEL] = np.nan
    df[COL_DISP] = np.nan
    df[COL_ASYM] = np.nan

    n = len(df)
    if n < _WINDOW + 1:
        return df

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    stock_ret = np.log(work["Close"] / work["Close"].shift(1)).to_numpy(dtype=float)

    down_betas = []  # list of (n,) arrays, one per available index
    up_betas = []

    for sym in _INDEX_SYMBOLS:
        try:
            idx_close = _indexes.index_close(sym)
        except Exception:
            continue
        if idx_close is None or idx_close.empty:
            continue

        idx_df = idx_close.rename("idx_close").reset_index()
        idx_df.columns = ["Date", "idx_close"]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])
        idx_df = idx_df.sort_values("Date")
        idx_df["idx_ret"] = np.log(idx_df["idx_close"] / idx_df["idx_close"].shift(1))

        merged = pd.merge_asof(
            work[["Date"]].sort_values("Date"),
            idx_df[["Date", "idx_ret"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )
        merged = merged.set_index(work.index)
        idx_ret = merged["idx_ret"].to_numpy(dtype=float)

        down_beta = _rolling_expectile_beta_grid(
            stock_ret, idx_ret, _WINDOW, _STRIDE, _TAU_DOWN, _MIN_OBS, _N_ITER
        )
        up_beta = _rolling_expectile_beta_grid(
            stock_ret, idx_ret, _WINDOW, _STRIDE, _TAU_UP, _MIN_OBS, _N_ITER
        )

        down_betas.append(down_beta)
        up_betas.append(up_beta)

    if len(down_betas) < 2:
        # Need at least 2 indexes for a meaningful dispersion measure.
        return df

    down_stack = np.vstack(down_betas)   # shape (k, n), k in [2,4]
    up_stack = np.vstack(up_betas)

    with np.errstate(invalid="ignore"):
        down_count = np.sum(np.isfinite(down_stack), axis=0)
        up_count = np.sum(np.isfinite(up_stack), axis=0)

        level = np.where(down_count >= 2, np.nanmean(down_stack, axis=0), np.nan)
        disp_down = np.where(down_count >= 2, np.nanstd(down_stack, axis=0), np.nan)
        disp_up = np.where(up_count >= 2, np.nanstd(up_stack, axis=0), np.nan)

    asym = np.where(
        np.isfinite(disp_down) & np.isfinite(disp_up),
        disp_down - disp_up,
        np.nan,
    )

    df[COL_LEVEL] = level
    df[COL_DISP] = disp_down
    df[COL_ASYM] = asym

    return df
