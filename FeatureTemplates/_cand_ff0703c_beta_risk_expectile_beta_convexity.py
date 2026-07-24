"""
Expectile-regression beta convexity feature block.

Over a rolling 120-day window, fits asymmetric-least-squares (expectile) regressions
of the stock's daily return on SPY's daily return at tau = 0.10, 0.50, 0.90 via
iteratively-reweighted least squares (IRLS). The tau=0.50 slope is the central
(robust) market beta. The curvature term

    convexity = beta_0.10 + beta_0.90 - 2 * beta_0.50

is positive when the stock's comovement with SPY intensifies in BOTH tails
(fat-tail co-loading, i.e. beta rises symmetrically in up- and down-crashes
relative to the middle of the distribution) and negative when the central mass
of the joint distribution loads more than the tails. This isolates a
second-order (curvature) tail-comovement structure that is orthogonal to any
single directional beta estimate.

Per-ticker OHLCV-based proxy using SPY via the _indexes helper (no cross-sectional
regression is available inside a single-ticker block, so the "market" side of the
regression is SPY's own daily return series, which is the standard single-index
proxy for beta estimation).

Causal striding: the expensive IRLS fit is only evaluated on a fixed-from-start
grid (i % 5 == 0) and forward-filled between grid points, so the causality gate
(truncation invariance) holds exactly.
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
    "name": "ff0703c_beta_risk_expectile_beta_convexity",
    "description": (
        "Rolling 120-day expectile-regression (IRLS asymmetric least squares) betas "
        "of stock return on SPY return at tau=0.10/0.50/0.90. "
        "ff0703c_beta_risk_expectile_beta_convexity_mid = beta_0.50 (central expectile "
        "beta, robust market beta level). "
        "ff0703c_beta_risk_expectile_beta_convexity_curv = beta_0.10 + beta_0.90 - "
        "2*beta_0.50 (curvature: positive = both-tail comovement intensification, "
        "negative = center-loaded comovement). "
        "ff0703c_beta_risk_expectile_beta_convexity_dyn = 60-day change of the "
        "curvature term (regime shift in tail-comovement shape). "
        "Per-ticker proxy: single-index (SPY) expectile beta rather than a "
        "cross-sectional multi-factor expectile regression, computed on a "
        "fixed-from-start 5-day grid and forward-filled for causal speed; "
        "zero-variance windows guarded to NaN."
    ),
    "requires": ["Close"],
    "produces": [
        "ff0703c_beta_risk_expectile_beta_convexity_mid",
        "ff0703c_beta_risk_expectile_beta_convexity_curv",
        "ff0703c_beta_risk_expectile_beta_convexity_dyn",
    ],
    "tags": ["beta-risk", "beta", "tail-risk", "expectile", "spy", "convexity", "ff0703c"],
    "version": "1.0.0",
    "author": (
        "ff0703c batch spec author. Faithful per-ticker single-index proxy: "
        "expectile IRLS regression of stock ret on SPY ret (economic signal = "
        "curvature of tail comovement) since a true cross-sectional expectile "
        "regression is not available inside a single-ticker OHLCV block."
    ),
}

_WINDOW = 120
_STRIDE = 5
_DYN_LAG = 60
_TAUS = (0.10, 0.50, 0.90)
_N_IRLS = 8
_MIN_VALID = 30


def _expectile_slope(x: np.ndarray, y: np.ndarray, tau: float) -> float:
    """Return the slope (beta) of an expectile (tau) regression y ~ a + b*x via IRLS."""
    n = x.shape[0]
    if n < _MIN_VALID:
        return np.nan
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return np.nan

    X = np.column_stack([np.ones(n), x])

    # Initialise with OLS (tau=0.5 IRLS fixed point)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    for _ in range(_N_IRLS):
        resid = y - X @ beta
        w = np.where(resid >= 0.0, tau, 1.0 - tau)
        Xw = X * w[:, None]
        XtWX = X.T @ Xw
        XtWy = X.T @ (w * y)
        try:
            new_beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return np.nan
        if not np.all(np.isfinite(new_beta)):
            return np.nan
        beta = new_beta

    return float(beta[1])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expectile-regression beta convexity vs SPY."""
    n = len(df)

    mid_col = "ff0703c_beta_risk_expectile_beta_convexity_mid"
    curv_col = "ff0703c_beta_risk_expectile_beta_convexity_curv"
    dyn_col = "ff0703c_beta_risk_expectile_beta_convexity_dyn"

    df[mid_col] = np.nan
    df[curv_col] = np.nan
    df[dyn_col] = np.nan

    if n < _WINDOW + 2:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close and align to this stock's dates (lookahead-safe)
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
    # 2. Daily returns (causal, no negative shift)
    # ------------------------------------------------------------------
    close_arr = df["Close"].values.astype(np.float64)

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
    # 3. Fixed-from-start grid (i % stride == 0), IRLS expectile fits
    # ------------------------------------------------------------------
    curv_grid = np.full(n, np.nan)
    mid_grid = np.full(n, np.nan)

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

        betas = {}
        ok = True
        for tau in _TAUS:
            b = _expectile_slope(ws, wk, tau)
            if np.isnan(b):
                ok = False
                break
            betas[tau] = b
        if not ok:
            continue

        b10, b50, b90 = betas[0.10], betas[0.50], betas[0.90]
        mid_grid[t] = b50
        curv_grid[t] = b10 + b90 - 2.0 * b50

    # ------------------------------------------------------------------
    # 4. Forward-fill grid values (causal)
    # ------------------------------------------------------------------
    mid_s = pd.Series(mid_grid).where(~np.isnan(mid_grid)).ffill()
    curv_s = pd.Series(curv_grid).where(~np.isnan(curv_grid)).ffill()

    mid_out = mid_s.values
    curv_out = curv_s.values

    # ------------------------------------------------------------------
    # 5. Dynamic = 60-day change of curvature (on the forward-filled series)
    # ------------------------------------------------------------------
    dyn_out = np.full(n, np.nan)
    dyn_out[_DYN_LAG:] = curv_out[_DYN_LAG:] - curv_out[:-_DYN_LAG]

    # ------------------------------------------------------------------
    # 6. Guard inf, assign
    # ------------------------------------------------------------------
    mid_out = np.where(np.isinf(mid_out), np.nan, mid_out)
    curv_out = np.where(np.isinf(curv_out), np.nan, curv_out)
    dyn_out = np.where(np.isinf(dyn_out), np.nan, dyn_out)

    df[mid_col] = mid_out
    df[curv_col] = curv_out
    df[dyn_col] = dyn_out

    return df
