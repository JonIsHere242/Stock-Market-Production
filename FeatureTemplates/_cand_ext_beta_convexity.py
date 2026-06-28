"""
ext_beta_convexity: Beta convexity via quadratic market-timing OLS.

Rolling 150-day OLS: stock_ret ~ beta * spy_ret + convexity * spy_ret^2
The convexity coefficient captures whether the stock profits more than linearly
when SPY rips (positive = beneficial convexity) or bleeds more than linearly
on drops (negative = concavity/crash-prone). The linear beta is also emitted
from the same fit so both components can enter the model together.

This is orthogonal to xdom2_downside_beta (asymmetric split-sample beta):
that feature conditions on sign of SPY; this feature uses the quadratic term
on the full distribution -- different statistical structure and economic axis.

Author: Extension/exploration of gate-validated winner (xdom2_downside_beta).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY close) -- by file path, never via package import
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_beta_convexity",
    "description": (
        "Rolling 150-day quadratic OLS of stock daily return on SPY daily return "
        "and SPY^2. The coefficient on SPY^2 is beta convexity (positive = "
        "beneficial convexity, negative = crash-prone concavity). Also emits the "
        "linear beta from the same fit and a 20/60 day change in convexity. "
        "Orthogonal to xdom2_downside_beta (which uses sign-split OLS; this uses "
        "the squared term on the full distribution). Per-ticker proxy -- no "
        "cross-sectional ranking."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_beta_convexity_coef",   # quadratic (convexity) coefficient
        "ext_beta_convexity_beta",   # linear beta from same fit
        "ext_beta_convexity_chg20",  # 20-day change in convexity (momentum)
    ],
    "tags": ["beta", "market-timing", "convexity", "quadratic", "regression"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner (xdom2_downside_beta). "
        "Quadratic market-timing term from Merton (1981) / Henriksson-Merton "
        "market-timing literature."
    ),
}

# ---------------------------------------------------------------------------
# Rolling 150-day quadratic OLS helper using sliding_window_view
# ---------------------------------------------------------------------------
_WINDOW = 150
_MIN_OBS = 60   # require at least 60 valid obs before emitting a value


def _rolling_quad_ols(
    y: np.ndarray,   # stock returns, length N
    x: np.ndarray,   # SPY returns, length N  (aligned, may contain nan)
    window: int,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each position t, regress y[t-window+1:t+1] on [1, x, x^2] via OLS.
    Returns (beta_linear, beta_quad) arrays of length N (leading entries = nan).

    Vectorised: forms the 3-column design matrix for each window as a
    (N - window + 1, window, 3) array, then solves via lstsq in a loop over
    windows. This avoids O(n^2) per-row Python and remains < 100ms for n=700.
    """
    N = len(y)
    beta_lin = np.full(N, np.nan, dtype=np.float64)
    beta_quad = np.full(N, np.nan, dtype=np.float64)

    for t in range(window - 1, N):
        yw = y[t - window + 1 : t + 1]
        xw = x[t - window + 1 : t + 1]
        # drop rows where either series is nan
        mask = np.isfinite(yw) & np.isfinite(xw)
        if mask.sum() < min_obs:
            continue
        yw_ = yw[mask]
        xw_ = xw[mask]
        # design matrix: intercept, x, x^2
        A = np.column_stack([np.ones(len(xw_)), xw_, xw_ ** 2])
        # lstsq: (intercept, beta, gamma)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, yw_, rcond=None)
        except np.linalg.LinAlgError:
            continue
        beta_lin[t] = coeffs[1]
        beta_quad[t] = coeffs[2]

    return beta_lin, beta_quad


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Stock returns
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    stock_ret = np.empty_like(close)
    stock_ret[0] = np.nan
    stock_ret[1:] = np.diff(np.log(np.where(close > 0, close, np.nan)))

    # ------------------------------------------------------------------
    # 2. SPY returns aligned to df dates via merge_asof
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        df_dates = df[["Date"]].copy()
        df_dates["Date"] = pd.to_datetime(df_dates["Date"])
        merged = pd.merge_asof(
            df_dates.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original row order
        merged = merged.set_index(df_dates.sort_values("Date").index)
        spy_c = merged["spy_close"].values.astype(np.float64)
    except Exception:
        spy_c = np.full(len(close), np.nan)

    spy_ret = np.empty_like(spy_c)
    spy_ret[0] = np.nan
    spy_ret[1:] = np.diff(np.log(np.where(spy_c > 0, spy_c, np.nan)))

    # ------------------------------------------------------------------
    # 3. Rolling quadratic OLS
    # ------------------------------------------------------------------
    beta_lin, beta_quad = _rolling_quad_ols(stock_ret, spy_ret, _WINDOW, _MIN_OBS)

    # ------------------------------------------------------------------
    # 4. Guard inf/-inf (shouldn't occur from lstsq but be safe)
    # ------------------------------------------------------------------
    beta_lin = np.where(np.isfinite(beta_lin), beta_lin, np.nan)
    beta_quad = np.where(np.isfinite(beta_quad), beta_quad, np.nan)

    # ------------------------------------------------------------------
    # 5. 20-day momentum on convexity coefficient
    # ------------------------------------------------------------------
    chg20 = np.full(len(beta_quad), np.nan)
    chg20[20:] = beta_quad[20:] - beta_quad[:-20]
    chg20 = np.where(np.isfinite(chg20), chg20, np.nan)

    # ------------------------------------------------------------------
    # 6. Assign back (preserve original index order)
    # ------------------------------------------------------------------
    df["ext_beta_convexity_coef"] = beta_quad
    df["ext_beta_convexity_beta"] = beta_lin
    df["ext_beta_convexity_chg20"] = chg20

    return df
