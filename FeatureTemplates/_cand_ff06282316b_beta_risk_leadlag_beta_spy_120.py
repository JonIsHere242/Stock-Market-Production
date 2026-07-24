"""
ff06282316b_beta_risk_leadlag_beta_spy_120
------------------------------------------
Lag-beta vs sync-beta divergence over a trailing 120-day window.

Two betas are computed per ticker against SPY:
  - sync_beta : OLS slope of ticker_ret ~ SPY_ret (same day)
  - lag_beta  : OLS slope of ticker_ret ~ SPY_ret_lag1 (yesterday's SPY move)

Feature = (lag_beta - sync_beta) / (|sync_beta| + eps)

A high positive value indicates the stock is a "slow-information" name that
reacts more to yesterday's index movement than today's -- capturing lead-lag
risk structure distinct from simple beta or realized vol.

Additional produced columns:
  - _sync_beta : the contemporaneous beta itself (useful standalone)
  - _lag_beta  : the lagged beta itself
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never package import)
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
    "name": "ff06282316b_beta_risk_leadlag_beta_spy_120",
    "description": (
        "Lead-lag beta divergence: trailing-120 lag_beta (ticker vs yesterday SPY) "
        "minus sync_beta (ticker vs today SPY), normalised by |sync_beta|+eps. "
        "High positive = slow-information stock reacting to prior-day index moves. "
        "Also produces raw sync_beta and lag_beta."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_risk_leadlag_beta_spy_120",
        "ff06282316b_beta_risk_leadlag_beta_spy_120_sync_beta",
        "ff06282316b_beta_risk_leadlag_beta_spy_120_lag_beta",
    ],
    "tags": ["beta", "lead_lag", "market", "spy", "risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120
_EPS = 1e-8
_MIN_PERIODS = 30  # require at least 30 obs for a meaningful regression

# ---------------------------------------------------------------------------
# Rolling OLS helper (vectorised with numpy sliding window)
# ---------------------------------------------------------------------------

def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Return rolling OLS beta (slope of y ~ x) without intercept shift.

    Uses the identity: beta = cov(y,x) / var(x) to stay fully vectorised.
    Both y and x must be 1-D arrays of equal length (NaN-aligned before call).
    """
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)

    # Build rolling sums via cumsum (O(n), no Python loop over each row)
    # We need: sum_x, sum_y, sum_xx, sum_xy, count_valid
    # Treat NaN pairs as missing: mask them out together.
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x_m = np.where(valid, x, 0.0)
    y_m = np.where(valid, y, 0.0)
    xx_m = np.where(valid, x * x, 0.0)
    xy_m = np.where(valid, x * y, 0.0)
    cnt_m = valid.astype(np.float64)

    # Pad one zero at start for diff-based rolling sum
    def _rollsum(arr: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        result = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i + 1 - window)
            result[i] = cs[i + 1] - cs[start]
        return result

    # Avoid pure-Python loop: use stride-tricks for the window sums
    # For n~700, window=120: vectorised cumsum diff is fast enough
    cs_x  = np.concatenate([[0.0], np.cumsum(x_m)])
    cs_y  = np.concatenate([[0.0], np.cumsum(y_m)])
    cs_xx = np.concatenate([[0.0], np.cumsum(xx_m)])
    cs_xy = np.concatenate([[0.0], np.cumsum(xy_m)])
    cs_c  = np.concatenate([[0.0], np.cumsum(cnt_m)])

    for i in range(n):
        start = max(0, i + 1 - window)
        cnt = cs_c[i + 1] - cs_c[start]
        if cnt < min_periods:
            continue
        sx  = cs_x[i + 1]  - cs_x[start]
        sy  = cs_y[i + 1]  - cs_y[start]
        sxx = cs_xx[i + 1] - cs_xx[start]
        sxy = cs_xy[i + 1] - cs_xy[start]
        # OLS slope: (n*sxy - sx*sy) / (n*sxx - sx*sx)
        denom = cnt * sxx - sx * sx
        if abs(denom) < _EPS:
            continue
        out[i] = (cnt * sxy - sx * sy) / denom

    return out


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN (required on every code path)
    col_main  = "ff06282316b_beta_risk_leadlag_beta_spy_120"
    col_sync  = "ff06282316b_beta_risk_leadlag_beta_spy_120_sync_beta"
    col_lag   = "ff06282316b_beta_risk_leadlag_beta_spy_120_lag_beta"
    df[col_main] = np.nan
    df[col_sync] = np.nan
    df[col_lag]  = np.nan

    if len(df) < _MIN_PERIODS + 2:
        return df

    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Merge SPY returns onto df (backward-safe; direction="backward")
    spy_ret = spy_close.pct_change()
    spy_df = spy_ret.rename("spy_ret").reset_index()
    spy_df.columns = ["Date", "spy_ret"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    work = work.set_index(df.index)

    spy_ret_today = work["spy_ret"].to_numpy(dtype=np.float64)
    spy_ret_lag1  = np.concatenate([[np.nan], spy_ret_today[:-1]])

    # Ticker returns
    ticker_ret = df["Close"].pct_change().to_numpy(dtype=np.float64)

    # Rolling betas
    sync_beta = _rolling_beta(ticker_ret, spy_ret_today, _WINDOW, _MIN_PERIODS)
    lag_beta  = _rolling_beta(ticker_ret, spy_ret_lag1,  _WINDOW, _MIN_PERIODS)

    # Normalised divergence
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.abs(sync_beta) + _EPS
        divergence = (lag_beta - sync_beta) / denom

    # Guard inf/-inf (shouldn't occur given _EPS, but be safe)
    divergence = np.where(np.isfinite(divergence), divergence, np.nan)
    sync_beta  = np.where(np.isfinite(sync_beta),  sync_beta,  np.nan)
    lag_beta   = np.where(np.isfinite(lag_beta),   lag_beta,   np.nan)

    df[col_main] = divergence
    df[col_sync] = sync_beta
    df[col_lag]  = lag_beta

    return df
