"""
Price Delay features (Hou & Moskowitz 2005) — all three variants:
  osap_pricedelay_rsq    : 1 - R²(no-lag) / R²(with 4 lags)
  osap_pricedelay_slope  : weighted-lag-beta / total-beta ratio
  osap_pricedelay_tstat  : weighted-lag-tstat / total-tstat ratio

Implemented as a rolling 252-day regression per-ticker using SPY as the
market-return proxy (the original uses CRSP value-weight market excess return,
which is unavailable; SPY daily return is the faithful OHLCV+index proxy).

Original method: annual July regressions across all stocks simultaneously.
Per-ticker adaptation: same regression structure, sliding 252-day window,
updated daily.  Cross-sectional ranking is done by the downstream predictor.

Source: OpenSourceAP (Chen-Zimmermann); Hou and Moskowitz (2005).
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY market returns)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_pricedelay",
    "description": (
        "Price Delay measures of Hou and Moskowitz (2005): how slowly a stock's "
        "price responds to market-wide information. Three variants: R-squared "
        "ratio (osap_pricedelay_rsq), slope-weighted lag-beta ratio "
        "(osap_pricedelay_slope), and t-stat-weighted lag ratio "
        "(osap_pricedelay_tstat). Computed via rolling 252-day OLS of stock "
        "return on contemporaneous + 4 lagged SPY returns (proxy for market "
        "return). Cross-sectional ranking is left to the downstream predictor. "
        "Per-ticker OHLCV+SPY proxy; original uses annual July CRSP regressions "
        "across all stocks."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_pricedelay_rsq",
        "osap_pricedelay_slope",
        "osap_pricedelay_tstat",
    ],
    "tags": ["lead_lag", "market_efficiency", "price_delay", "regression", "osap"],
    "version": "1.0",
    "author": "Hou and Moskowitz (2005) via OpenSourceAP (Chen-Zimmermann); per-ticker proxy impl.",
}

# ---------------------------------------------------------------------------
# Rolling OLS helper — vectorised via numpy sliding windows
# ---------------------------------------------------------------------------
_WIN = 252   # ~1 trading year
_NLAGS = 4   # number of market return lags


def _ols_stats(y: np.ndarray, X: np.ndarray):
    """
    Tiny OLS: return (betas, se_betas, rsquared).
    X already has intercept column prepended.
    Returns (betas, se, rsq); all NaN on failure.
    """
    n, k = X.shape
    try:
        XtX = X.T @ X
        Xty = X.T @ y
        betas = np.linalg.solve(XtX, Xty)
        y_hat = X @ betas
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        rsq = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else np.nan
        sigma2 = ss_res / max(n - k, 1)
        cov = sigma2 * np.linalg.inv(XtX)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return betas, se, rsq
    except (np.linalg.LinAlgError, FloatingPointError):
        nb = np.full(k, np.nan)
        return nb, nb.copy(), np.nan


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # --- market returns from SPY -----------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
    except Exception:
        spy_close = None

    # stock daily log-return
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stk_ret = np.empty(n)
        stk_ret[0] = np.nan
        stk_ret[1:] = np.where(close[:-1] > 0, np.log(close[1:] / close[:-1]), np.nan)

    # market daily log-return aligned to df dates
    if spy_close is not None and len(spy_close) > 0:
        dates = pd.to_datetime(df["Date"])
        spy_df = spy_close.rename("spy").to_frame()
        spy_df.index = pd.to_datetime(spy_df.index)
        merged = pd.merge_asof(
            dates.to_frame(name="Date").reset_index(drop=True),
            spy_df.reset_index().rename(columns={"index": "Date", "Date": "Date"}),
            on="Date",
            direction="backward",
        )
        spy_vals = merged["spy"].values.astype(np.float64)
        mkt_ret = np.empty(n)
        mkt_ret[0] = np.nan
        # spy_vals is Close prices; compute return
        mkt_ret[1:] = np.where(
            spy_vals[:-1] > 0,
            np.log(spy_vals[1:] / spy_vals[:-1]),
            np.nan,
        )
    else:
        # fallback: no SPY available -> all NaN outputs
        df["osap_pricedelay_rsq"] = np.nan
        df["osap_pricedelay_slope"] = np.nan
        df["osap_pricedelay_tstat"] = np.nan
        return df

    # Output arrays
    out_rsq = np.full(n, np.nan)
    out_slope = np.full(n, np.nan)
    out_tstat = np.full(n, np.nan)

    # We need _NLAGS extra rows before the window starts
    # t-th row uses window [t-_WIN+1 .. t]; need lags so effective start is _WIN + _NLAGS
    min_start = _WIN + _NLAGS  # first index where a full window + lags is available

    for t in range(min_start - 1, n):
        # slice for this window (length _WIN)
        s = t - _WIN + 1   # start of window (inclusive)
        # We need indices s .. t for y, and s-_NLAGS .. t for market lags
        # So y = stk_ret[s:t+1], mkt_t = mkt_ret[s:t+1],
        # mkt_lag1 = mkt_ret[s-1:t], ..., mkt_lag4 = mkt_ret[s-4:t-3]
        y = stk_ret[s : t + 1]          # shape (_WIN,)
        mkt0 = mkt_ret[s : t + 1]       # contemporaneous
        lag1 = mkt_ret[s - 1 : t]
        lag2 = mkt_ret[s - 2 : t - 1]
        lag3 = mkt_ret[s - 3 : t - 2]
        lag4 = mkt_ret[s - 4 : t - 3]

        # mask rows where any value is NaN
        valid = (
            np.isfinite(y) & np.isfinite(mkt0) &
            np.isfinite(lag1) & np.isfinite(lag2) &
            np.isfinite(lag3) & np.isfinite(lag4)
        )
        nv = valid.sum()
        if nv < 20:  # need enough obs for stable OLS
            continue

        yv = y[valid]
        m0 = mkt0[valid]
        l1 = lag1[valid]
        l2 = lag2[valid]
        l3 = lag3[valid]
        l4 = lag4[valid]
        ones = np.ones(nv)

        # --- UNRESTRICTED regression: intercept + mkt0 + lag1..lag4 (k=6) ---
        X_full = np.column_stack([ones, m0, l1, l2, l3, l4])
        betas_f, se_f, rsq_full = _ols_stats(yv, X_full)

        # --- RESTRICTED regression: intercept + mkt0 only (k=2) ---
        X_restr = np.column_stack([ones, m0])
        _, _, rsq_restr = _ols_stats(yv, X_restr)

        if not (np.isfinite(rsq_full) and np.isfinite(rsq_restr)):
            continue

        # --- osap_pricedelay_rsq ---
        # = 1 - R²(restricted) / R²(unrestricted)
        # If rsq_full ≈ 0, the ratio is undefined -> NaN
        if abs(rsq_full) > 1e-10:
            out_rsq[t] = 1.0 - rsq_restr / rsq_full
        else:
            out_rsq[t] = np.nan

        # betas: index 0=intercept, 1=mkt0, 2=lag1, 3=lag2, 4=lag3, 5=lag4
        b = betas_f        # length 6
        se = se_f          # length 6
        # t-stats (skip intercept=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstats = np.where(se[1:] > 0, b[1:] / se[1:], np.nan)
        # indices within tstats/b[1:]: 0=mkt0, 1=lag1, 2=lag2, 3=lag3, 4=lag4

        # --- osap_pricedelay_slope ---
        # numerator:   1*b_lag1 + 2*b_lag2 + 3*b_lag3 + 4*b_lag4
        # denominator: b_mkt0 + b_lag1 + b_lag2 + b_lag3 + b_lag4
        b_lags = b[2:]          # [b_lag1, b_lag2, b_lag3, b_lag4]
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        num_slope = np.dot(weights, b_lags)
        denom_slope = b[1] + b_lags.sum()
        if abs(denom_slope) > 1e-12 and np.isfinite(num_slope):
            out_slope[t] = num_slope / denom_slope
        else:
            out_slope[t] = np.nan

        # --- osap_pricedelay_tstat ---
        # numerator:   1*ts_lag1 + 2*ts_lag2 + 3*ts_lag3 + 4*ts_lag4
        # denominator: ts_mkt0 + ts_lag1 + ts_lag2 + ts_lag3 + ts_lag4
        ts = tstats           # [ts_mkt0, ts_lag1, ts_lag2, ts_lag3, ts_lag4]
        num_ts = np.dot(weights, ts[1:])     # weighted lag t-stats
        denom_ts = ts[0] + ts[1:].sum()      # all t-stats
        if np.isfinite(num_ts) and np.isfinite(denom_ts) and abs(denom_ts) > 1e-12:
            out_tstat[t] = num_ts / denom_ts
        else:
            out_tstat[t] = np.nan

    df["osap_pricedelay_rsq"] = out_rsq
    df["osap_pricedelay_slope"] = out_slope
    df["osap_pricedelay_tstat"] = out_tstat
    return df
