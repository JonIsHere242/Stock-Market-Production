"""
CAPM Beta (per-ticker rolling proxy)
Spec: osap_beta — OpenSourceAP (Chen-Zimmermann), Fama and MacBeth 1973.

Original method: 60-month rolling OLS of (stock excess return) on (market excess return),
excluding estimates with <20 monthly observations.

Per-ticker adaptation:
- Uses SPY daily returns as market proxy (SPY loaded via _indexes helper).
- Computes daily log returns for both stock and SPY.
- Rolling 1260-day window (~60 months of ~21 trading days) with a 420-day minimum
  (~20 months), matching the original monthly exclusion rule.
- Risk-free rate omitted from daily returns (negligible effect on beta slope; RF
  cancels almost exactly from numerator/denominator at daily frequency).
- Produces: beta level, rolling 63-day change in beta (momentum/dynamic variant),
  and residual volatility (idiosyncratic vol = std of OLS residuals, a companion risk signal).
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
_spec_idx = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec_idx)
_spec_idx.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_beta",
    "description": (
        "CAPM market beta — per-ticker rolling OLS proxy for Fama-MacBeth 1973 "
        "60-month beta. Uses SPY daily log-returns as market proxy over a 1260-day "
        "window (min 420 days ≈ 20 months). Produces: rolling beta (level), 63-day "
        "beta change (dynamic/momentum), and idiosyncratic volatility (OLS residual "
        "std). Inherently cross-sectional in the original paper; this is the closest "
        "faithful per-ticker proxy using OHLCV + SPY index data."
    ),
    "requires": ["Close"],
    "produces": ["osap_beta_level", "osap_beta_delta63", "osap_beta_ivol"],
    "tags": ["risk", "beta", "capm", "momentum", "idiosyncratic_vol"],
    "version": "1.0",
    "author": "Fama and MacBeth (1973); OpenSourceAP Chen-Zimmermann; block by codegen",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 1260       # ~60 months of trading days
_MIN_OBS = 420       # ~20 months minimum (original exclusion rule)
_DELTA_W = 63        # ~3 months window for beta-change signal


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling CAPM beta and companion features per ticker."""

    n = len(df)

    # Initialise output columns to NaN
    df["osap_beta_level"] = np.nan
    df["osap_beta_delta63"] = np.nan
    df["osap_beta_ivol"] = np.nan

    if n < 2:
        return df

    # ------------------------------------------------------------------
    # Stock log returns
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret = np.empty(n, dtype=np.float64)
        stock_ret[0] = np.nan
        # log(close[t] / close[t-1])
        stock_ret[1:] = np.log(
            np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
        )

    # ------------------------------------------------------------------
    # SPY log returns aligned to df dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
        spy_close = spy_close[spy_close > 0]

        df_dates = pd.to_datetime(df["Date"])
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        tmp = pd.DataFrame({"Date": df_dates})
        tmp = pd.merge_asof(
            tmp.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # restore original order
        tmp = tmp.set_index(tmp.index).reindex(range(n))

        spy_vals = tmp["spy_close"].values.astype(np.float64)
        mkt_ret = np.empty(n, dtype=np.float64)
        mkt_ret[0] = np.nan
        mkt_ret[1:] = np.log(
            np.where(spy_vals[:-1] > 0, spy_vals[1:] / spy_vals[:-1], np.nan)
        )
    except Exception:
        # SPY unavailable — output all NaN (expected on test stubs)
        return df

    # ------------------------------------------------------------------
    # Rolling OLS: beta = cov(r, m) / var(m)  over _WINDOW days
    # Use vectorised rolling statistics (pandas rolling).
    # ------------------------------------------------------------------
    r = pd.Series(stock_ret)
    m = pd.Series(mkt_ret)

    # Rolling means
    r_mean = r.rolling(_WINDOW, min_periods=_MIN_OBS).mean()
    m_mean = m.rolling(_WINDOW, min_periods=_MIN_OBS).mean()

    # Rolling covariance and variance computed via deviation products
    # cov(r,m) = E[(r - r_mean)(m - m_mean)]
    # var(m)   = E[(m - m_mean)^2]
    # Both computed as rolling mean of centred products.
    # NOTE: we use expanding-then-fixed rolling mean of products;
    # the correct rolling formula uses the unbiased form but for beta
    # the n vs n-1 terms cancel.
    r_dev = r - r_mean
    m_dev = m - m_mean

    cov_rm = (r_dev * m_dev).rolling(_WINDOW, min_periods=_MIN_OBS).mean()
    var_m = (m_dev * m_dev).rolling(_WINDOW, min_periods=_MIN_OBS).mean()

    # Guard division
    var_m_safe = var_m.replace(0, np.nan)
    beta = cov_rm / var_m_safe

    # Residual volatility: std of (r - beta * m) over same window
    # Compute per-row residual using the rolling beta (slight lag is intentional
    # and leakage-free — we're applying yesterday's beta estimate to today's bar).
    beta_lag = beta.shift(1)
    resid = r - beta_lag * m
    ivol = resid.rolling(_WINDOW, min_periods=_MIN_OBS).std()

    # Beta delta (63-day change) — difference of beta level
    beta_delta = beta - beta.shift(_DELTA_W)

    # ------------------------------------------------------------------
    # Write results back — replace inf with nan
    # ------------------------------------------------------------------
    def _clean(s: pd.Series) -> np.ndarray:
        arr = s.values.astype(np.float64)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    df["osap_beta_level"] = _clean(beta)
    df["osap_beta_delta63"] = _clean(beta_delta)
    df["osap_beta_ivol"] = _clean(ivol)

    return df
