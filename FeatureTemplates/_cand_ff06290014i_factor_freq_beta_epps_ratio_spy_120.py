"""
Epps-effect beta ratio vs SPY.

Short-horizon beta (1-day) divided by long-horizon beta (5-day overlapping
log-returns). The Epps effect predicts that measured co-movement shrinks as
the return horizon shortens due to asynchronous trading. The ratio
beta_5d / beta_1d captures how much beta "builds up" as horizon lengthens.
Values near 1 mean symmetric co-movement at all horizons; values <1 mean
short-horizon co-movement dominates (unusual); values >1 mean beta grows
with horizon (classic Epps regime). Per-ticker proxy; causal, no lookahead.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: SPY index close
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
    "name": "ff06290014i_factor_freq_beta_epps_ratio_spy_120",
    "description": (
        "Epps-effect beta ratio: rolling beta(5-day overlapping log-returns) / "
        "beta(1-day log-returns) minus 1. Captures whether co-movement with SPY "
        "grows with horizon (classic Epps regime). Computed over a 120-bar lookback "
        "with a 5-day overlapping window for low-frequency beta. Guard for near-zero "
        "beta_1d uses max(|beta_1d|, 1e-3). Per-ticker proxy; causal/no-lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_factor_freq_beta_epps_ratio_spy_120_ratio",   # beta_5d/beta_1d - 1
        "ff06290014i_factor_freq_beta_epps_ratio_spy_120_b1d",     # rolling 1-day beta vs SPY
        "ff06290014i_factor_freq_beta_epps_ratio_spy_120_b5d",     # rolling 5-day beta vs SPY
    ],
    "tags": ["beta", "epps", "frequency", "market", "factor"],
    "version": "1.0.0",
    "author": "feature-factory ff06290014i",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120      # lookback for rolling covariance estimation
_HORIZON5 = 5      # multi-day return horizon for low-frequency beta
_MIN_PERIODS = 30  # minimum observations before emitting a value
_EPS_VAR = 1e-12   # floor for variance guard
_EPS_BETA = 1e-3   # floor for |beta_1d| denominator guard


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Epps-effect beta ratio features."""
    col_ratio = "ff06290014i_factor_freq_beta_epps_ratio_spy_120_ratio"
    col_b1d   = "ff06290014i_factor_freq_beta_epps_ratio_spy_120_b1d"
    col_b5d   = "ff06290014i_factor_freq_beta_epps_ratio_spy_120_b5d"

    # Initialise output columns to NaN on every code path
    df[col_ratio] = np.nan
    df[col_b1d]   = np.nan
    df[col_b5d]   = np.nan

    if len(df) < _MIN_PERIODS + _HORIZON5:
        return df

    # ------------------------------------------------------------------
    # 1. Fetch SPY close; merge backward (lookahead-safe)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("_spy_close").reset_index()
    spy_df.columns = ["Date", "_spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    df_work = df[["Date", "Close"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = pd.merge_asof(
        df_work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    df_work = df_work.set_index(df.index if df_work.shape[0] == df.shape[0] else df_work.index)

    stock_close = df_work["Close"].values.astype(float)
    spy_close_arr = df_work["_spy_close"].values.astype(float)

    n = len(stock_close)

    # ------------------------------------------------------------------
    # 2. 1-day log returns (length n-1, aligned to bars 1..n-1)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1_stock = np.where(
            (stock_close[:-1] > 0) & (spy_close_arr[:-1] > 0),
            np.log(stock_close[1:] / stock_close[:-1]),
            np.nan,
        )
        r1_spy = np.where(
            (stock_close[:-1] > 0) & (spy_close_arr[:-1] > 0),
            np.log(spy_close_arr[1:] / spy_close_arr[:-1]),
            np.nan,
        )

    # ------------------------------------------------------------------
    # 3. 5-day overlapping log returns (bar i -> i+4 mapped to bar i+4)
    # ------------------------------------------------------------------
    r5_stock = np.full(n, np.nan)
    r5_spy   = np.full(n, np.nan)
    for i in range(_HORIZON5 - 1, n):
        s0 = stock_close[i - _HORIZON5 + 1]
        s1 = stock_close[i]
        m0 = spy_close_arr[i - _HORIZON5 + 1]
        m1 = spy_close_arr[i]
        if s0 > 0 and s1 > 0 and m0 > 0 and m1 > 0:
            r5_stock[i] = np.log(s1 / s0)
            r5_spy[i]   = np.log(m1 / m0)

    # ------------------------------------------------------------------
    # 4. Rolling beta via cov/var over _WINDOW bars
    #    beta_1d uses the 1-day returns series (offset by 1 bar)
    #    beta_5d uses the 5-day overlapping series
    # ------------------------------------------------------------------
    # We compute rolling cov and var using pandas for convenience.
    # Both series are indexed on bar position; we assign back by position.

    def _rolling_beta(r_stock: np.ndarray, r_market: np.ndarray) -> np.ndarray:
        """Rolling OLS beta = cov(r_s, r_m) / var(r_m) over _WINDOW."""
        s = pd.Series(r_stock)
        m = pd.Series(r_market)
        # joint valid mask handled by pandas (NaN propagated automatically)
        cov = s.rolling(_WINDOW, min_periods=_MIN_PERIODS).cov(m)
        var = m.rolling(_WINDOW, min_periods=_MIN_PERIODS).var()
        beta = np.where(var.values > _EPS_VAR, cov.values / var.values, np.nan)
        return beta

    # 1-day beta: returns are offset by 1 (r1_stock has length n-1)
    # pad front with NaN so index aligns to bar position 1..n-1
    beta1d_raw = _rolling_beta(r1_stock, r1_spy)   # length n-1
    beta1d = np.full(n, np.nan)
    beta1d[1:] = beta1d_raw

    # 5-day beta: r5 already has length n, aligned to bar position 0..n-1
    beta5d = _rolling_beta(r5_stock, r5_spy)        # length n

    # ------------------------------------------------------------------
    # 5. Ratio: beta_5d / beta_1d - 1  (guard near-zero beta_1d)
    # ------------------------------------------------------------------
    sign1d = np.sign(beta1d)
    sign1d[sign1d == 0] = 1.0
    denom = np.where(
        np.isfinite(beta1d),
        np.maximum(np.abs(beta1d), _EPS_BETA) * sign1d,
        np.nan,
    )
    ratio = np.where(
        np.isfinite(beta5d) & np.isfinite(denom),
        beta5d / denom - 1.0,
        np.nan,
    )

    # Replace inf/-inf with NaN (safety)
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    beta1d = np.where(np.isfinite(beta1d), beta1d, np.nan)
    beta5d = np.where(np.isfinite(beta5d), beta5d, np.nan)

    # ------------------------------------------------------------------
    # 6. Write back; df_work was sorted by Date, restore original index order
    # ------------------------------------------------------------------
    # df_work index may have been reset; use positional assignment via iloc
    # We need to map back to original df order.
    # df_work was sorted by Date; df may already be sorted ascending (guaranteed
    # by caller), so positional alignment is safe when shapes match.
    if df_work.shape[0] == df.shape[0]:
        df.iloc[:, df.columns.get_loc(col_ratio)] = ratio
        df.iloc[:, df.columns.get_loc(col_b1d)]   = beta1d
        df.iloc[:, df.columns.get_loc(col_b5d)]   = beta5d
    # If shapes mismatch (degenerate), columns remain NaN (already initialised).

    return df
