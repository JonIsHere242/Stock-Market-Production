"""
ext4_factor_momentum_load — Loading on the leading index (factor momentum).

Identifies which of SPY/QQQ/IWM has the strongest trailing 63-day return
(the "leading style") and computes the stock's rolling 60-day beta to that
leading index and the beta to the lagging index.  The spread captures whether
the stock is loaded on the currently winning factor.

Per-ticker proxy: betas are computed from trailing rolling windows, fully
causal and lookahead-free.  Cross-sectional ranking (e.g. which stocks load
most on the leader) is not performed here; the raw loading itself is the signal.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
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
    "name": "ext4_factor_momentum_load",
    "description": (
        "Factor-momentum loading: identifies the leading style index "
        "(SPY/QQQ/IWM by trailing 63-day return) and the lagging style index, "
        "then computes the stock's rolling 60-day OLS beta to each. "
        "ext4_factor_momentum_load_lead_beta = beta to the winner; "
        "ext4_factor_momentum_load_lag_beta  = beta to the loser; "
        "ext4_factor_momentum_load_spread    = lead_beta - lag_beta. "
        "Per-ticker proxy: no cross-sectional ranking."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_factor_momentum_load_lead_beta",
        "ext4_factor_momentum_load_lag_beta",
        "ext4_factor_momentum_load_spread",
    ],
    "tags": ["factor_momentum", "beta", "index", "style", "multi_index"],
    "version": "1.0.0",
    "author": "Round-5 expansion (NEW: multi-index factor)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SYMBOLS = ["SPY", "QQQ", "IWM"]
_LEADER_WINDOW = 63   # days to rank leading index
_BETA_WINDOW   = 60   # days for rolling beta estimation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_index_returns(symbol: str) -> pd.Series:
    """Return daily log-return Series indexed by Date; empty on failure."""
    try:
        s = _indexes.index_close(symbol)
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        return np.log(s / s.shift(1))
    except Exception:
        return pd.Series(dtype=float)


def _rolling_beta(
    stock_ret: np.ndarray,
    idx_ret: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling OLS beta (covar / var) using a sliding window.
    Both arrays are 1-D, same length, already aligned.
    Returns float array with leading NaNs.
    """
    n = len(stock_ret)
    beta = np.full(n, np.nan)
    if n < window:
        return beta

    for i in range(window - 1, n):
        y = stock_ret[i - window + 1 : i + 1]
        x = idx_ret[i - window + 1 : i + 1]
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < max(2, window // 2):
            continue
        x_m = x[mask]
        y_m = y[mask]
        var_x = np.var(x_m, ddof=1)
        if var_x == 0 or not np.isfinite(var_x):
            continue
        cov = np.cov(y_m, x_m, ddof=1)[0, 1]
        beta[i] = cov / var_x

    return beta


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    nan_col = np.full(n, np.nan)

    df["ext4_factor_momentum_load_lead_beta"] = nan_col.copy()
    df["ext4_factor_momentum_load_lag_beta"]  = nan_col.copy()
    df["ext4_factor_momentum_load_spread"]    = nan_col.copy()

    if n < _BETA_WINDOW + 1:
        return df

    # -----------------------------------------------------------------------
    # 1. Load all three index return series and align to df via merge_asof
    # -----------------------------------------------------------------------
    dates_df = df[["Date"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(dates_df["Date"]):
        dates_df["Date"] = pd.to_datetime(dates_df["Date"])

    aligned: dict[str, np.ndarray] = {}
    for sym in _SYMBOLS:
        raw = _load_index_returns(sym)
        if len(raw) == 0:
            aligned[sym] = np.full(n, np.nan)
            continue
        raw_df = raw.reset_index()
        raw_df.columns = ["Date", "ret"]
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values("Date").dropna(subset=["Date"])
        merged = pd.merge_asof(
            dates_df.sort_values("Date"),
            raw_df,
            on="Date",
            direction="backward",
        )
        # Restore original row order
        merged.index = dates_df.sort_values("Date").index
        merged = merged.reindex(df.index)
        aligned[sym] = merged["ret"].values

    # -----------------------------------------------------------------------
    # 2. Compute stock daily log-returns
    # -----------------------------------------------------------------------
    close = df["Close"].values.astype(float)
    stock_ret = np.empty(n)
    stock_ret[0] = np.nan
    stock_ret[1:] = np.log(
        np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
    )

    # -----------------------------------------------------------------------
    # 3. Pre-compute rolling beta for each symbol
    # -----------------------------------------------------------------------
    betas: dict[str, np.ndarray] = {}
    for sym in _SYMBOLS:
        betas[sym] = _rolling_beta(stock_ret, aligned[sym], _BETA_WINDOW)

    # -----------------------------------------------------------------------
    # 4. For each row, identify the leading index (best 63d return)
    #    and the lagging index (worst 63d return), then assign betas.
    # -----------------------------------------------------------------------
    # Compute rolling 63-day cumulative return for each index on the index series
    # (not the stock) -- fully causal
    idx_cum: dict[str, np.ndarray] = {}
    for sym in _SYMBOLS:
        r = aligned[sym].copy()
        # rolling sum of log-returns over LEADER_WINDOW
        cum = np.full(n, np.nan)
        for i in range(_LEADER_WINDOW - 1, n):
            window_r = r[i - _LEADER_WINDOW + 1 : i + 1]
            valid = window_r[np.isfinite(window_r)]
            if len(valid) >= _LEADER_WINDOW // 2:
                cum[i] = np.sum(valid)
        idx_cum[sym] = cum

    lead_beta_arr = np.full(n, np.nan)
    lag_beta_arr  = np.full(n, np.nan)

    for i in range(n):
        cum_vals = {sym: idx_cum[sym][i] for sym in _SYMBOLS}
        # Need all three to be finite for ranking
        if not all(np.isfinite(v) for v in cum_vals.values()):
            continue
        ranked = sorted(cum_vals, key=lambda s: cum_vals[s], reverse=True)
        lead_sym = ranked[0]
        lag_sym  = ranked[-1]

        lb = betas[lead_sym][i]
        lgb = betas[lag_sym][i]

        if np.isfinite(lb):
            lead_beta_arr[i] = lb
        if np.isfinite(lgb):
            lag_beta_arr[i] = lgb

    spread_arr = np.where(
        np.isfinite(lead_beta_arr) & np.isfinite(lag_beta_arr),
        lead_beta_arr - lag_beta_arr,
        np.nan,
    )

    df["ext4_factor_momentum_load_lead_beta"] = lead_beta_arr
    df["ext4_factor_momentum_load_lag_beta"]  = lag_beta_arr
    df["ext4_factor_momentum_load_spread"]    = spread_arr

    return df
