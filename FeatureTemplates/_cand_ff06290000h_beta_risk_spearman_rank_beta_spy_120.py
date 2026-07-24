from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# ---------------------------------------------------------------------------
# Load index helper (SPY close prices)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_beta_risk_spearman_rank_beta_spy_120",
    "description": (
        "Outlier-robust Spearman/Gaussian-rank beta vs SPY over a 120-day rolling window. "
        "Converts both ticker and SPY daily returns to Gaussian ranks (rank/N centered at 0), "
        "then fits OLS slope of rank_r on rank_m ('rank-beta'). "
        "Primary output: the rank-beta itself. "
        "Secondary output: robustness-gap = rank_beta - ordinary OLS beta, measuring how much "
        "tail returns inflate the raw beta. "
        "Tertiary: 21-day rolling change in rank-beta (momentum of robustness). "
        "Per-ticker proxy; leakage-free causal rolling windows."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290000h_beta_risk_spearman_rank_beta_spy_120_rank_beta",
        "ff06290000h_beta_risk_spearman_rank_beta_spy_120_robust_gap",
        "ff06290000h_beta_risk_spearman_rank_beta_spy_120_gap_chg21",
    ],
    "tags": ["beta", "robustness", "rank", "spearman", "market_sensitivity"],
    "version": "1.0.0",
    "author": "feature-factory ff06290000h",
}

_WINDOW = 120
_MIN_OBS = 30
_GAP_CHNG = 21

# Column name aliases
_COL_RANK_BETA = "ff06290000h_beta_risk_spearman_rank_beta_spy_120_rank_beta"
_COL_GAP = "ff06290000h_beta_risk_spearman_rank_beta_spy_120_robust_gap"
_COL_GAP_CHG = "ff06290000h_beta_risk_spearman_rank_beta_spy_120_gap_chg21"


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x (no intercept correction -- mean-centered inputs)."""
    xvar = np.dot(x, x)
    if xvar == 0.0:
        return np.nan
    return np.dot(x, y) / xvar


def _gaussian_rank(arr: np.ndarray) -> np.ndarray:
    """Convert array to Gaussian ranks: rank/N, centered so mean ~ 0."""
    n = len(arr)
    if n == 0:
        return arr.copy()
    # argsort of argsort gives ranks 0..n-1
    ranks = np.argsort(np.argsort(arr)).astype(np.float64)
    # center: (rank + 0.5) / n  -> (0, 1) then subtract 0.5 -> (-0.5, 0.5)
    return (ranks + 0.5) / n - 0.5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_RANK_BETA] = np.nan
    df[_COL_GAP] = np.nan
    df[_COL_GAP_CHG] = np.nan

    if len(df) < _MIN_OBS + 1:
        return df

    # Fetch SPY closes and align to this ticker's dates
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build a daily SPY return series aligned to ticker dates
    ticker_dates = pd.to_datetime(df["Date"])
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.sort_index()

    # Merge SPY into ticker frame via merge_asof (backward, no lookahead)
    spy_df = spy_close.rename("spy_close").reset_index().rename(columns={"index": "Date", "Date": "Date"})
    # Handle the case index might already be called "Date"
    if "Date" not in spy_df.columns:
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]

    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date")

    tmp = df[["Date", "Close"]].copy()
    tmp["Date"] = ticker_dates
    tmp = tmp.sort_values("Date")

    merged = pd.merge_asof(tmp, spy_df, on="Date", direction="backward")
    # Compute returns
    ticker_ret = merged["Close"].pct_change().values          # len N
    spy_ret = merged["spy_close"].pct_change().values         # len N

    n = len(ticker_ret)
    rank_beta_arr = np.full(n, np.nan)
    gap_arr = np.full(n, np.nan)

    for i in range(_WINDOW - 1, n):
        start = i - _WINDOW + 1
        r = ticker_ret[start: i + 1]
        m = spy_ret[start: i + 1]

        # Drop NaN pairs
        mask = np.isfinite(r) & np.isfinite(m)
        r_c = r[mask]
        m_c = m[mask]

        obs = len(r_c)
        if obs < _MIN_OBS:
            continue

        # --- Rank-beta (Spearman-OLS) ---
        r_rank = _gaussian_rank(r_c)
        m_rank = _gaussian_rank(m_c)

        # center (already ~0 by construction, but subtract mean to be safe)
        r_rank -= r_rank.mean()
        m_rank -= m_rank.mean()

        rank_var = np.dot(m_rank, m_rank)
        if rank_var == 0.0:
            rank_beta_arr[i] = 0.0
            gap_arr[i] = 0.0
            continue

        rb = np.dot(m_rank, r_rank) / rank_var
        rank_beta_arr[i] = rb

        # --- Ordinary OLS beta ---
        r_dm = r_c - r_c.mean()
        m_dm = m_c - m_c.mean()
        ols_var = np.dot(m_dm, m_dm)
        if ols_var == 0.0:
            gap_arr[i] = 0.0
        else:
            ols_b = np.dot(m_dm, r_dm) / ols_var
            gap_arr[i] = rb - ols_b

    # Map back to original df index order (df may not be sorted identically)
    # merged rows correspond to sorted tmp rows; we need to align back to df
    sorted_idx = tmp.index.values  # original df index values, in date-sorted order
    df.loc[sorted_idx, _COL_RANK_BETA] = rank_beta_arr
    df.loc[sorted_idx, _COL_GAP] = gap_arr

    # 21-day change in robustness gap (use the gap column now in df, date-sorted order)
    gap_series = df.loc[sorted_idx, _COL_GAP]
    gap_chg = gap_series.diff(_GAP_CHNG)
    df.loc[sorted_idx, _COL_GAP_CHG] = gap_chg.values

    return df
