from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load index helper (SPY)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_beta_risk_beta_sign_flip_count_spy_250",
    "description": (
        "Beta-direction instability: rolling 30-day OLS SPY-beta is computed on a "
        "stride-5 grid across a trailing 250-day window. The feature counts how often "
        "that stride-level beta flips sign (ignoring near-zero betas |b|<0.05) divided "
        "by the number of stride points observed -- the fraction of transitions that "
        "represent a direction reversal in market exposure. High values signal a stock "
        "that erratically switches between positive and negative market loading "
        "(regime instability). A second variant reports the raw flip count. "
        "Per-ticker causal proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_beta_sign_flip_freq",   # flip count / stride points (rate)
        "ff06282316b_beta_sign_flip_count",  # raw flip count in trailing 250-day window
        "ff06282316b_beta_last_30d",         # most recent 30-day OLS beta (context)
    ],
    "tags": ["beta", "regime", "instability", "market_exposure", "beta_risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_EPSILON = 0.05       # beta near-zero threshold: flips across this are ignored
_WIN_BETA = 30        # days for each rolling OLS beta estimate
_WIN_LONG = 250       # lookback window for the flip-count metric
_STRIDE = 5           # compute beta every 5 bars (causal fixed-from-start grid)


def _ols_beta(ret_stock: np.ndarray, ret_spy: np.ndarray) -> float:
    """Return OLS beta of stock on SPY; NaN if insufficient variance."""
    var_spy = np.nanvar(ret_spy)
    if var_spy < 1e-12 or len(ret_stock) < 5:
        return np.nan
    cov = np.nanmean((ret_stock - np.nanmean(ret_stock)) *
                     (ret_spy   - np.nanmean(ret_spy)))
    return cov / var_spy


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN upfront (required on every path)
    df["ff06282316b_beta_sign_flip_freq"]  = np.nan
    df["ff06282316b_beta_sign_flip_count"] = np.nan
    df["ff06282316b_beta_last_30d"]        = np.nan

    if len(df) < _WIN_BETA + 1:
        return df

    # ------------------------------------------------------------------
    # 1. Align SPY returns to this ticker's dates
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # Build a date-aligned SPY return series
    ticker_dates = pd.to_datetime(df["Date"])
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_aligned = spy_close.reindex(ticker_dates).ffill()  # safe: only fills from past

    stock_close = df["Close"].values
    spy_vals    = spy_aligned.values
    n           = len(df)

    stock_ret = np.empty(n)
    spy_ret   = np.empty(n)
    stock_ret[0] = np.nan
    spy_ret[0]   = np.nan
    stock_ret[1:] = np.diff(stock_close) / np.where(stock_close[:-1] == 0, np.nan, stock_close[:-1])
    spy_ret[1:]   = np.diff(spy_vals)    / np.where(spy_vals[:-1] == 0, np.nan, spy_vals[:-1])

    # ------------------------------------------------------------------
    # 2. Compute stride-level betas on a FIXED-FROM-START grid
    #    stride_indices = all i where i % _STRIDE == 0, i >= _WIN_BETA
    # ------------------------------------------------------------------
    stride_betas_idx  = []   # (bar_index, beta_value)

    for i in range(_WIN_BETA, n):
        if i % _STRIDE != 0:
            continue
        r_s = stock_ret[i - _WIN_BETA + 1: i + 1]
        r_m = spy_ret[  i - _WIN_BETA + 1: i + 1]
        valid = (~np.isnan(r_s)) & (~np.isnan(r_m))
        if valid.sum() < 10:
            stride_betas_idx.append((i, np.nan))
        else:
            stride_betas_idx.append((i, _ols_beta(r_s[valid], r_m[valid])))

    if not stride_betas_idx:
        return df

    stride_arr = np.array(stride_betas_idx, dtype=float)  # shape (K, 2)
    bar_indices = stride_arr[:, 0].astype(int)
    beta_vals   = stride_arr[:, 1]

    # ------------------------------------------------------------------
    # 3. For each bar, look back over stride points that fall within
    #    [bar - _WIN_LONG, bar] and count sign flips.
    #    We only fill output at stride bars (forward-filled below).
    # ------------------------------------------------------------------
    flip_freq_at_stride  = np.full(len(stride_betas_idx), np.nan)
    flip_count_at_stride = np.full(len(stride_betas_idx), np.nan)
    last_beta_at_stride  = np.full(len(stride_betas_idx), np.nan)

    for j, cur_bar in enumerate(bar_indices):
        # Find stride points within trailing _WIN_LONG bars
        mask = (bar_indices >= cur_bar - _WIN_LONG) & (bar_indices <= cur_bar)
        window_betas = beta_vals[mask]

        # Only count non-NaN betas with |beta| >= epsilon
        non_tiny = window_betas[~np.isnan(window_betas)]
        # Replace near-zero with zero (they don't define a sign)
        signed = np.where(np.abs(non_tiny) >= _EPSILON, np.sign(non_tiny), 0.0)
        # Remove zeros (near-zero betas skipped in flip counting)
        signed = signed[signed != 0.0]

        n_pts = len(signed)
        if n_pts >= 2:
            flips = int(np.sum(signed[1:] != signed[:-1]))
            flip_count_at_stride[j] = flips
            flip_freq_at_stride[j]  = flips / (n_pts - 1)
        elif n_pts == 1:
            flip_count_at_stride[j] = 0
            flip_freq_at_stride[j]  = 0.0

        # Last beta in window (most recent stride point at or before cur_bar)
        valid_b = window_betas[~np.isnan(window_betas)]
        if len(valid_b) > 0:
            last_beta_at_stride[j] = valid_b[-1]

    # ------------------------------------------------------------------
    # 4. Place values into df at stride bars, then forward-fill
    # ------------------------------------------------------------------
    freq_col  = np.full(n, np.nan)
    count_col = np.full(n, np.nan)
    last_col  = np.full(n, np.nan)

    for j, bar in enumerate(bar_indices):
        freq_col[bar]  = flip_freq_at_stride[j]
        count_col[bar] = flip_count_at_stride[j]
        last_col[bar]  = last_beta_at_stride[j]

    # Forward-fill (propagate last known value — causal)
    freq_s  = pd.array(freq_col,  dtype="Float64")
    count_s = pd.array(count_col, dtype="Float64")
    last_s  = pd.array(last_col,  dtype="Float64")

    df["ff06282316b_beta_sign_flip_freq"]  = pd.array(freq_col).copy()
    df["ff06282316b_beta_sign_flip_count"] = pd.array(count_col).copy()
    df["ff06282316b_beta_last_30d"]        = pd.array(last_col).copy()

    # Forward-fill the stride-sparse series
    df["ff06282316b_beta_sign_flip_freq"]  = (
        df["ff06282316b_beta_sign_flip_freq"].astype(float).ffill()
    )
    df["ff06282316b_beta_sign_flip_count"] = (
        df["ff06282316b_beta_sign_flip_count"].astype(float).ffill()
    )
    df["ff06282316b_beta_last_30d"] = (
        df["ff06282316b_beta_last_30d"].astype(float).ffill()
    )

    return df
