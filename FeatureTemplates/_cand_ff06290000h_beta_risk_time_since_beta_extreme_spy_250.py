from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load _indexes helper by file path
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_beta_risk_time_since_beta_extreme_spy_250",
    "description": (
        "Bars since last beta extreme vs SPY. Computes rolling 60-day beta on a fixed "
        "stride (every 5 bars from series start, forward-filled), then finds the most recent "
        "stride bar where |beta - median(beta)| > 2*MAD(beta) over the trailing 250-bar window "
        "of beta values. Feature = (current_bar - last_extreme_bar) / 250 (normalised recency "
        "of last beta shock). Guard: no extreme found -> 1.0; insufficient data -> NaN. "
        "A secondary variant tracks the magnitude of that extreme. Per-ticker proxy; "
        "causal/no-lookahead."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06290000h_beta_risk_time_since_beta_extreme_spy_250_recency",
        "ff06290000h_beta_risk_time_since_beta_extreme_spy_250_mag",
    ],
    "tags": ["beta", "beta_risk", "regime", "recency", "spy", "market"],
    "version": "1.0",
    "author": "feature-factory ff06290000h",
}

_COL_RECENCY = "ff06290000h_beta_risk_time_since_beta_extreme_spy_250_recency"
_COL_MAG = "ff06290000h_beta_risk_time_since_beta_extreme_spy_250_mag"

_BETA_WIN = 60      # bars used for each rolling beta estimate
_STRIDE = 5         # compute beta only at stride bars (fixed from series start)
_LOOKBACK = 250     # bars of beta history to look across for extremes
_MAD_MULT = 2.0     # threshold multiplier


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN on all code paths
    df[_COL_RECENCY] = np.nan
    df[_COL_MAG] = np.nan

    if len(df) < _BETA_WIN + _STRIDE:
        return df

    # --- Pull SPY close and align to df dates ---
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build a date-aligned SPY return series
    df_dates = pd.to_datetime(df["Date"].values)

    spy_close = spy_close.sort_index()
    spy_close.index = pd.to_datetime(spy_close.index)

    # Reindex SPY to df dates using backward fill (causal)
    spy_aligned = spy_close.reindex(df_dates, method="ffill")

    stock_close = df["Close"].values.astype(np.float64)
    spy_vals = spy_aligned.values.astype(np.float64)

    n = len(df)

    # --- Compute beta at stride points only (fixed from series start) ---
    # beta[i] = Cov(stock_ret, spy_ret over window ending at i) / Var(spy_ret)
    # Returns (log or simple) over the beta window
    stock_ret = np.empty(n)
    spy_ret = np.empty(n)
    stock_ret[0] = np.nan
    spy_ret[0] = np.nan

    prev_s = stock_close[0]
    prev_spy = spy_vals[0]
    for i in range(1, n):
        if prev_s > 0 and not np.isnan(prev_s):
            stock_ret[i] = stock_close[i] / prev_s - 1.0
        else:
            stock_ret[i] = np.nan
        if prev_spy > 0 and not np.isnan(prev_spy):
            spy_ret[i] = spy_vals[i] / prev_spy - 1.0
        else:
            spy_ret[i] = np.nan
        prev_s = stock_close[i]
        prev_spy = spy_vals[i]

    # beta_at[i] = rolling beta over [i-_BETA_WIN+1 .. i]; only computed at stride points
    beta_at = np.full(n, np.nan)

    for i in range(n):
        # Fixed-from-start stride grid
        if i % _STRIDE != 0:
            continue
        start = i - _BETA_WIN + 1
        if start < 1:
            continue
        sr = stock_ret[start: i + 1]
        mr = spy_ret[start: i + 1]
        mask = ~(np.isnan(sr) | np.isnan(mr))
        if mask.sum() < _BETA_WIN // 2:
            continue
        sr_m = sr[mask]
        mr_m = mr[mask]
        mr_mean = mr_m.mean()
        vr = np.sum((mr_m - mr_mean) ** 2)
        if vr == 0.0:
            continue
        cov = np.sum((sr_m - sr_m.mean()) * (mr_m - mr_mean))
        beta_at[i] = cov / vr

    # Forward-fill stride betas across non-stride bars (causal)
    beta_ff = np.full(n, np.nan)
    last_beta = np.nan
    for i in range(n):
        if not np.isnan(beta_at[i]):
            last_beta = beta_at[i]
        beta_ff[i] = last_beta

    # --- Build recency and magnitude features at each bar ---
    recency_out = np.full(n, np.nan)
    mag_out = np.full(n, np.nan)

    for i in range(n):
        # Gather stride indices within the last _LOOKBACK bars
        start_lb = max(0, i - _LOOKBACK + 1)

        # Collect (index, beta_value) for stride points in [start_lb, i]
        stride_idx = []
        stride_beta = []
        for j in range(start_lb, i + 1):
            if j % _STRIDE == 0 and not np.isnan(beta_at[j]):
                stride_idx.append(j)
                stride_beta.append(beta_at[j])

        if len(stride_beta) < 4:
            # Not enough data to estimate median/MAD reliably; leave NaN
            continue

        betas = np.array(stride_beta)
        med = np.median(betas)
        mad = np.median(np.abs(betas - med))
        threshold = _MAD_MULT * mad

        # Find the most recent stride bar where beta was extreme
        last_extreme_pos = -1  # position in stride_idx list
        last_extreme_beta = np.nan
        for k in range(len(stride_idx) - 1, -1, -1):
            if abs(stride_beta[k] - med) > threshold:
                last_extreme_pos = stride_idx[k]
                last_extreme_beta = stride_beta[k]
                break

        if last_extreme_pos == -1:
            # No extreme found in lookback window -> recency = 1.0 (very stale)
            recency_out[i] = 1.0
            mag_out[i] = 0.0
        else:
            bars_since = i - last_extreme_pos
            recency_out[i] = bars_since / _LOOKBACK
            mag_out[i] = abs(last_extreme_beta - med) / (mad if mad > 0 else 1.0)

    df[_COL_RECENCY] = recency_out
    df[_COL_MAG] = mag_out

    return df
