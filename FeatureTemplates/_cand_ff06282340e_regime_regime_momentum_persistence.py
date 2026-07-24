"""
ff06282340e_regime_regime_momentum_persistence

Within the current VIX-tercile regime state, compute the lag-1 autocorrelation of
the ticker's daily returns using only the trailing days that share the current regime
label (last ~120 lookback). Captures whether momentum persists specifically in the
prevailing regime.

Per-ticker proxy: VIX tercile regime is determined via the trailing distribution of
VIX close values (low/mid/high), then lag-1 autocorrelation of returns is computed
using only same-regime bars within the lookback window. A slope variant tracks how
this regime-conditional autocorrelation has been changing over a secondary window.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# Load index/VIX helper
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282340e_regime_regime_momentum_persistence",
    "description": (
        "Lag-1 autocorrelation of daily returns conditioned on the current VIX-tercile "
        "regime (low/mid/high), computed over the trailing ~120 bars that share the same "
        "regime. A positive value means momentum tends to persist in the current regime; "
        "negative means mean-reversion. Slope variant measures how this autocorrelation "
        "has been trending over the last ~20 bars. Guard: NaN when fewer than 15 same-regime "
        "observations are available in the lookback. Per-ticker proxy of a genuinely "
        "cross-sectional regime signal."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282340e_regime_mom_persist",   # lag-1 autocorr within current VIX regime
        "ff06282340e_regime_mom_persist_slope",  # slope of autocorr over secondary window
        "ff06282340e_regime_label",         # current regime label: 0=low, 1=mid, 2=high VIX
    ],
    "tags": ["regime", "autocorrelation", "momentum", "vix", "conditional"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}

_LOOKBACK = 120       # bars of history to look back for same-regime autocorr
_MIN_OBS = 15         # minimum same-regime obs required to emit a value
_SLOPE_WIN = 20       # secondary window for slope of autocorr
_VIX_TERCILE_WIN = 252  # rolling window for VIX tercile boundaries


def _regime_acf1(returns: np.ndarray, regime_labels: np.ndarray,
                 current_regime: int, lookback: int, min_obs: int) -> float:
    """Compute lag-1 autocorrelation of returns within same-regime bars in lookback."""
    n = len(returns)
    start = max(0, n - lookback)
    r_window = returns[start:]
    reg_window = regime_labels[start:]

    mask = reg_window == current_regime
    r_reg = r_window[mask]

    if len(r_reg) < min_obs + 1:
        return np.nan

    x = r_reg[:-1]
    y = r_reg[1:]

    if len(x) < min_obs:
        return np.nan

    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if denom == 0.0:
        return np.nan

    return float(np.sum(xm * ym) / denom)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (required for every code path)
    df["ff06282340e_regime_mom_persist"] = np.nan
    df["ff06282340e_regime_mom_persist_slope"] = np.nan
    df["ff06282340e_regime_label"] = np.nan

    n = len(df)
    if n < _MIN_OBS + 2:
        return df

    # Load VIX and merge (backward-safe)
    try:
        vix_df = _indexes.vix_daily_close()
        df_sorted = df.copy()
        df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        merged = pd.merge_asof(
            df_sorted.sort_values("Date"),
            vix_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original row order
        merged = merged.set_index(df_sorted.sort_values("Date").index)
        vix_vals = merged["vix_close"].values.astype(float)
    except Exception:
        # If VIX unavailable, degrade gracefully: all NaN already set
        return df

    # Compute daily returns
    close = df["Close"].values.astype(float)
    returns = np.empty(n, dtype=float)
    returns[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = close[:-1]
        curr = close[1:]
        ret = np.where(prev == 0.0, np.nan, (curr - prev) / prev)
    returns[1:] = ret

    # Determine VIX tercile regime labels using rolling window
    # Fixed-from-start approach: for each bar, use trailing VIX_TERCILE_WIN bars
    # to define 33rd and 67th percentile, classify current bar
    regime_labels = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(vix_vals[i]):
            continue
        start = max(0, i + 1 - _VIX_TERCILE_WIN)
        window_vix = vix_vals[start: i + 1]
        valid = window_vix[~np.isnan(window_vix)]
        if len(valid) < 10:
            continue
        p33 = np.percentile(valid, 33.33)
        p67 = np.percentile(valid, 66.67)
        v = vix_vals[i]
        if v <= p33:
            regime_labels[i] = 0
        elif v <= p67:
            regime_labels[i] = 1
        else:
            regime_labels[i] = 2

    # Compute regime-conditional lag-1 autocorrelation for each bar
    acf_values = np.full(n, np.nan)
    min_start = _MIN_OBS + 2  # need enough history

    for i in range(min_start, n):
        if np.isnan(regime_labels[i]):
            continue
        cur_regime = int(regime_labels[i])
        # Use returns and regime_labels up to and including bar i (causal)
        r_hist = returns[: i + 1]
        reg_hist = regime_labels[: i + 1]
        # Replace nan regime labels with -1 so they never match
        reg_hist_clean = np.where(np.isnan(reg_hist), -1, reg_hist).astype(int)
        acf_values[i] = _regime_acf1(r_hist, reg_hist_clean, cur_regime,
                                      _LOOKBACK, _MIN_OBS)

    # Compute slope of acf_values over secondary window using fixed-grid striding
    # to stay causal (not re-anchored to last bar)
    slope_values = np.full(n, np.nan)
    stride = 1  # compute every bar for slope window (window is small)
    for i in range(_SLOPE_WIN - 1, n):
        window = acf_values[i - _SLOPE_WIN + 1: i + 1]
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < max(5, _SLOPE_WIN // 2):
            continue
        xs = np.arange(_SLOPE_WIN, dtype=float)[valid_mask]
        ys = window[valid_mask]
        if len(xs) < 2:
            continue
        xm = xs - xs.mean()
        denom = np.sum(xm ** 2)
        if denom == 0.0:
            continue
        slope_values[i] = float(np.sum(xm * (ys - ys.mean())) / denom)

    # Write back to df
    df["ff06282340e_regime_mom_persist"] = acf_values
    df["ff06282340e_regime_mom_persist_slope"] = slope_values
    df["ff06282340e_regime_label"] = regime_labels

    return df
