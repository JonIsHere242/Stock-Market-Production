"""
Volume→Volatility vs Volatility→Volume Lead-Lag feature block.

Extension/exploration of gate-validated winner xdom2_autocorr_volume_return.
Captures DIRECTED lead-lag asymmetry: does volume predict future volatility
more than volatility predicts future volume?

Per-ticker proxy (inherently single-stock here — no cross-sectional ranks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_volume_vol_leadlag",
    "description": (
        "Rolling 60-day directed lead-lag cross-correlation between detrended "
        "log-volume and realized absolute returns. Computes two directed pairs "
        "within each causal window: (1) vol_lead = corr(log_vol[t-1], |ret|[t]) "
        "averaged over the window (volume leads volatility); "
        "(2) ret_lead = corr(|ret|[t-1], log_vol[t]) averaged over the window "
        "(volatility leads volume). The difference (vol_lead - ret_lead) gives "
        "the net lead-lag sign. All correlations use only past data at each bar. "
        "Extension of xdom2_autocorr_volume_return; captures directional "
        "information flow asymmetry rather than simple autocorrelation."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_volume_vol_leadlag_vol_leads",   # corr(log_vol[t-1], |ret|[t]), 60d roll
        "ext_volume_vol_leadlag_ret_leads",   # corr(|ret|[t-1], log_vol[t]), 60d roll
        "ext_volume_vol_leadlag_net",         # vol_leads - ret_leads (sign of causal flow)
    ],
    "tags": ["volume", "volatility", "lead-lag", "cross-correlation", "microstructure"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner xdom2_autocorr_volume_return. "
        "Spec: ext_volume_vol_leadlag. Directed-correlation method inspired by "
        "Granger-causality literature on volume-volatility information flow."
    ),
}

# Rolling window length
_WINDOW = 60
# Minimum observations required inside a window before emitting a value
_MIN_PERIODS = 30


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directed volume<->volatility lead-lag cross-correlations.

    Within every causal 60-bar rolling window we estimate:
      vol_leads[t]  = corr( log_vol[t-k-1], |ret|[t-k] )   averaged over k in window
      ret_leads[t]  = corr( |ret|[t-k-1],  log_vol[t-k] )  averaged over k in window
      net[t]        = vol_leads[t] - ret_leads[t]

    All values use only data available at or before bar t → no lookahead.
    """
    n = len(df)

    # --- raw series ---
    log_vol = np.log(df["Volume"].replace(0, np.nan).values.astype(np.float64))

    close = df["Close"].values.astype(np.float64)
    # Log returns; NaN at first bar
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    abs_ret = np.abs(log_ret)  # realized absolute return (proxy for volatility)

    # --- detrend log_vol with a simple 20-bar rolling mean subtraction ---
    # (mimics "detrended" as mentioned in the spec; keeps it causal)
    lv_series = pd.Series(log_vol)
    lv_trend = lv_series.rolling(20, min_periods=5).mean()
    lv_detrended = (lv_series - lv_trend).values  # shape (n,)

    # lag-1 series (shift right by 1): lagged_lv[t] = lv_detrended[t-1]
    lagged_lv = np.empty(n, dtype=np.float64)
    lagged_lv[0] = np.nan
    lagged_lv[1:] = lv_detrended[:-1]

    lagged_ar = np.empty(n, dtype=np.float64)
    lagged_ar[0] = np.nan
    lagged_ar[1:] = abs_ret[:-1]

    # We'll compute rolling Pearson correlation over a window using numpy
    # via a vectorized sliding approach with pd.Series.rolling.
    # For two series x, y: corr = cov(x,y) / (std(x) * std(y))
    # pandas rolling().corr() is causal (backward-looking).

    lv_s = pd.Series(lv_detrended)
    ar_s = pd.Series(abs_ret)
    lag_lv_s = pd.Series(lagged_lv)
    lag_ar_s = pd.Series(lagged_ar)

    # (1) volume leads volatility: corr(lv[t-1], |ret|[t]) over 60-bar window
    #     x = lagged_lv, y = abs_ret
    vol_leads = _rolling_corr(lag_lv_s, ar_s, window=_WINDOW, min_periods=_MIN_PERIODS)

    # (2) volatility leads volume: corr(|ret|[t-1], lv[t]) over 60-bar window
    #     x = lagged_ar, y = lv_detrended
    ret_leads = _rolling_corr(lag_ar_s, lv_s, window=_WINDOW, min_periods=_MIN_PERIODS)

    # (3) net = vol_leads - ret_leads
    net = vol_leads - ret_leads

    # Guard against any inf that might slip through
    def _clean(arr: np.ndarray) -> np.ndarray:
        arr = np.where(np.isinf(arr), np.nan, arr)
        return arr

    df["ext_volume_vol_leadlag_vol_leads"] = _clean(vol_leads)
    df["ext_volume_vol_leadlag_ret_leads"] = _clean(ret_leads)
    df["ext_volume_vol_leadlag_net"] = _clean(net)

    return df


def _rolling_corr(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: int,
) -> np.ndarray:
    """
    Causal rolling Pearson correlation of x and y over `window` bars.
    Uses pandas rolling machinery (O(n) per series, vectorised C backend).
    Returns numpy array of float64.
    """
    # pandas rolling corr is backward-looking and handles NaN propagation
    result = x.rolling(window=window, min_periods=min_periods).corr(y)
    return result.values.astype(np.float64)
