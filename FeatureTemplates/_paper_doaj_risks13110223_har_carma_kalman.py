"""
HAR-RV Features (Proxy)  —  doi:10.3390/risks13110223
"HAR-RV-CARMA: A Kalman Filter-Weighted Hybrid Model for Enhanced Volatility Forecasting"

PROXY RATIONALE:
The paper combines HAR-RV (Heterogeneous Autoregressive Realized Volatility) with a CARMA
(Continuous Autoregressive Moving Average) model, weighted via a Kalman filter trained on
5-min realized volatility data. We cannot replicate:
  - Intraday realized volatility (only daily OHLCV available)
  - CARMA model fitting (requires continuous-time state-space estimation)
  - Kalman filter weighting (requires the fitted CARMA + HAR-RV models)

PROXY IMPLEMENTED:
  - HAR-RV structure: multi-scale (daily, weekly, monthly) volatility estimators
    using the OHLCV range (Parkinson/Garman-Klass style) as proxies for realized volatility.
  - HAR-RV predictors: lagged 1d / 5d-avg / 22d-avg of the range-based RV proxy,
    exactly mirroring the HAR-RV regressor structure from the paper.
  - CARMA proxy: exponential moving averages at short/medium windows to approximate
    the smooth continuous-time ARMA dynamics the paper uses as a complement.
  - Combined weighting proxy: a simple adaptive blend based on recent forecast error
    direction (a deterministic heuristic, NOT a fitted Kalman gain).

All features are causal and computed from daily OHLCV only.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_risks13110223_har_carma_kalman",
    "description": (
        "HAR-RV proxy features (range-based daily/weekly/monthly realized-vol estimators "
        "at HAR lag structure) inspired by doi:10.3390/risks13110223 HAR-RV-CARMA model; "
        "CARMA and Kalman filter components are proxied by EMA differences and adaptive "
        "blending heuristics — no intraday data or model fitting available."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "harv_rv_d1",
        "harv_rv_w5",
        "harv_rv_m22",
        "harv_har_fitted",
        "harv_carma_proxy_5",
        "harv_carma_proxy_22",
        "harv_blend_ratio",
        "harv_rv_momentum",
        "harv_vol_regime",
    ],
    "tags": ["volatility", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.3390/risks13110223 (HAR-RV-CARMA). "
        "CARMA+Kalman dropped (needs intraday RV + state-space fitting). "
        "Replaced by EMA-based CARMA proxy and adaptive blend heuristic."
    ),
}


def _garman_klass_rv(o, h, l, c):
    """
    Garman-Klass range-based realized variance proxy (single-bar scalar arrays).
    RV ~ 0.5*(log(H/L))^2 - (2*log(2)-1)*(log(C/O))^2
    Returns a non-negative float array (annualized variance, daily scale).
    """
    log_hl = np.log(np.where(h > 0, h / np.where(l > 0, l, np.nan), np.nan))
    log_co = np.log(np.where(o > 0, c / o, np.nan))
    rv = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    # Clip to non-negative (numerical noise can push tiny negative)
    return np.where(np.isfinite(rv), np.maximum(rv, 0.0), np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HAR-RV proxy features.

    The HAR-RV regressor structure uses three volatility aggregates:
      - RV_d  : yesterday's (lag-1) realized volatility
      - RV_w  : past 5-day average realized volatility
      - RV_m  : past 22-day average realized volatility

    These are the exact HAR regressors (HAR = Heterogeneous AR) from:
      Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility"
    as used in the paper's HAR-RV sub-model.
    """
    o = df["Open"].values.astype(np.float64)
    h = df["High"].values.astype(np.float64)
    l = df["Low"].values.astype(np.float64)
    c = df["Close"].values.astype(np.float64)
    n = len(c)

    # --- Garman-Klass range-based RV proxy per bar ---
    rv_daily = _garman_klass_rv(o, h, l, c)  # shape (n,)

    # --- HAR-RV structure: lag-1, 5d-avg, 22d-avg ---
    # rv_d: yesterday's RV (lag-1 of rv_daily)
    rv_d = np.full(n, np.nan)
    rv_d[1:] = rv_daily[:-1]

    # rv_w: 5-day trailing mean of rv_daily (shift by 1 for causal: yesterday's 5d avg)
    rv_w_raw = pd.Series(rv_daily).rolling(5, min_periods=3).mean().values
    rv_w = np.full(n, np.nan)
    rv_w[1:] = rv_w_raw[:-1]

    # rv_m: 22-day trailing mean of rv_daily (shift by 1 for causal)
    rv_m_raw = pd.Series(rv_daily).rolling(22, min_periods=11).mean().values
    rv_m = np.full(n, np.nan)
    rv_m[1:] = rv_m_raw[:-1]

    df["harv_rv_d1"] = rv_d
    df["harv_rv_w5"] = rv_w
    df["harv_rv_m22"] = rv_m

    # --- HAR fitted value proxy: weighted combination of lag regressors ---
    # Weights from the paper's typical OLS fit: beta_d~0.4, beta_w~0.3, beta_m~0.3
    # This is a static heuristic (not fitted) — PROXY only
    har_fitted = np.where(
        np.isfinite(rv_d) & np.isfinite(rv_w) & np.isfinite(rv_m),
        0.4 * rv_d + 0.3 * rv_w + 0.3 * rv_m,
        np.nan,
    )
    df["harv_har_fitted"] = har_fitted

    # --- CARMA proxy: EMA-based smooth trend in RV ---
    # CARMA(p,q) in continuous time acts as a smoother; EMA approximates this
    rv_ser = pd.Series(rv_daily)
    ema_5  = rv_ser.ewm(span=5,  min_periods=3, adjust=False).mean().values
    ema_22 = rv_ser.ewm(span=22, min_periods=11, adjust=False).mean().values

    # Shift by 1 to keep causal
    carma_5 = np.full(n, np.nan)
    carma_5[1:] = ema_5[:-1]
    carma_22 = np.full(n, np.nan)
    carma_22[1:] = ema_22[:-1]

    df["harv_carma_proxy_5"]  = carma_5
    df["harv_carma_proxy_22"] = carma_22

    # --- Adaptive blend ratio: HAR vs CARMA (Kalman proxy) ---
    # In the paper the Kalman filter dynamically weights HAR vs CARMA based on forecast errors.
    # Proxy: ratio of recent HAR fitted vs CARMA proxy deviation from realized RV.
    # Higher ratio → HAR recently more accurate; lower → CARMA more accurate.
    har_err  = np.abs(har_fitted - rv_d)          # HAR "error" on yesterday's RV
    carma_err = np.abs(carma_5 - rv_d)            # CARMA proxy "error" on yesterday's RV

    blend = np.where(
        np.isfinite(har_err) & np.isfinite(carma_err) & (har_err + carma_err > 1e-12),
        carma_err / (har_err + carma_err),         # → 1 means CARMA is worse (prefer HAR)
        np.nan,
    )
    df["harv_blend_ratio"] = blend

    # --- RV momentum: ratio of short vs long RV ---
    rv_momentum = np.where(
        np.isfinite(rv_w) & np.isfinite(rv_m) & (rv_m > 1e-12),
        rv_w / rv_m,
        np.nan,
    )
    df["harv_rv_momentum"] = rv_momentum

    # --- Vol regime: z-score of current RV vs 63d rolling distribution ---
    rv_ser2 = pd.Series(rv_d)
    roll_mean = rv_ser2.rolling(63, min_periods=20).mean()
    roll_std  = rv_ser2.rolling(63, min_periods=20).std(ddof=1)
    vol_regime = (rv_ser2 - roll_mean) / roll_std.replace(0, np.nan)
    df["harv_vol_regime"] = vol_regime.values

    return df
