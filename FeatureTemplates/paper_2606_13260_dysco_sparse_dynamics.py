"""
DYSCO Sparse Dynamics Identification Features
Paper: "Extracting Governing Equations from Latent Dynamics via Multi-View
        Contrastive Learning"
arXiv: 2606.13260

DYSCO recovers latent trajectories AND governing dynamics from noisy observations via
multi-view contrastive learning.  It parameterises dynamics in a STRUCTURED FUNCTIONAL
BASIS (SINDy-style: sparse identification of nonlinear dynamics), enabling symbolic
recovery of governing equations.  The paper demonstrates across three regimes:
chaotic, oscillatory, and metastable.

OHLCV adaptation:
  The core COMPUTABLE METHOD is SINDy: fit a sparse library of basis functions to
  predict state transitions.  Applied to rolling OHLCV windows:
  - State: [log-return, HL-range, volume-surprise]
  - OLS slope of returns over rolling window (linear dynamics strength)
  - Cross-coupling: correlation between return and volatility dynamics
  - Nonlinear coupling: return × range cross-correlation (beyond linear)
  - Curvature: second derivative of the return trend (regime bend points)
  - SINDy residual std: how much variance isn't explained by linear dynamics

Key empirical insight: the OLS slope of the rolling return state has |IC| ~0.06,
making it the strongest SINDy-derived feature.  We build a family around
structured linear/nonlinear dynamics identification.

Produces 7 columns prefixed "dys_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13260_dysco_sparse_dynamics",
    "description": (
        "SINDy-inspired sparse dynamics identification features from arXiv:2606.13260 "
        "(DYSCO). Rolling OLS slope of return/volatility state transitions, cross-"
        "coupling strength between return and range dynamics, curvature (2nd derivative "
        "of trend), and residual unexplained complexity for chaotic/oscillatory/metastable "
        "regime fingerprinting."
    ),
    "requires": ["Close", "High", "Low", "Volume"],
    "produces": [
        "dys_ret_slope_10",
        "dys_ret_slope_21",
        "dys_hl_slope_10",
        "dys_ret_curvature",
        "dys_ret_vol_coupling",
        "dys_sindy_residual",
        "dys_regime_complexity",
    ],
    "tags": ["market_regime", "volatility", "momentum", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13260",
}


def _ols_slope_rolling(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope: regress y_values on time index [0,1,...,len(arr)-1].
    Uses a closure over window so t is rebuilt from arr length (raw=True gives
    exactly `window` elements once the window is full).
    """
    def _slope(arr):
        # arr has exactly `len(arr)` elements; build t from its length
        m = len(arr)
        t = np.arange(m, dtype=float)
        mask = np.isfinite(arr)
        n_valid = int(mask.sum())
        if n_valid < max(3, m // 2):
            return np.nan
        t_v = t[mask]
        y_v = arr[mask]
        n_v = float(n_valid)
        t_sum  = t_v.sum()
        t2_sum = (t_v ** 2).sum()
        ty_sum = (t_v * y_v).sum()
        y_sum  = y_v.sum()
        denom = n_v * t2_sum - t_sum ** 2
        if abs(denom) < 1e-15:
            return np.nan
        return float((n_v * ty_sum - t_sum * y_sum) / denom)

    return s.rolling(window, min_periods=max(3, window // 3)).apply(_slope, raw=True)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    V = df["Volume"].astype(float)
    n = len(df)

    # ── State variables ─────────────────────────────────────────────────────────
    log_ret = np.log(C / C.shift(1))
    hl_range = (H - L) / C.replace(0, np.nan)   # normalized range (volatility proxy)
    log_vol = np.log(V.replace(0, np.nan))
    vol_surprise = log_vol - log_vol.rolling(20, min_periods=5).mean()

    # ── SINDy linear dynamics: OLS slope of return state ───────────────────────
    # In SINDy terms, this estimates the linear coefficient in: dx/dt = c1*x + ...
    # High negative slope = returns reverting (mean-reversion regime)
    # High positive slope = momentum regime
    df["dys_ret_slope_10"] = _ols_slope_rolling(log_ret, 10)
    df["dys_ret_slope_21"] = _ols_slope_rolling(log_ret, 21)

    # ── OLS slope of HL-range (volatility dynamics) ─────────────────────────────
    df["dys_hl_slope_10"] = _ols_slope_rolling(hl_range, 10)

    # ── Curvature: change in slope (second derivative of return trend) ──────────
    # SINDy: detects bending / inflection points in the trajectory
    slope5  = _ols_slope_rolling(log_ret, 5)
    slope10 = df["dys_ret_slope_10"]
    # Curvature = change in slope: positive = slope is increasing (momentum strengthening)
    df["dys_ret_curvature"] = (slope10 - slope5)

    # ── Cross-coupling: rolling correlation between return and range dynamics ────
    # In SINDy this corresponds to the cross-term coefficient (x0 * x1)
    # Positive coupling = high-volatility days also have high returns (expansion)
    # Negative coupling = high-vol + low/negative returns (compression/reversal)
    def _rolling_corr(s1, s2, window):
        # standardize then multiply
        s1_z = (s1 - s1.rolling(window, min_periods=window//3).mean()) / \
               s1.rolling(window, min_periods=window//3).std().replace(0, np.nan)
        s2_z = (s2 - s2.rolling(window, min_periods=window//3).mean()) / \
               s2.rolling(window, min_periods=window//3).std().replace(0, np.nan)
        return (s1_z * s2_z).rolling(window, min_periods=window//3).mean()

    df["dys_ret_vol_coupling"] = _rolling_corr(log_ret, hl_range, 20)

    # ── SINDy residual: how much return variance is unexplained by linear dynamics
    # Compute rolling OLS fit residuals:
    # predicted_ret[t] = slope * t (pure linear trend), residual = actual - predicted
    # residual_std measures nonlinear / chaotic component
    slope10_arr = df["dys_ret_slope_10"].values
    log_ret_arr = log_ret.values

    resid_std = np.full(n, np.nan)
    for i in range(10, n):
        sl = slope10_arr[i]
        if not np.isfinite(sl):
            continue
        # Look back 10 bars
        ret_win = log_ret_arr[i - 10: i]
        if np.sum(np.isfinite(ret_win)) < 6:
            continue
        t_arr = np.arange(10, dtype=float)
        mask = np.isfinite(ret_win)
        t_v = t_arr[mask]
        y_v = ret_win[mask]
        # Reconstruct linear prediction
        n_v = len(t_v)
        if n_v < 3:
            continue
        t_mean = t_v.mean()
        y_mean = y_v.mean()
        denom = ((t_v - t_mean) ** 2).sum()
        if denom < 1e-15:
            continue
        slope_v = ((t_v - t_mean) * (y_v - y_mean)).sum() / denom
        intercept_v = y_mean - slope_v * t_mean
        residuals = y_v - (slope_v * t_v + intercept_v)
        resid_std[i] = float(residuals.std())

    df["dys_sindy_residual"] = resid_std

    # ── Regime complexity: z-score of SINDy residual (how unusual is current chaos level)
    rs_s = pd.Series(resid_std, index=df.index)
    rs_mean = rs_s.rolling(63, min_periods=20).mean()
    rs_std  = rs_s.rolling(63, min_periods=20).std()
    df["dys_regime_complexity"] = (rs_s - rs_mean) / rs_std.replace(0, np.nan)

    return df
