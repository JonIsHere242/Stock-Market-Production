"""
Matern 5/2 kernel-weighted realized volatility per-ticker proxy derived from:
  "A Hybrid Gaussian Process Regression Framework for Stable Volatility-Covariance
   Estimation: Evidence from Global Equity Indices" (arXiv:2605.17275).

The paper uses GPR with a Matern 5/2 kernel to forecast individual asset volatility,
arguing the Matern kernel's specific roughness (differentiable but not infinitely smooth)
better captures the local persistence structure of financial volatility than EWMA/GARCH.

The Matern 5/2 kernel weight at distance d with lengthscale l is:
    w(d) = (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)

We approximate GPR prediction as a kernel-weighted average of past squared log-returns
(realized variance) using this weight function.  This gives a richer smoothing than
exponential decay:
  - Near-zero lag: weight ≈ 1 (EWMA-like)
  - Intermediate lag: weight decays more slowly than exp (captures persistence)
  - Long lag: weight decays faster than power (no long-memory bleed)

Three lengthscales map to short/medium/long-term vol forecasts.

Also adds:
  - mkv_vol_ratio_short_long: short/long Matern vol ratio (vol regime indicator)
  - mkv_vol_vs_ewma_20: Matern mid-scale vol / standard 20-day EWMA vol (novelty of
    Matern smoothing vs simple exponential baseline)

All rolling, causal, no lookahead.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name":        "_paper_2605_17275_matern_kernel_vol",
    "description": (
        "Matern 5/2 kernel-weighted realized volatility at short/mid/long lengthscales; "
        "captures financial-vol roughness structure of GPR-HS framework (arXiv 2605.17275)."
    ),
    "requires":    ["Close"],
    "produces":    [
        "mkv_vol_short",        # Matern vol, lengthscale=10d (annualized)
        "mkv_vol_mid",          # Matern vol, lengthscale=30d (annualized)
        "mkv_vol_long",         # Matern vol, lengthscale=90d (annualized)
        "mkv_ratio_short_long", # mkv_vol_short / mkv_vol_long
        "mkv_vs_ewma20",        # mkv_vol_mid / EWMA-20 vol (Matern vs exponential)
    ],
    "tags":        ["experimental", "volatility", "kernel"],
    "version":     "1.0",
    "author":      "paper-mining slate 6",
}

# Annualization factor
_ANNUAL = np.sqrt(252.0)

# Lengthscales (in trading days)
_LS_SHORT = 10.0
_LS_MID   = 30.0
_LS_LONG  = 90.0

# Lookback for kernel weight computation (3× longest lengthscale, capped)
_LOOKBACK = 270   # 3 * 90
_MIN_OBS  = 10    # minimum observations for a valid estimate


def _matern52_weights(lags: np.ndarray, lengthscale: float) -> np.ndarray:
    """
    Matern 5/2 kernel weights for an array of non-negative integer lags.
    w(d) = (1 + sqrt5*d/l + 5*d^2/(3*l^2)) * exp(-sqrt5*d/l)
    Normalised to sum to 1 over the provided lags.
    """
    sqrt5 = np.sqrt(5.0)
    d = lags / lengthscale
    w = (1.0 + sqrt5 * d + (5.0 / 3.0) * d * d) * np.exp(-sqrt5 * d)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s < 1e-14:
        return np.ones(len(lags)) / max(len(lags), 1)
    return w / s


def _matern_vol_series(sq_ret: np.ndarray,
                       lengthscale: float,
                       lookback: int,
                       min_obs: int) -> np.ndarray:
    """
    Compute per-bar Matern 5/2 kernel-weighted variance (annualized vol).
    At each bar t, use past `lookback` squared log-returns weighted by Matern kernel.
    """
    n = len(sq_ret)
    out = np.full(n, np.nan)

    # Precompute weights for all lags 0..lookback-1
    lags = np.arange(lookback, dtype=float)
    weights_all = _matern52_weights(lags, lengthscale)

    for t in range(n):
        lo = max(0, t - lookback + 1)
        window_sq = sq_ret[lo : t + 1]
        # Lags: t - lo down to 0 (most recent is lag 0)
        # sq_ret[lo] corresponds to lag (t - lo), sq_ret[t] = lag 0
        n_window = len(window_sq)
        if n_window < min_obs:
            continue
        # Weights: lags 0, 1, ..., n_window-1 — index 0 is most recent (sq_ret[t])
        w = weights_all[:n_window][::-1]  # flip so w[0] matches sq_ret[lo]
        wsum = w.sum()
        if wsum < 1e-14:
            continue
        w_norm = w / wsum
        # Check for enough finite observations
        fin = np.isfinite(window_sq)
        if fin.sum() < min_obs:
            continue
        # Zero-fill NaN in squared returns for weighting
        sq_clean = np.where(fin, window_sq, 0.0)
        w_clean  = np.where(fin, w_norm, 0.0)
        w_tot    = w_clean.sum()
        if w_tot < 1e-14:
            continue
        var_est  = np.dot(sq_clean, w_clean) / w_tot
        out[t]   = np.sqrt(max(var_est, 0.0)) * _ANNUAL

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Matern 5/2 kernel-weighted volatility features."""
    close = df["Close"].values.astype(float)
    n = len(close)

    # ── Log returns and squared log returns ───────────────────────────────────
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    sq_ret = log_ret ** 2

    # ── Matern kernel vol at three lengthscales ───────────────────────────────
    vol_short = _matern_vol_series(sq_ret, _LS_SHORT, int(3 * _LS_SHORT), _MIN_OBS)
    vol_mid   = _matern_vol_series(sq_ret, _LS_MID,   _LOOKBACK,           _MIN_OBS)
    vol_long  = _matern_vol_series(sq_ret, _LS_LONG,  _LOOKBACK,           _MIN_OBS)

    # ── Ratio: short-term vs long-term Matern vol ─────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_sl = np.where(vol_long > 1e-8, vol_short / vol_long, np.nan)
    ratio_sl = np.clip(ratio_sl, 0.0, 10.0)

    # ── Matern mid vs standard 20-day EWMA vol ────────────────────────────────
    ewma_var20 = pd.Series(sq_ret).ewm(span=20, min_periods=5, adjust=False).mean().values
    ewma_vol20 = np.sqrt(np.maximum(ewma_var20, 0.0)) * _ANNUAL
    with np.errstate(divide="ignore", invalid="ignore"):
        vs_ewma = np.where(ewma_vol20 > 1e-8, vol_mid / ewma_vol20, np.nan)
    vs_ewma = np.clip(vs_ewma, 0.0, 5.0)

    # ── Write outputs ─────────────────────────────────────────────────────────
    df["mkv_vol_short"]        = vol_short
    df["mkv_vol_mid"]          = vol_mid
    df["mkv_vol_long"]         = vol_long
    df["mkv_ratio_short_long"] = ratio_sl
    df["mkv_vs_ewma20"]        = vs_ewma

    return df
