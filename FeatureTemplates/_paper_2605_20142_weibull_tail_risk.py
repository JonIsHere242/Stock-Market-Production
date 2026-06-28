"""
Mirrored-Weibull tail risk features.

Based on arXiv 2605.20142: "Mining Financial Data using Mixtures of Mirrored
Weibull Distributions".

The paper fits a mixture of mirrored Weibull (MMW) distributions to stock
return data and uses the fitted model to estimate Value-at-Risk (VaR). The key
advantage of the Weibull family is its flexibility to capture heavy tails and
asymmetry in financial returns — better than Gaussian or t-mixtures.

A single Weibull distribution on the positive half-line has PDF:
  f(x; k, λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)   x >= 0

A "mirrored Weibull" mirrors this to the negative half, modelling losses.
We fit Weibull shape (k) and scale (λ) to the rolling distribution of *losses*
(negative returns) using MLE via moment-matching (no scipy required):

  Method-of-moments for Weibull: given mean m and std s of |neg-returns|,
    CV = s/m => k = (CV)^{-1.086}  [Teimouri & Gupta approximation]
    λ = m / Gamma(1 + 1/k)
  where Gamma is approximated via Lanczos series.

Per-ticker features:
  1. wbl_shape_k_60d       — Weibull shape of loss distribution (k<1: heavy tail)
  2. wbl_scale_lam_60d     — Weibull scale λ (typical loss magnitude)
  3. wbl_var95_60d         — Rolling 95% VaR estimate from fitted Weibull
  4. wbl_var99_60d         — Rolling 99% VaR estimate
  5. wbl_tail_index_60d    — 1/k (Pareto tail index proxy; higher = heavier tail)
  6. wbl_var_surprise_5d   — Realised 5-day loss vs Weibull VaR95 (exceedance)
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2605_20142_weibull_tail_risk",
    "description": (
        "Rolling mirrored-Weibull fit to loss returns for flexible VaR estimation "
        "(shape k, scale λ, VaR95/99, tail index); arXiv 2605.20142. "
        "Moment-matching MLE; no scipy needed."
    ),
    "requires": ["Close"],
    "produces": [
        "wbl_shape_k_60d",
        "wbl_scale_lam_60d",
        "wbl_var95_60d",
        "wbl_var99_60d",
        "wbl_tail_index_60d",
        "wbl_var_surprise_5d",
    ],
    "tags": ["volatility", "risk", "statistical", "experimental"],
    "version": "1.0",
    "author": "paper:2605.20142",
}

_WINDOW = 60
_MIN_OBS = 20
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Lanczos approximation for Gamma function (avoids scipy)
# ---------------------------------------------------------------------------
_LG_G = 7
_LG_C = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def _gamma(z: float) -> float:
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * _gamma(1.0 - z))
    z -= 1.0
    x = _LG_C[0]
    for i in range(1, _LG_G + 2):
        x += _LG_C[i] / (z + i)
    t = z + _LG_G + 0.5
    return np.sqrt(2 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x


def _fit_weibull_mom(losses: np.ndarray):
    """
    Method-of-moments Weibull fit on positive loss values.
    Returns (k, lam) or (nan, nan) if insufficient data.

    Uses the Teimouri-Gupta CV-based approximation:
        k ≈ (CV)^{-1.086}   (works well for CV in [0.2, 2.0])
        λ = mean / Γ(1 + 1/k)
    """
    losses = losses[np.isfinite(losses) & (losses > 0)]
    if len(losses) < 5:
        return np.nan, np.nan
    m = losses.mean()
    s = losses.std(ddof=1)
    if m < _EPS or s < _EPS:
        return np.nan, np.nan
    cv = s / m
    if cv < 0.05 or cv > 5.0:
        return np.nan, np.nan
    k = cv ** (-1.086)
    k = max(0.1, min(k, 20.0))
    try:
        g = _gamma(1.0 + 1.0 / k)
    except Exception:
        g = 1.0
    if g < _EPS:
        return np.nan, np.nan
    lam = m / g
    return k, lam


def _weibull_var(k: float, lam: float, p: float) -> float:
    """
    Weibull quantile (VaR at level p) = λ * (-ln(1-p))^(1/k).
    p = 0.95 gives 95% VaR (the loss level exceeded with prob 5%).
    """
    if not (np.isfinite(k) and np.isfinite(lam) and k > 0 and lam > 0):
        return np.nan
    return lam * ((-np.log(1.0 - p)) ** (1.0 / k))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns (positive = gain, negative = loss)
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values.astype(np.float64)

    n = len(log_ret)
    k_arr = np.full(n, np.nan)
    lam_arr = np.full(n, np.nan)
    var95_arr = np.full(n, np.nan)
    var99_arr = np.full(n, np.nan)

    for i in range(_WINDOW - 1, n):
        start = i - _WINDOW + 1
        seg = log_ret[start: i + 1]
        # Losses = negative returns flipped to positive
        losses = -seg[seg < 0]
        k, lam = _fit_weibull_mom(losses)
        k_arr[i] = k
        lam_arr[i] = lam
        var95_arr[i] = _weibull_var(k, lam, 0.95)
        var99_arr[i] = _weibull_var(k, lam, 0.99)

    df["wbl_shape_k_60d"] = k_arr
    df["wbl_scale_lam_60d"] = lam_arr
    df["wbl_var95_60d"] = var95_arr
    df["wbl_var99_60d"] = var99_arr
    df["wbl_tail_index_60d"] = np.where(k_arr > 0, 1.0 / k_arr, np.nan)

    # VaR surprise: actual 5-day worst daily loss vs lagged VaR95 forecast
    log_ret_s = pd.Series(log_ret, index=df.index)
    worst_5d = (-log_ret_s).rolling(5, min_periods=2).max()
    lagged_var95 = pd.Series(var95_arr, index=df.index).shift(1)
    df["wbl_var_surprise_5d"] = worst_5d - lagged_var95

    return df
