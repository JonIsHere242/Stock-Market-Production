"""
trend_quality.py — trend smoothness / persistence quality (Tier-2).

A trend's *quality* (how clean and persistent it is) discriminates future winners
better than raw momentum: a smooth, low-noise uptrend tends to continue, a jagged
one mean-reverts. Three published lenses, all trailing-only & vectorised:

(1) Rolling R^2 of log-price on time  — the linear "trend strength" used in the
    classic dual-momentum / "Clenow" momentum-quality screen (Andreas Clenow,
    *Stocks on the Move*, 2015): regress log(Close) on a time index over W bars
    and keep the coefficient of determination. High R^2 = straight-line trend.
    Computed in closed form from rolling cov/var (NO rolling.apply).

(2) Trend "Sharpe" — rolling mean daily log-return / rolling std of log-return,
    the information-ratio of the path. Smooth persistent drift -> high value.

(3) Kaufman Efficiency Ratio (Perry Kaufman, *Smarter Trading*, 1995):
        ER = |net move over W| / sum(|daily move|) over W,  in [0,1].
    1 = perfectly efficient straight move, ~0 = choppy noise. The basis of the
    KAMA adaptive moving average.

Windows: 21d (1m) and 63d (3m). Per-ticker, vectorised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINS = [21, 63]
_MINP = {21: 15, 63: 45}


def _rolling_r2_logprice(logp: pd.Series, w: int, minp: int) -> np.ndarray:
    """R^2 of OLS of logp on time-index t=0..w-1, vectorised via rolling moments.

    For y regressed on x where x is a fixed integer ramp, R^2 = cov(x,y)^2 /
    (var(x) * var(y)). x has constant variance over any window, so we only need
    rolling cov(t, y) and rolling var(y).
    """
    n = len(logp)
    t = np.arange(n, dtype=float)
    ts = pd.Series(t, index=logp.index)

    mean_y = logp.rolling(w, min_periods=minp).mean()
    mean_t = ts.rolling(w, min_periods=minp).mean()
    # E[t*y] and E[t^2], E[y^2] over the window
    ty = (ts * logp).rolling(w, min_periods=minp).mean()
    tt = (ts * ts).rolling(w, min_periods=minp).mean()
    yy = (logp * logp).rolling(w, min_periods=minp).mean()

    cov_ty = ty - mean_t * mean_y
    var_t = tt - mean_t * mean_t
    var_y = yy - mean_y * mean_y

    denom = (var_t * var_y).replace(0.0, np.nan)
    r2 = (cov_ty * cov_ty) / denom
    return r2.clip(0.0, 1.0).values


def _signed_r2_slope(logp: pd.Series, w: int, minp: int) -> np.ndarray:
    """R^2 signed by the slope direction (trend strength with direction)."""
    n = len(logp)
    t = pd.Series(np.arange(n, dtype=float), index=logp.index)
    mean_y = logp.rolling(w, min_periods=minp).mean()
    mean_t = t.rolling(w, min_periods=minp).mean()
    ty = (t * logp).rolling(w, min_periods=minp).mean()
    tt = (t * t).rolling(w, min_periods=minp).mean()
    cov_ty = ty - mean_t * mean_y
    var_t = (tt - mean_t * mean_t).replace(0.0, np.nan)
    slope = cov_ty / var_t
    return np.sign(slope.values)


METADATA = {
    "name":        "trend_quality",
    "description": "Trend-smoothness quality: rolling R^2 of log-price~time (Clenow), signed trend strength, trend Sharpe (mean/std of log-rets), and Kaufman efficiency ratio at 21d/63d.",
    "requires":    ["Close"],
    "produces":    [
        "trq_r2_21", "trq_r2_63",
        "trq_trend_strength_63",
        "trq_sharpe_21", "trq_sharpe_63",
        "trq_kaufman_er_21", "trq_kaufman_er_63",
    ],
    "tags":        ["trend", "momentum", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Clenow 2015 / Kaufman 1995)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].clip(lower=1e-8)
    logp = np.log(close)
    logret = logp.diff()

    # (1) Rolling R^2 of log-price on time.
    df["trq_r2_21"] = _rolling_r2_logprice(logp, 21, _MINP[21])
    r2_63 = _rolling_r2_logprice(logp, 63, _MINP[63])
    df["trq_r2_63"] = r2_63
    # signed trend strength = R^2 with the sign of the regression slope (63d).
    df["trq_trend_strength_63"] = r2_63 * _signed_r2_slope(logp, 63, _MINP[63])

    # (2) Trend Sharpe = rolling mean log-ret / rolling std log-ret.
    for w in _WINS:
        mp = _MINP[w]
        mu = logret.rolling(w, min_periods=mp).mean()
        sd = logret.rolling(w, min_periods=mp).std()
        df[f"trq_sharpe_{w}"] = (mu / sd.replace(0.0, np.nan)).clip(-5.0, 5.0).values

    # (3) Kaufman efficiency ratio = |net move| / sum(|daily move|) over W.
    abs_ret = logret.abs()
    for w in _WINS:
        mp = _MINP[w]
        net = (logp - logp.shift(w)).abs()
        churn = abs_ret.rolling(w, min_periods=mp).sum().replace(0.0, np.nan)
        df[f"trq_kaufman_er_{w}"] = (net / churn).clip(0.0, 1.0).values

    return df
