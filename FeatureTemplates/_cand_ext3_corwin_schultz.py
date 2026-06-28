"""
Corwin-Schultz (2012) high-low bid-ask spread estimator.

Reference: Corwin & Schultz, "A Simple Way to Estimate Bid-Ask Spreads from Daily
High and Low Prices", Journal of Finance, 2012.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_corwin_schultz",
    "description": (
        "Corwin & Schultz (2012) high-low bid-ask spread estimator. "
        "Uses consecutive-day High/Low to build beta (sum of squared log(H/L) "
        "over 2 days), gamma (squared log of 2-day H / 2-day L), then alpha "
        "and implied spread S = 2*(exp(alpha)-1)/(1+exp(alpha)). "
        "Negative daily estimates are clipped to 0 (a known CS property). "
        "Produces: rolling 21-day mean spread (level) and its 60-day OLS slope "
        "(trend/momentum of liquidity cost). Pure OHLC, per-ticker."
    ),
    "requires": ["High", "Low"],
    "produces": [
        "ext3_corwin_schultz_spread21",   # 21-day rolling mean CS spread (clipped >=0)
        "ext3_corwin_schultz_trend60",    # 60-day OLS slope of daily CS spread
    ],
    "tags": ["microstructure", "bid_ask", "liquidity", "spread", "corwin_schultz"],
    "version": "1.0",
    "author": "Corwin & Schultz (2012), Journal of Finance — implemented per spec ext3_corwin_schultz",
}

_SQRT2 = np.sqrt(2.0)
_DENOM_ALPHA = 3.0 - 2.0 * _SQRT2   # 3 - 2√2 ≈ 0.1716


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Corwin-Schultz spread features for a single ticker."""

    high = df["High"].values.astype(np.float64)
    low  = df["Low"].values.astype(np.float64)
    n    = len(df)

    # --- daily log(H/L) squared ------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.where(low > 0.0, np.log(high / low), np.nan)
    log_hl_sq = log_hl ** 2  # (ln H_t/L_t)^2

    # --- beta: sum of squared log(H/L) over day t and day t-1 -------------
    # beta_t = (ln H_{t-1}/L_{t-1})^2 + (ln H_t/L_t)^2
    beta = np.empty(n)
    beta[:] = np.nan
    beta[1:] = log_hl_sq[1:] + log_hl_sq[:-1]

    # --- gamma: (ln(max(H_{t-1},H_t) / min(L_{t-1},L_t)))^2 ---------------
    two_day_high = np.empty(n)
    two_day_low  = np.empty(n)
    two_day_high[:] = np.nan
    two_day_low[:]  = np.nan
    two_day_high[1:] = np.maximum(high[1:], high[:-1])
    two_day_low[1:]  = np.minimum(low[1:],  low[:-1])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_2day = np.where(two_day_low > 0.0,
                            np.log(two_day_high / two_day_low),
                            np.nan)
    gamma = log_2day ** 2

    # --- alpha: Corwin & Schultz eq. (10) -----------------------------------
    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
    #         - sqrt(gamma / (3 - 2*sqrt(2)))
    with np.errstate(invalid="ignore"):
        sqrt_2beta = np.sqrt(2.0 * beta)
        sqrt_beta  = np.sqrt(beta)
        sqrt_gamma_term = np.sqrt(np.where(gamma >= 0.0,
                                           gamma / _DENOM_ALPHA,
                                           np.nan))

    alpha = (sqrt_2beta - sqrt_beta) / _DENOM_ALPHA - sqrt_gamma_term

    # --- spread S = 2*(exp(alpha)-1)/(1+exp(alpha)) -------------------------
    with np.errstate(over="ignore", invalid="ignore"):
        exp_a = np.exp(alpha)
        denom = 1.0 + exp_a
        spread_raw = np.where(
            np.isfinite(exp_a) & (denom != 0.0),
            2.0 * (exp_a - 1.0) / denom,
            np.nan,
        )

    # Clip negatives to 0 (CS 2012 recommendation; negative values are an
    # artefact of the approximation, not a meaningful signal)
    spread_daily = np.where(spread_raw < 0.0, 0.0, spread_raw)
    # Propagate NaN
    spread_daily = np.where(np.isnan(spread_raw), np.nan, spread_daily)

    spread_s = pd.Series(spread_daily, index=df.index)

    # --- feature 1: 21-day rolling mean spread ------------------------------
    spread21 = spread_s.rolling(21, min_periods=10).mean()

    # --- feature 2: 60-day OLS slope of daily spread ------------------------
    # slope via rolling covariance / variance of position index
    # Using a vectorised approach: cov(x, y) / var(x) where x = [0..59]
    window = 60
    min_p  = 20

    def _rolling_ols_slope(series: pd.Series, w: int, mp: int) -> pd.Series:
        """Vectorised rolling OLS slope via cov/var of integer positions."""
        # x = [0, 1, ..., w-1]; E[x] = (w-1)/2; Var(x) = (w^2-1)/12
        # slope = cov(pos, y) / var(pos)
        # We use rolling mean to get E[y] and rolling mean of pos*y to get E[pos*y].
        # pos relative to window start = 0..w-1
        # cov(pos, y) = E[pos*y] - E[pos]*E[y]

        y = series.values.astype(np.float64)
        result = np.full(len(y), np.nan)

        # Pre-compute weights: x = 0..w-1
        x_arr = np.arange(w, dtype=np.float64)
        x_mean = x_arr.mean()
        x_var  = x_arr.var()  # biased variance of 0..w-1

        if x_var == 0.0:
            return pd.Series(result, index=series.index)

        for i in range(w - 1, len(y)):
            window_vals = y[i - w + 1: i + 1]
            valid = ~np.isnan(window_vals)
            if valid.sum() < mp:
                continue
            # Use only positions where y is not NaN
            x_w = x_arr[valid]
            y_w = window_vals[valid]
            # OLS slope
            xm = x_w.mean()
            ym = y_w.mean()
            cov_xy = ((x_w - xm) * (y_w - ym)).mean()
            var_x  = ((x_w - xm) ** 2).mean()
            if var_x == 0.0:
                continue
            result[i] = cov_xy / var_x

        return pd.Series(result, index=series.index)

    # The python loop is over ~700 rows at most; acceptable for a 60-row window.
    trend60 = _rolling_ols_slope(spread_s, window, min_p)

    df["ext3_corwin_schultz_spread21"] = spread21.values
    df["ext3_corwin_schultz_trend60"]  = trend60.values

    return df
