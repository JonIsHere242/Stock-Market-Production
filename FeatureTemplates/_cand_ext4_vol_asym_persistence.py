"""
Asymmetric volatility persistence (GARCH leverage proxy).

Rolling 90-day OLS of next-day squared return on:
  x1 = today's squared return
  x2 = negative-return indicator * today's squared return   (leverage term)

Produces the asymmetry coefficient (x2 slope) and its 20-day change.
Pure OHLCV, no lookahead.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_vol_asym_persistence",
    "description": (
        "GARCH leverage-effect proxy: rolling 90-day OLS regresses next-day squared "
        "return (y) on (i) today's squared return and (ii) the interaction "
        "negative_day * squared_return (leverage term).  The coefficient on the "
        "leverage term measures whether volatility rises MORE after down days than "
        "after up days of equal magnitude.  Also produces the 20-day change in that "
        "coefficient to capture regime shifts.  Per-ticker time-series proxy for the "
        "cross-sectional GARCH leverage effect.  Source: Round-5 expansion "
        "(xdom2_downside_beta)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_vol_asym_persistence_coef",   # leverage asymmetry coefficient
        "ext4_vol_asym_persistence_slope",  # 20-day change in the coefficient
    ],
    "tags": ["volatility", "asymmetry", "garch", "leverage_effect", "rolling_ols"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom2_downside_beta)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asymmetric volatility persistence per ticker."""
    WINDOW = 90
    SLOPE_LOOKBACK = 20

    # Log returns (current bar only uses Close[t] and Close[t-1] -- causal)
    ret = df["Close"].pct_change()          # r_t = (C_t - C_{t-1}) / C_{t-1}

    # x1: squared return today
    sq_ret = ret ** 2

    # x2: leverage interaction -- negative day indicator * squared return
    neg_ind = (ret < 0).astype(float)
    lev_term = neg_ind * sq_ret

    # y: next-day squared return   -- shift(-1) would be lookahead; we use shift(+1)
    # to align: y[t] = sq_ret[t+1], but we must not use future data.
    # Correct causal framing: at bar t we observe ret[t] and try to predict sq_ret[t+1].
    # In the rolling window ending at t we have past (x1[s], x2[s], y[s]) for s < t.
    # So we align y = sq_ret.shift(-1) -- BUT that IS lookahead in isolation.
    # The safe way: build the lagged feature matrix where y_lagged[t] = sq_ret[t+1]
    # is known at t+1.  We lag the x's by one: x1_lag[t] = sq_ret[t-1],
    # x2_lag[t] = lev_term[t-1], y_curr[t] = sq_ret[t].
    # Regression: y[t] ~ x1_lag[t] + x2_lag[t]  (all past info at decision time t).
    x1 = sq_ret.shift(1)      # yesterday's squared return
    x2 = lev_term.shift(1)    # yesterday's leverage term
    y  = sq_ret               # today's squared return (known at t)

    n = len(df)
    coef_arr = np.full(n, np.nan)

    # Vectorised rolling OLS via numpy sliding windows
    # We need at least WINDOW rows with valid data.
    x1_v = x1.to_numpy(dtype=float)
    x2_v = x2.to_numpy(dtype=float)
    y_v  = y.to_numpy(dtype=float)

    for t in range(WINDOW - 1, n):
        sl = slice(t - WINDOW + 1, t + 1)
        _x1 = x1_v[sl]
        _x2 = x2_v[sl]
        _y  = y_v[sl]

        # Drop NaNs jointly
        mask = np.isfinite(_x1) & np.isfinite(_x2) & np.isfinite(_y)
        if mask.sum() < 10:
            continue

        _x1m = _x1[mask]
        _x2m = _x2[mask]
        _ym  = _y[mask]

        # Build design matrix [const, x1, x2]
        ones = np.ones(mask.sum())
        X = np.column_stack([ones, _x1m, _x2m])

        # OLS via normal equations  (small matrix -- 3x3)
        try:
            XtX = X.T @ X
            Xty = X.T @ _ym
            # Use lstsq for numerical stability
            coeffs, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            coef_arr[t] = coeffs[2]   # leverage asymmetry coefficient
        except (np.linalg.LinAlgError, ValueError):
            pass

    coef_s = pd.Series(coef_arr, index=df.index)

    # Replace inf with nan
    coef_s = coef_s.replace([np.inf, -np.inf], np.nan)

    # 20-day change in the coefficient
    slope_s = coef_s.diff(SLOPE_LOOKBACK)
    slope_s = slope_s.replace([np.inf, -np.inf], np.nan)

    df["ext4_vol_asym_persistence_coef"]  = coef_s.values
    df["ext4_vol_asym_persistence_slope"] = slope_s.values

    return df
