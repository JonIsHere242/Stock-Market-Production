"""
ext4_vol_mean_reversion — Volatility mean-reversion speed & gap.

Rolling AR(1) fit on the 10-day realized-vol series over a 120-day window.
Produces:
  - AR(1) coefficient (persistence); (1 - coef) = reversion speed
  - Current vol minus its 120-day mean (vol gap; expected to revert)
  - Reversion speed proxy (1 - AR1 coefficient)

All per-ticker, pure OHLCV, no lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext4_vol_mean_reversion",
    "description": (
        "Rolling 120-day AR(1) fit on the 10-day realised-volatility series "
        "(log-return std). Produces: AR(1) coefficient (persistence of vol "
        "shocks), reversion speed (1 - AR1 coef), and the current vol gap "
        "relative to its 120-day rolling mean. High reversion speed + positive "
        "vol gap => vol likely to compress soon. Pure OHLCV, per-ticker proxy "
        "for mean-reversion dynamics in volatility."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_vol_mean_reversion_ar1",
        "ext4_vol_mean_reversion_speed",
        "ext4_vol_mean_reversion_gap",
    ],
    "tags": ["volatility", "mean_reversion", "ar1", "realized_vol", "ohlcv"],
    "version": "1.0.0",
    "author": "Round-5 expansion (xdom_allan_variance); spec ext4_vol_mean_reversion",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AR(1) coefficient, reversion speed, and vol gap on rolling basis."""

    # ------------------------------------------------------------------ #
    # 1. Build the 10-day realized-vol series (annualised log-return std)  #
    # ------------------------------------------------------------------ #
    log_ret = np.log(df["Close"] / df["Close"].shift(1))   # no lookahead

    # 10-day rolling std of log returns (use min_periods=5 to tolerate gaps)
    rvol_10 = log_ret.rolling(window=10, min_periods=5).std()

    # ------------------------------------------------------------------ #
    # 2. Rolling 120-day AR(1) on rvol_10                                 #
    #    AR(1): y_t = c + phi * y_{t-1} + eps                             #
    #    We estimate phi via OLS: cov(y_t, y_{t-1}) / var(y_{t-1})        #
    #    Vectorised: for each window compute rolling cross-moments.        #
    # ------------------------------------------------------------------ #
    AR_WINDOW = 120

    # We need y and y_lag = y.shift(1) aligned
    y = rvol_10
    y_lag = rvol_10.shift(1)

    # Rolling moments: E[y], E[y_lag], E[y*y_lag], E[y_lag^2]
    # Using min_periods = half the window so we get values after enough data.
    min_p = AR_WINDOW // 2

    roll_ey     = y.rolling(window=AR_WINDOW, min_periods=min_p).mean()
    roll_eylag  = y_lag.rolling(window=AR_WINDOW, min_periods=min_p).mean()
    roll_eyylag = (y * y_lag).rolling(window=AR_WINDOW, min_periods=min_p).mean()
    roll_eylag2 = (y_lag ** 2).rolling(window=AR_WINDOW, min_periods=min_p).mean()

    # phi = Cov(y, y_lag) / Var(y_lag)
    #     = (E[y*y_lag] - E[y]*E[y_lag]) / (E[y_lag^2] - E[y_lag]^2)
    cov_num = roll_eyylag - roll_ey * roll_eylag
    var_den = roll_eylag2 - roll_eylag ** 2

    # Guard against zero or near-zero denominator
    var_den_safe = var_den.where(var_den.abs() > 1e-12, other=np.nan)
    ar1_coef = cov_num / var_den_safe

    # Clip to a reasonable range [-1.5, 1.5]; extreme values signal instability
    ar1_coef = ar1_coef.clip(-1.5, 1.5)

    # ------------------------------------------------------------------ #
    # 3. Reversion speed = 1 - AR(1) coefficient                          #
    #    Higher => faster mean-reversion; negative => explosive regime     #
    # ------------------------------------------------------------------ #
    rev_speed = 1.0 - ar1_coef

    # ------------------------------------------------------------------ #
    # 4. Vol gap = current rvol_10 minus its 120-day rolling mean          #
    #    Positive => vol above average (expected to fall if mean-reverting) #
    # ------------------------------------------------------------------ #
    rvol_mean = rvol_10.rolling(window=AR_WINDOW, min_periods=min_p).mean()
    vol_gap   = rvol_10 - rvol_mean

    # ------------------------------------------------------------------ #
    # 5. Assign produced columns                                           #
    # ------------------------------------------------------------------ #
    df["ext4_vol_mean_reversion_ar1"]   = ar1_coef
    df["ext4_vol_mean_reversion_speed"] = rev_speed
    df["ext4_vol_mean_reversion_gap"]   = vol_gap

    return df
