"""
Pastor-Stambaugh (2003) liquidity gamma -- per-ticker rolling OLS proxy.

The canonical PS measure is cross-sectional (monthly rank across stocks), but
the reversal coefficient gamma is estimable per-ticker from time-series alone.
The economic content is preserved: a more-negative gamma means that signed
order flow forecasts a stronger next-day reversal, implying lower liquidity.

Reference: Pastor, L. and Stambaugh, R. F. (2003). "Liquidity Risk and Expected
Stock Returns." Journal of Political Economy, 111(3), 642-685.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext3_pastor_stambaugh",
    "description": (
        "Per-ticker rolling Pastor-Stambaugh (2003) liquidity reversal coefficient "
        "(gamma). Estimated via 60-day rolling OLS: r_{t+1} ~ a + b*r_t + gamma*(sign(r_t)*dvol_t), "
        "where dvol_t = Close_t * Volume_t. More-negative gamma = less liquid stock. "
        "Produces gamma level and its 20-day change (momentum of liquidity). "
        "Per-ticker time-series proxy; lacks cross-sectional rank normalisation used "
        "in the original but preserves the within-stock reversal-coefficient signal."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext3_pastor_stambaugh_gamma",       # rolling 60d OLS reversal coefficient
        "ext3_pastor_stambaugh_gamma_chg20", # 20-day change in gamma (liquidity dynamics)
        "ext3_pastor_stambaugh_signed_dvol", # signed dollar volume (input signal; useful standalone)
    ],
    "tags": ["microstructure", "liquidity", "reversal", "pastor_stambaugh"],
    "version": "1.0",
    "author": "Pastor & Stambaugh (2003), JPE 111(3) 642-685. Per-ticker proxy implementation.",
}

# Rolling window parameters
_WINDOW = 60   # days for OLS gamma estimation
_CHG_LAG = 20  # days for gamma change


def _rolling_ols_gamma(ret: np.ndarray, signed_dvol: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling OLS gamma coefficient for the regression:
        ret[t+1] = a + b*ret[t] + gamma*signed_dvol[t] + eps

    For each position i (using rows i-window .. i-1 as in-sample data):
      y  = ret[i-window+1 : i+1]          length = window
      x1 = ret[i-window   : i  ]          length = window  (lagged return)
      x2 = signed_dvol[i-window : i]      length = window  (signed dollar vol)

    Returns array of gamma estimates, NaN for insufficient history.
    Note: all data used are PAST (index <= i), so no lookahead.
    """
    n = len(ret)
    gamma = np.full(n, np.nan)

    for i in range(window, n):
        # y = next-day returns for days [i-window .. i-1]
        # x = same-day ret and signed_dvol for days [i-window .. i-1]
        y = ret[i - window + 1 : i + 1]      # shape (window,)  -- day i is the last
        r_lag = ret[i - window : i]           # shape (window,)
        sdv = signed_dvol[i - window : i]     # shape (window,)

        # Build design matrix [1, r_t, signed_dvol_t]
        ones = np.ones(window)
        X = np.column_stack([ones, r_lag, sdv])

        # Guard: drop rows with NaN in any column
        mask = np.isfinite(y) & np.isfinite(r_lag) & np.isfinite(sdv)
        if mask.sum() < 10:  # need at least 10 clean observations
            continue

        Xm = X[mask]
        ym = y[mask]

        # OLS via normal equations: (X'X)^{-1} X'y
        try:
            XtX = Xm.T @ Xm
            Xty = Xm.T @ ym
            # Use lstsq for numerical stability
            coeffs, _, rank, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            if rank < 3:
                continue
            # coeffs = [a, b, gamma]
            gamma[i] = coeffs[2]
        except (np.linalg.LinAlgError, ValueError):
            continue

    return gamma


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # --- daily close-to-close returns ---
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Returns: (Close[t] - Close[t-1]) / Close[t-1]
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = np.where(
            close[:-1] != 0,
            (close[1:] - close[:-1]) / close[:-1],
            np.nan,
        )

    # --- dollar volume ---
    dvol = close * volume  # scalar, no lookahead

    # --- signed dollar volume: sign(ret_t) * dvol_t ---
    # sign(0) = 0 by numpy convention, which is fine
    signed_dvol = np.sign(ret) * dvol
    # first element NaN (ret[0] is NaN, sign propagates)
    # actually np.sign(nan) = nan, good

    # --- rolling gamma ---
    gamma = _rolling_ols_gamma(ret, signed_dvol, _WINDOW)

    # --- 20-day change in gamma ---
    gamma_ser = pd.Series(gamma)
    gamma_chg20 = (gamma_ser - gamma_ser.shift(_CHG_LAG)).to_numpy()

    # --- guard: replace inf with nan ---
    with np.errstate(invalid="ignore"):
        gamma = np.where(np.isfinite(gamma), gamma, np.nan)
        gamma_chg20 = np.where(np.isfinite(gamma_chg20), gamma_chg20, np.nan)
        signed_dvol_out = np.where(np.isfinite(signed_dvol), signed_dvol, np.nan)

    df["ext3_pastor_stambaugh_gamma"] = gamma
    df["ext3_pastor_stambaugh_gamma_chg20"] = gamma_chg20
    df["ext3_pastor_stambaugh_signed_dvol"] = signed_dvol_out

    return df
