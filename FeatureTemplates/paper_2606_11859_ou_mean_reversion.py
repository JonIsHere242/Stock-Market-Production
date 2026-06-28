"""
Ornstein-Uhlenbeck mean-reversion features derived from:
  "Scenario Generation for Time Series and Curves: A Comparison of
   Nonparametric and Semiparametric Bootstrap" (arXiv 2606.11859).

The paper uses autoregressive / mean-reverting parametric structures
(AR and OU/Vasicek) as the backbone of semiparametric bootstrap.

Key insight: semiparametric methods fit AR(1)/OU to the *level* process,
then resample residuals. We extract the structural OU parameters:

  - phi (AR1 coefficient on log-price levels) — momentum vs reversion
  - theta = 1 - phi — reversion speed (positive = mean-reverting)
  - Half-life = ln(2)/max(theta, eps)
  - Residual from OU equilibrium (z-score)
  - Residual momentum (cumulative last-k residuals)

All computed via rolling OLS on log-price levels. Multiple windows.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_11859_ou_mean_reversion",
    "description": (
        "Ornstein-Uhlenbeck reversion speed, half-life and residual z-scores "
        "from rolling AR(1) OLS on log-price levels; arXiv 2606.11859."
    ),
    "requires":    ["Close"],
    "produces": [
        "ou_phi_20d",
        "ou_phi_60d",
        "ou_theta_signed_20d",
        "ou_theta_signed_60d",
        "ou_halflife_20d",
        "ou_halflife_60d",
        "ou_resid_zscore_20d",
        "ou_resid_zscore_60d",
    ],
    "tags":        ["mean_reversion", "statistical", "experimental"],
    "version":     "1.1",
    "author":      "paper:2606.11859",
}


def _rolling_ar1_params(log_price: np.ndarray, window: int, min_obs: int):
    """
    Rolling AR(1) on log-price levels:
      x_t = c + phi * x_{t-1} + eps
    Returns arrays: phi, mu_hat (= c/(1-phi)), resid_zscore
    """
    n = len(log_price)
    phi_arr    = np.full(n, np.nan)
    mu_arr     = np.full(n, np.nan)
    resid_z    = np.full(n, np.nan)
    resid_std_arr = np.full(n, np.nan)

    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        xi = log_price[start: i + 1]
        mask = ~np.isnan(xi)
        xi = xi[mask]
        k = len(xi)
        if k < min_obs:
            continue

        y = xi[1:]      # x_t
        x = xi[:-1]     # x_{t-1}
        if len(y) < 4:
            continue

        # OLS with intercept
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx < 1e-12:
            continue
        phi = ((x - x_mean) * (y - y_mean)).sum() / ss_xx
        c   = y_mean - phi * x_mean

        # Long-run equilibrium: mu = c / (1 - phi)
        if abs(1 - phi) < 1e-6:
            mu_hat = x_mean
        else:
            mu_hat = c / (1 - phi)

        phi_arr[i] = phi
        mu_arr[i]  = mu_hat

        # Residual of current x_i from equilibrium, z-scored by residual std
        resids = y - (c + phi * x)
        std_r = resids.std()
        if std_r < 1e-12:
            continue
        # Current point's deviation from equilibrium
        cur_dev = log_price[i] - mu_hat
        resid_z[i] = cur_dev / std_r
        resid_std_arr[i] = std_r

    return phi_arr, mu_arr, resid_z


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_p = np.log(df["Close"].replace(0, np.nan)).values.astype(np.float64)
    n = len(df)

    for window, min_obs in [(20, 10), (60, 20)]:
        phi_arr, mu_arr, resid_z = _rolling_ar1_params(log_p, window, min_obs)

        phi_s = pd.Series(phi_arr, index=df.index)
        # theta = 1 - phi: positive = mean-reverting, negative = explosive/trending
        theta_s = 1.0 - phi_s
        # Half-life: ln(2)/theta, only meaningful when phi in (0,1), i.e. theta in (0,1)
        # When theta <= 0 (trending), half-life is undefined/negative -> NaN
        hl = np.log(2) / theta_s.where(theta_s > 0.001)
        hl = hl.clip(upper=252)

        df[f"ou_phi_{window}d"]           = phi_s
        df[f"ou_theta_signed_{window}d"]  = theta_s
        df[f"ou_halflife_{window}d"]      = hl
        df[f"ou_resid_zscore_{window}d"]  = pd.Series(resid_z, index=df.index)

    return df
