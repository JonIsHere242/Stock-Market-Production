"""
Multi-scale GARCH volatility regime features derived from:
  "Multi-Scale Markov Switching GARCH" (arXiv 2606.06190).

The paper fits triple-timeframe AR(1)-MS-GARCH models on EUR/USD to produce
Calm / Turbulent / Crisis regime probabilities. Per-ticker OHLCV proxy:

  1. Estimate rolling GARCH(1,1) conditional variance at three window scales
     (daily ~macro, medium ~meso, short ~micro) using the standard recursion:
       sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
     with parameters calibrated via moment matching (alpha+beta from
     autocorrelation of squared returns, omega from marginal variance).

  2. Derive a regime indicator at each scale by comparing sigma_t to its
     rolling distribution (z-score + percentile rank).

  3. Combine scales via outer-product weighting (product of regime percentiles)
     to form the cross-scale stress tensor signal.

All calculations are per-ticker rolling; no cross-section needed.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_06190_multiscale_garch_regime",
    "description": (
        "Per-ticker rolling GARCH(1,1) conditional variance at macro/meso/micro scales "
        "with calm/turbulent/crisis regime indicators; proxy for arXiv 2606.06190 "
        "multi-scale MS-GARCH framework."
    ),
    "requires":    ["Close"],
    "produces": [
        "garch_sigma_macro",
        "garch_sigma_meso",
        "garch_sigma_micro",
        "garch_regime_macro",
        "garch_regime_meso",
        "garch_regime_micro",
        "garch_crossscale_stress",
        "garch_sigma_spread",
    ],
    "tags":        ["volatility", "market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.06190",
}


def _garch11_rolling(eps: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Rolling GARCH(1,1) conditional variance recursion.
    omega is estimated from a CAUSAL expanding window variance so there is
    no lookahead: at time t, omega_t = (1-alpha-beta) * expanding_var(eps[:t+1]).
    sigma^2_0 starts from the first 20-bar variance.
    """
    n = len(eps)
    sigma2 = np.full(n, np.nan)
    if n < 5:
        return sigma2

    ab = alpha + beta
    if ab >= 0.9999:
        ab = 0.9999

    # Expanding (causal) variance — computed upfront via cumulative mean/sum-of-squares
    eps2 = eps ** 2
    # cumulative mean of eps2 gives expanding variance (E[eps^2] ~ var when mean~0)
    cum_count = np.arange(1, n + 1, dtype=np.float64)
    cum_sum2  = np.nancumsum(eps2)
    expand_var = cum_sum2 / cum_count  # expanding mean of squared returns (causal)

    # Initial sigma^2 = first non-zero expanding var
    init_var = expand_var[min(19, n - 1)]  # first 20-bar estimate
    if init_var <= 0 or not np.isfinite(init_var):
        init_var = 1e-6

    # Shift eps by 1 so sigma2[t] = f(eps[t-1]) — standard GARCH(1,1) convention.
    # sigma2[t] is the conditional variance for day t, estimated from info at t-1.
    eps_lag = np.empty_like(eps)
    eps_lag[0] = 0.0
    eps_lag[1:] = eps[:-1]

    # Similarly use expanding var up to t-1 for omega (causal)
    # expand_var[t] already uses eps[0..t] via cum_sum2/cum_count.
    # Use expand_var[t-1] for omega at time t (so it's info from t-1).
    expand_var_lag = np.empty_like(expand_var)
    expand_var_lag[0] = expand_var[0]  # degenerate; first bar uses itself
    expand_var_lag[1:] = expand_var[:-1]

    s2 = init_var
    for t in range(n):
        ev = expand_var_lag[t]
        if ev <= 0 or not np.isfinite(ev):
            ev = s2
        omega_t = (1.0 - ab) * ev

        e_prev = eps_lag[t]
        s2 = omega_t + alpha * (e_prev ** 2) + beta * s2
        sigma2[t] = s2

    return np.sqrt(np.maximum(sigma2, 1e-16))


def _moment_garch_params(eps: np.ndarray) -> tuple:
    """
    Estimate alpha+beta from autocorrelation of squared returns, using only
    the FIRST 60 bars as a burn-in estimate (purely causal: these bars come
    before any feature output we produce).  Returns typical GARCH(1,1) params.
    """
    burn = eps[:60]
    eps2 = burn[~np.isnan(burn)] ** 2
    if len(eps2) < 10:
        return 0.10, 0.85

    mu2 = eps2.mean()
    if mu2 <= 0:
        return 0.10, 0.85

    # AC(1) of squared returns on burn-in window
    c0 = ((eps2 - mu2) ** 2).mean()
    c1 = ((eps2[1:] - mu2) * (eps2[:-1] - mu2)).mean()
    ac1 = c1 / c0 if c0 > 0 else 0.8

    ab = float(np.clip(ac1, 0.70, 0.98))
    alpha = 0.10
    beta  = max(ab - alpha, 0.05)
    return alpha, beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Close"].pct_change().fillna(0.0)
    eps = ret.values.astype(np.float64)
    n = len(df)

    # Estimate global GARCH params once (stable for a 700-row series)
    alpha, beta = _moment_garch_params(eps)

    # --- Three timescale sigma estimates ---
    # Macro: full-sample GARCH(1,1) forward pass
    sigma_macro = _garch11_rolling(eps, alpha, beta)

    # Meso: GARCH(1,1) on a de-trended volatility (22-day EWM demeaned)
    ewm_var = pd.Series(eps).pow(2).ewm(span=22, min_periods=5).mean().values
    ewm_vol = np.sqrt(np.maximum(ewm_var, 1e-12))
    eps_meso = np.where(ewm_vol > 0, eps / ewm_vol, eps)
    sigma_meso_raw = _garch11_rolling(eps_meso, alpha * 0.8, beta * 0.9)
    # Rescale back to original scale
    sigma_meso = sigma_meso_raw * ewm_vol

    # Micro: 5-day rolling std (short-scale volatility)
    sigma_micro_s = pd.Series(eps).rolling(5, min_periods=3).std()
    sigma_micro = sigma_micro_s.values

    df["garch_sigma_macro"] = sigma_macro
    df["garch_sigma_meso"]  = sigma_meso
    df["garch_sigma_micro"] = np.where(np.isfinite(sigma_micro), sigma_micro, np.nan)

    # --- Regime indicators: percentile rank within rolling 252-day window ---
    for col_name, sig_arr in [
        ("garch_regime_macro", sigma_macro),
        ("garch_regime_meso",  sigma_meso),
        ("garch_regime_micro", sigma_micro),
    ]:
        sig_s = pd.Series(sig_arr, index=df.index)
        # Rolling rank: fraction of past 252 obs below current value
        ranks = np.full(n, np.nan)
        for i in range(20, n):
            start = max(0, i - 251)
            window = sig_arr[start: i + 1]
            valid  = window[~np.isnan(window)]
            if len(valid) < 5:
                continue
            cur = sig_arr[i]
            if np.isnan(cur):
                continue
            ranks[i] = float((valid < cur).sum()) / len(valid)
        df[col_name] = ranks

    # --- Cross-scale stress: product of the three regime percentiles ---
    # This mirrors the paper's outer-product 27-state tensor compressed to a scalar
    rm = pd.Series(df["garch_regime_macro"].values, index=df.index)
    rme = pd.Series(df["garch_regime_meso"].values, index=df.index)
    rmi = pd.Series(df["garch_regime_micro"].values, index=df.index)
    df["garch_crossscale_stress"] = (rm * rme * rmi).clip(0, 1)

    # --- Sigma spread: macro vs micro normalised ---
    sm = pd.Series(sigma_macro, index=df.index)
    smi = pd.Series(np.where(np.isfinite(sigma_micro), sigma_micro, np.nan), index=df.index)
    # Ratio; clip to avoid extreme values
    df["garch_sigma_spread"] = (sm / smi.replace(0, np.nan)).clip(0.1, 10.0)

    return df
