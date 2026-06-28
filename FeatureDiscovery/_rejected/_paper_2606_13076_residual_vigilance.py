"""
Residual structure vigilance features derived from:
  "Detecting Explanatory Insufficiency in Learned Representations:
   A Framework for Representational Vigilance" (arXiv 2606.13076)

The paper (VER — Vigilant Evaluator of Representations) formalises detection
of PERSISTENT RESIDUAL STRUCTURES that simple representations miss.
The key diagnostic steps:
  1. Fit a baseline (simple) representation/model to the series.
  2. Compute residuals.
  3. Check residuals for PERSISTENT STRUCTURE (autocorrelation, non-linearity,
     heteroscedasticity) — i.e., unexplained patterns the model misses.

Applied to OHLCV: fit a simple AR(1) model to log-returns, compute residuals,
then measure multiple forms of residual structure:
  - Residual autocorrelation (Durbin-Watson proxy via lag-1 ACF)
  - Residual heteroscedasticity (ARCH effect: squared-residual autocorrelation)
  - Residual non-linearity (sign autocorrelation: serial sign correlation)
  - Residual tail persistence (run-length of same-sign residuals)
  - Ljung-Box-style Q statistic (sum of squared ACFs over lags 1-5)
  - "Vigilance signal": composite score of all residual structure measures
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13076_residual_vigilance",
    "description": (
        "AR(1) residual structure vigilance features: autocorrelation, ARCH effects, "
        "sign persistence, Ljung-Box Q; based on arXiv 2606.13076 VER framework."
    ),
    "requires":    ["Close"],
    "produces": [
        "ver_resid_acf1_32d",
        "ver_resid_acf1_64d",
        "ver_arch_effect_32d",
        "ver_sign_acf_32d",
        "ver_lb_q5_32d",
        "ver_lb_q5_64d",
        "ver_vigilance_signal",
    ],
    "tags":        ["mean_reversion", "volatility", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13076",
}


def _ar1_residuals(x: np.ndarray) -> np.ndarray:
    """
    Fit rolling AR(1) with OLS:  x[t] = a + b * x[t-1] + e[t].
    Returns residual series e.
    """
    n = len(x)
    if n < 2:
        return np.full(n, np.nan)
    xlag = np.roll(x, 1)
    xlag[0] = np.nan
    mask = np.isfinite(x) & np.isfinite(xlag)
    resid = np.full(n, np.nan)
    if mask.sum() < 4:
        return resid
    # Simple vectorised OLS (global; residuals used in rolling fashion below)
    xl = xlag[mask]
    xc = x[mask]
    xl_mean = xl.mean()
    xc_mean = xc.mean()
    b = ((xl - xl_mean) * (xc - xc_mean)).sum() / ((xl - xl_mean) ** 2).sum()
    a = xc_mean - b * xl_mean
    pred = a + b * xlag
    resid = x - pred
    resid[~mask] = np.nan
    return resid


def _rolling_acf1(vals: np.ndarray, window: int) -> np.ndarray:
    """Rolling lag-1 autocorrelation of a series."""
    n = len(vals)
    out = np.full(n, np.nan)
    min_obs = max(8, window // 2)
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        seg = vals[start: i + 1]
        mask = np.isfinite(seg)
        s = seg[mask]
        if len(s) < min_obs:
            continue
        if s.std() < 1e-12:
            continue
        # lag-1 ACF
        s_demean = s - s.mean()
        num = (s_demean[:-1] * s_demean[1:]).sum()
        den = (s_demean ** 2).sum()
        out[i] = float(num / den) if den > 1e-12 else np.nan
    return out


def _rolling_arch(resid: np.ndarray, window: int) -> np.ndarray:
    """
    ARCH effect: lag-1 ACF of squared residuals.
    High = volatility clustering (heteroscedastic residuals).
    """
    sq_resid = resid ** 2
    return _rolling_acf1(sq_resid, window)


def _rolling_sign_acf(resid: np.ndarray, window: int) -> np.ndarray:
    """
    Sign autocorrelation of residuals.
    Measures non-linear serial dependence (sign of residuals correlated).
    """
    signs = np.sign(resid).astype(float)
    signs[~np.isfinite(resid)] = np.nan
    return _rolling_acf1(signs, window)


def _rolling_lb_q(resid: np.ndarray, window: int, max_lag: int = 5) -> np.ndarray:
    """
    Rolling Ljung-Box Q statistic summing squared ACF over lags 1..max_lag.
    Q = n(n+2) * sum_{k=1}^{max_lag} rho_k^2 / (n-k)
    Normalised by max_lag for comparability across windows.
    """
    n = len(resid)
    out = np.full(n, np.nan)
    min_obs = max(max_lag + 4, window // 2)
    for i in range(min_obs, n):
        start = max(0, i - window + 1)
        seg = resid[start: i + 1]
        mask = np.isfinite(seg)
        s = seg[mask]
        nn = len(s)
        if nn < min_obs:
            continue
        s_demean = s - s.mean()
        var = (s_demean ** 2).sum()
        if var < 1e-12:
            continue
        q = 0.0
        for k in range(1, max_lag + 1):
            if k >= nn:
                break
            rho_k = (s_demean[:-k] * s_demean[k:]).sum() / var
            q += (nn * (nn + 2)) * (rho_k ** 2) / (nn - k)
        out[i] = float(q / max_lag)
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    log_ret = np.log(close / close.shift(1)).values.astype(np.float64)

    # Fit global AR(1) and extract residuals
    resid = _ar1_residuals(log_ret)

    # Residual lag-1 autocorrelation
    df["ver_resid_acf1_32d"] = _rolling_acf1(resid, 32)
    df["ver_resid_acf1_64d"] = _rolling_acf1(resid, 64)

    # ARCH effect (squared residual autocorrelation)
    df["ver_arch_effect_32d"] = _rolling_arch(resid, 32)

    # Sign autocorrelation
    df["ver_sign_acf_32d"] = _rolling_sign_acf(resid, 32)

    # Ljung-Box Q stat (lags 1-5)
    df["ver_lb_q5_32d"] = _rolling_lb_q(resid, 32, max_lag=5)
    df["ver_lb_q5_64d"] = _rolling_lb_q(resid, 64, max_lag=5)

    # Composite vigilance signal: mean of abs(ACF1) + ARCH + abs(signACF)
    # All normalised to comparable scale; high = strong residual structure
    acf1 = df["ver_resid_acf1_32d"].abs()
    arch = df["ver_arch_effect_32d"].abs()
    sign_acf = df["ver_sign_acf_32d"].abs()
    # Average the three structure signals
    composite = pd.concat([acf1, arch, sign_acf], axis=1)
    composite.columns = ["a", "b", "c"]
    df["ver_vigilance_signal"] = composite.mean(axis=1)

    return df
