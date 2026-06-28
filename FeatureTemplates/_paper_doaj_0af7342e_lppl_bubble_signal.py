"""
Log-Periodic Power Law (LPPL) bubble/crash acceleration proxy per ticker, derived from:
  "Crash Diagnosis and Price Rebound Prediction in NYSE Composite Index Based on
   Visibility Graph and Time-Evolving Stock Correlation Network" (doaj:0af7342ef08d444f82ff02bfdb70f08c).

The paper extends the LPPL model (Johansen-Ledoit-Sornette) to detect market crashes.
LPPL characterises bubbles as a price trajectory that accelerates (power-law) with
superimposed log-periodic oscillations toward a critical time.  We approximate the
core LPPL feature PER TICKER using two cheap rolling statistics:
  1. lppl_accel_N: log-price acceleration = second derivative of log-price over window N,
     positive = super-exponential growth (bubble), negative = crash mode.
  2. lppl_lnp_dev_N: how far log-price is above/below a rolling linear trend
     (residual from OLS on log-price), capturing the "super-linear" component.
  3. lppl_osc_N: zero-crossing rate of the log-price residual within window N,
     a causal proxy for the log-periodic oscillation frequency.
All rolling, causal, no lookahead.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_paper_doaj_0af7342e_lppl_bubble_signal",
    "description": "Log-periodic power-law bubble/crash proxies: log-price acceleration, detrended residual, and oscillation rate.",
    "requires":    ["Close"],
    "produces":    [
        "lppl_accel_40",
        "lppl_lnp_dev_40",
        "lppl_osc_40",
        "lppl_accel_80",
        "lppl_lnp_dev_80",
        "lppl_osc_80",
    ],
    "tags":        ["experimental", "bubble", "regime", "trend"],
    "version":     "1.0",
    "author":      "paper-mining slate 2",
}


def _rolling_linreg_residual(log_p: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling causal OLS residual of log_p ~ t within each window.
    Returns array of same length; first (window-1) rows are NaN.
    """
    n = len(log_p)
    resid = np.full(n, np.nan)
    t = np.arange(window, dtype=float)
    # Precompute OLS matrices for a fixed-length window with x = [0..window-1]
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    for i in range(window - 1, n):
        y = log_p[i - window + 1 : i + 1]
        y_mean = y.mean()
        # slope via cov(t,y)/var(t)
        slope = ((t - t_mean) * (y - y_mean)).sum() / t_var
        intercept = y_mean - slope * t_mean
        fitted_last = intercept + slope * (window - 1)
        resid[i] = y[-1] - fitted_last
    return resid


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_p = np.log(df["Close"].values.astype(float))
    n = len(log_p)

    for window in (40, 80):
        # --- Feature 1: log-price acceleration (2nd finite difference of rolling EWM) ---
        # Use EWM to smooth then take 2nd diff as acceleration proxy
        log_s = pd.Series(log_p).ewm(span=window // 4, min_periods=1, adjust=False).mean().values
        d1 = np.diff(log_s, prepend=np.nan)
        d2 = np.diff(d1, prepend=np.nan)
        # Normalise by rolling std of d1 to make it scale-free
        d1_s = pd.Series(d1)
        d1_std = d1_s.rolling(window, min_periods=window // 2).std().values
        accel = np.where(d1_std > 0, d2 / d1_std, np.nan)
        df[f"lppl_accel_{window}"] = accel

        # --- Feature 2: rolling linear-trend residual of log-price ---
        resid = _rolling_linreg_residual(log_p, window)
        # Normalise by rolling log-price std
        lp_std = pd.Series(log_p).rolling(window, min_periods=window // 2).std().values
        dev = np.where(lp_std > 0, resid / lp_std, np.nan)
        df[f"lppl_lnp_dev_{window}"] = dev

        # --- Feature 3: oscillation rate (zero-crossing rate of residual) ---
        resid_arr = np.where(np.isnan(resid), 0.0, resid)
        signs = np.sign(resid_arr)
        sign_changes = np.abs(np.diff(signs, prepend=signs[0]))  # 2 where sign flips
        osc_rate = (
            pd.Series(sign_changes).rolling(window, min_periods=window // 2).sum().values
            / window
        )
        # Mask early NaN rows (where resid is NaN)
        osc_rate[: window - 1] = np.nan
        df[f"lppl_osc_{window}"] = osc_rate

    return df
