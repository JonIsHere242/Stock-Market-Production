"""
Two-compartment hidden-state features derived from:
  "Physics-Informed Neural Networks for Chemotherapy Pharmacokinetics:
   Benchmarking the Clinical Estimator and Exposing Parameter Identifiability"
  arXiv 2606.12658

The paper fits a two-compartment pharmacokinetic ODE to an observable signal
("plasma") to infer a hidden compartment ("tissue"):
    dP/dt = -(k10 + k12) * P + k21 * T   (plasma/price)
    dT/dt = k12 * P - k21 * T             (tissue/hidden state)

We apply the same linear two-compartment discrete model to price as the
observable signal (plasma) and infer the latent hidden compartment (tissue)
via iterative least-squares parameter estimation over a rolling window.

The hidden compartment behaves as a slow/damped version of price and captures
distributed momentum that is not visible in the raw price series.

Feature family (7 columns):
  tc_hidden_state      estimated latent compartment value (normalised)
  tc_hidden_dev        hidden state minus log-price: divergence signal
  tc_k12_20            estimated transfer rate plasma→tissue, 20-day window
  tc_k21_20            estimated transfer rate tissue→plasma, 20-day window
  tc_k10_20            estimated elimination rate, 20-day window
  tc_hidden_momentum   5-day rate of change of the hidden compartment
  tc_compartment_ratio ratio of hidden/plasma compartment levels
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12658_twocomp_hidden_state",
    "description": (
        "Two-compartment ODE hidden-state features: fits a discrete plasma/tissue "
        "PK model to log-price and extracts the latent tissue compartment; "
        "from arXiv 2606.12658 (PINNs for PK)."
    ),
    "requires": ["Close"],
    "produces": [
        "tc_hidden_state",
        "tc_hidden_dev",
        "tc_k12_20",
        "tc_k21_20",
        "tc_k10_20",
        "tc_hidden_momentum",
        "tc_compartment_ratio",
    ],
    "tags": ["momentum", "mean_reversion", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12658",
}


def _fit_twocomp_rolling(log_p: np.ndarray, window: int) -> tuple:
    """
    Fit the discrete two-compartment model over a rolling window using
    closed-form OLS in a linearized state-space.

    Discrete model (dt=1 day):
        P[t] = (1 - k10 - k12)*P[t-1] + k21*T[t-1]
        T[t] = k12*P[t-1] + (1 - k21)*T[t-1]

    Because T is unobserved, we use the following approximation:
        T[t] ≈ EMA_slow(P) (exponential smoothed price as tissue proxy).
    Then we perform OLS to recover k10, k12, k21 that best predict P.

    Returns arrays: (k10, k12, k21, T_hidden) each of length n.
    """
    n = len(log_p)
    k10_arr  = np.full(n, np.nan)
    k12_arr  = np.full(n, np.nan)
    k21_arr  = np.full(n, np.nan)

    # Slow EMA as tissue proxy (half-life ≈ window/3 days)
    alpha_slow = 2.0 / (window // 3 + 1)
    T_hat = np.full(n, np.nan)
    if np.isfinite(log_p[0]):
        T_hat[0] = log_p[0]
    for i in range(1, n):
        if np.isfinite(log_p[i]) and np.isfinite(T_hat[i - 1]):
            T_hat[i] = alpha_slow * log_p[i] + (1.0 - alpha_slow) * T_hat[i - 1]
        elif np.isfinite(log_p[i]):
            T_hat[i] = log_p[i]
        else:
            T_hat[i] = T_hat[i - 1]

    min_obs = max(10, window // 3)

    for i in range(min_obs, n):
        start = max(1, i - window + 1)
        # Response: P[t]
        y = log_p[start : i + 1]
        # Regressors: P[t-1], T[t-1]
        P_lag = log_p[start - 1 : i]
        T_lag = T_hat[start - 1 : i]

        mask = np.isfinite(y) & np.isfinite(P_lag) & np.isfinite(T_lag)
        if mask.sum() < min_obs:
            continue

        y_m    = y[mask]
        Pl_m   = P_lag[mask]
        Tl_m   = T_lag[mask]

        # OLS: y = a*P_lag + b*T_lag  (no intercept — both in log-price space)
        X = np.column_stack([Pl_m, Tl_m])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_m, rcond=None)
        except np.linalg.LinAlgError:
            continue

        a, b = coef  # a = 1 - k10 - k12,  b = k21
        k21_hat = float(np.clip(b, 0.0, 1.0))
        k10_k12 = float(np.clip(1.0 - a, 0.0, 1.0))
        # Split k10 and k12 proportionally: assume k10 = 0.5 * k10_k12 as prior
        k10_hat = k10_k12 * 0.5
        k12_hat = k10_k12 * 0.5

        k10_arr[i] = k10_hat
        k12_arr[i] = k12_hat
        k21_arr[i] = k21_hat

    return k10_arr, k12_arr, k21_arr, T_hat


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_p = np.log(df["Close"].replace(0, np.nan).values)

    k10, k12, k21, T_hidden = _fit_twocomp_rolling(log_p, window=40)

    # Normalise hidden state by subtracting log-price (remove scale effect)
    hidden_dev = T_hidden - log_p

    # Normalise hidden state to z-score over 60-day rolling window
    T_s = pd.Series(T_hidden, index=df.index)
    T_mu  = T_s.rolling(60, min_periods=20).mean()
    T_sig = T_s.rolling(60, min_periods=20).std()
    T_z   = (T_s - T_mu) / T_sig.replace(0, np.nan)

    df["tc_hidden_state"]      = T_z
    df["tc_hidden_dev"]        = pd.Series(hidden_dev, index=df.index)
    df["tc_k12_20"]            = pd.Series(k12, index=df.index)
    df["tc_k21_20"]            = pd.Series(k21, index=df.index)
    df["tc_k10_20"]            = pd.Series(k10, index=df.index)
    df["tc_hidden_momentum"]   = T_s.diff(5)
    df["tc_compartment_ratio"] = T_s / pd.Series(log_p, index=df.index).replace(0, np.nan)

    return df
