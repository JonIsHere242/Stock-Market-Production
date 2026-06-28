"""
Allan-deviation noise-color slope across tau.

Extends the xdom_allan_variance family by extracting the LOG-LOG SLOPE of
Allan deviation across tau = {2, 4, 8}, computed on an 80-day rolling window
of log-returns.  The slope mu (where sigma_A(tau) ~ tau^(mu/2)) classifies
noise color:  mu ~ -1 = white noise, mu ~ 0 = flicker / 1/f, mu ~ +1 =
random-walk.  This is orthogonal to the level (magnitude) that the parent
block captures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

METADATA = {
    "name": "ext_allan_tau_slope",
    "description": (
        "Rolling 80-day Allan-deviation log-log slope across tau={2,4,8} on "
        "log-returns.  The slope classifies the noise color (white/flicker/"
        "random-walk) and is orthogonal to the variance level captured by the "
        "parent block xdom_allan_variance.  Produces: slope mu, tau8/tau2 "
        "deviation ratio, and a 20-day z-score of the slope.  Per-ticker "
        "OHLCV proxy; no cross-sectional data needed."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_allan_tau_slope_mu",       # log-log slope of Allan dev vs tau
        "ext_allan_tau_slope_ratio",    # Allan_dev(tau=8) / Allan_dev(tau=2)
        "ext_allan_tau_slope_z20",      # 20-bar z-score of the slope
    ],
    "tags": ["noise", "volatility", "allan_variance", "time_series", "spectral"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner xdom_allan_variance",
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _allan_dev(arr: np.ndarray, tau: int) -> float:
    """
    Allan deviation for integer averaging factor tau.
    Groups the input array into non-overlapping blocks of size tau, computes
    block means, then returns sqrt(0.5 * mean of squared successive differences
    of block means).  Returns NaN if there are fewer than 2 complete blocks.
    """
    n = len(arr)
    n_blocks = n // tau
    if n_blocks < 2:
        return np.nan
    # trim to complete blocks
    trimmed = arr[: n_blocks * tau]
    # block means
    block_means = trimmed.reshape(n_blocks, tau).mean(axis=1)
    # successive differences
    diffs = np.diff(block_means)
    return float(np.sqrt(0.5 * np.mean(diffs ** 2)))


def _slope_from_window(log_ret_window: np.ndarray) -> tuple[float, float]:
    """
    Given a 1-D array of log-returns (the rolling window), compute:
      - slope mu of log sigma_A vs log tau (taus = 2, 4, 8)
      - ratio sigma_A(8) / sigma_A(2)
    Returns (nan, nan) if any Allan dev is nan or zero.
    """
    taus = np.array([2, 4, 8], dtype=float)
    devs = np.array([_allan_dev(log_ret_window, int(t)) for t in taus])

    if np.any(np.isnan(devs)) or np.any(devs <= 0):
        return np.nan, np.nan

    # log-log fit: log(dev) = (mu/2) * log(tau) + const
    # => slope of log(dev) vs log(tau)  = mu/2  => mu = 2 * slope
    log_tau = np.log(taus)
    log_dev = np.log(devs)

    # simple OLS slope (3 points)
    log_tau_m = log_tau - log_tau.mean()
    log_dev_m = log_dev - log_dev.mean()
    slope_half = float(np.dot(log_tau_m, log_dev_m) / np.dot(log_tau_m, log_tau_m))
    mu = 2.0 * slope_half

    ratio = float(devs[2] / devs[0])  # tau=8 / tau=2
    return mu, ratio


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

WINDOW = 80       # main rolling window
Z_WINDOW = 20     # z-score lookback for the slope


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # log-returns; first row is NaN
    close = df["Close"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ret = np.empty(n, dtype=float)
        log_ret[0] = np.nan
        denom = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret[1:] = np.where(denom > 0, np.log(close[1:] / denom), np.nan)

    mu_arr = np.full(n, np.nan, dtype=float)
    ratio_arr = np.full(n, np.nan, dtype=float)

    # minimum rows needed: window of 80 log-returns requires 81 close prices,
    # but the log-ret array itself has n elements (first is NaN).
    # We need at least WINDOW valid log-returns in the window.
    # Require tau=8 to have >=2 blocks => need >= 16 in window, but we use 80.

    for i in range(WINDOW, n):
        # window of the LAST `WINDOW` log-returns ending at i (inclusive)
        w = log_ret[i - WINDOW + 1: i + 1]   # length WINDOW
        if np.isnan(w).any():
            continue
        mu, ratio = _slope_from_window(w)
        mu_arr[i] = mu
        ratio_arr[i] = ratio

    # 20-bar z-score of mu
    mu_series = pd.Series(mu_arr)
    roll_mean = mu_series.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).mean()
    roll_std  = mu_series.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z20_arr = np.where(
            roll_std.to_numpy() > 0,
            (mu_arr - roll_mean.to_numpy()) / roll_std.to_numpy(),
            np.nan,
        )

    df["ext_allan_tau_slope_mu"]    = mu_arr
    df["ext_allan_tau_slope_ratio"] = ratio_arr
    df["ext_allan_tau_slope_z20"]   = z20_arr

    return df
