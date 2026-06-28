"""
Higuchi Fractal Dimension (HFD) of the cumulative log-price curve.

Cross-domain method from signal processing / econophysics / HRV / DSP.
Per-ticker rolling estimate: Higuchi 1988 algorithm on the log-close series
with a 100-day window and kmax=8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_higuchi_fd",
    "description": (
        "Higuchi fractal dimension (HFD) of the log-close price curve "
        "estimated on a rolling 100-day window with kmax=8. "
        "For each window, the Higuchi algorithm builds k sub-sampled "
        "curve-length estimates L(k) for k=1..kmax, then regresses "
        "log L(k) on log(1/k); the slope is the fractal dimension in [1,2]. "
        "Higher FD => rougher / more complex price path; lower FD => smoother "
        "trending path. This is a per-ticker time-series estimate, not "
        "cross-sectional. "
        "Produces: xdom_higuchi_fd_100 (rolling HFD), "
        "xdom_higuchi_fd_zscore (rolling z-score vs its own 60-day history), "
        "xdom_higuchi_fd_chg (5-day change in HFD). "
        "Distinct from Hurst exponent (which measures long memory, not "
        "curve-length fractal dimension)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_higuchi_fd_100",
        "xdom_higuchi_fd_zscore",
        "xdom_higuchi_fd_chg",
    ],
    "tags": ["fractal", "complexity", "cross-domain", "nonlinear", "econophysics"],
    "version": "1.0.0",
    "author": "Higuchi (1988) / Cross-domain method transfer (signal processing / econophysics / HRV / DSP)",
}

# ---------------------------------------------------------------------------
# Core Higuchi algorithm (no-loop over individual rows -- applied via stride tricks)
# ---------------------------------------------------------------------------

def _higuchi_fd_window(x: np.ndarray, kmax: int = 8) -> float:
    """
    Compute the Higuchi fractal dimension of a 1-D signal x.

    Parameters
    ----------
    x    : 1-D numpy array (the signal, e.g. log-close series for one window)
    kmax : maximum interval scale (default 8)

    Returns
    -------
    Fractal dimension (slope of log L(k) vs log(1/k)), typically in [1, 2].
    Returns NaN if computation fails.
    """
    n = len(x)
    if n < kmax * 2 + 1:
        return np.nan

    log_k = np.empty(kmax)
    log_L = np.empty(kmax)

    for k in range(1, kmax + 1):
        # Number of sub-series starting at each m in {1..k}
        # L_m(k) = (1/(n-1)) * (n-1) / floor((n-m)/k) / k^2 * sum|x[m+i*k] - x[m+(i-1)*k]|
        lengths = []
        for m in range(1, k + 1):
            # Indices: m-1, m-1+k, m-1+2k, ...  (0-based)
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            vals = x[idxs]
            num_steps = len(vals) - 1          # floor((n - m) / k)
            if num_steps < 1:
                continue
            curve_length = np.sum(np.abs(np.diff(vals)))
            # Normalise: multiply by (n-1) / (num_steps * k)
            norm = (n - 1) / (num_steps * k)
            L_m = curve_length * norm / k
            lengths.append(L_m)

        if len(lengths) == 0:
            return np.nan
        L_k = np.mean(lengths)
        if L_k <= 0:
            return np.nan
        log_k[k - 1] = np.log(1.0 / k)   # log(1/k)
        log_L[k - 1] = np.log(L_k)

    # Linear regression: log_L = FD * log_k + const
    # slope = FD
    x_arr = log_k
    y_arr = log_L
    n_pts = kmax
    mx = x_arr.mean()
    my = y_arr.mean()
    denom = np.sum((x_arr - mx) ** 2)
    if denom == 0:
        return np.nan
    slope = np.sum((x_arr - mx) * (y_arr - my)) / denom
    return float(slope)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    WIN = 100      # rolling window length
    KMAX = 8       # Higuchi kmax
    ZSCORE_WIN = 60
    CHG_LAG = 5

    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # log-close series (log of price level)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.where(close > 0, np.log(close), np.nan)

    fd_vals = np.full(n, np.nan)

    # Compute rolling HFD using a Python loop over windows.
    # With WIN=100 and typical ~700 rows, this is ~600 iterations.
    # Each call does a tiny O(kmax * WIN/kmax) loop -- well within 100 ms.
    for i in range(WIN - 1, n):
        window = log_close[i - WIN + 1 : i + 1]
        if np.any(np.isnan(window)):
            continue
        fd_vals[i] = _higuchi_fd_window(window, kmax=KMAX)

    # Replace inf / -inf with NaN
    fd_vals = np.where(np.isfinite(fd_vals), fd_vals, np.nan)

    df["xdom_higuchi_fd_100"] = fd_vals

    # Rolling z-score of HFD vs its own recent history
    fd_series = pd.Series(fd_vals, index=df.index)
    roll_mean = fd_series.rolling(ZSCORE_WIN, min_periods=ZSCORE_WIN // 2).mean()
    roll_std  = fd_series.rolling(ZSCORE_WIN, min_periods=ZSCORE_WIN // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (fd_series - roll_mean) / roll_std.replace(0, np.nan)
    # Guard against any residual inf
    df["xdom_higuchi_fd_zscore"] = zscore.replace([np.inf, -np.inf], np.nan)

    # 5-day change in HFD (captures acceleration/deceleration of roughness)
    df["xdom_higuchi_fd_chg"] = fd_series.diff(CHG_LAG)

    return df
