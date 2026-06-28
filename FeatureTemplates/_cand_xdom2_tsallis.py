"""
Tsallis entropy q=2 of returns (per-ticker proxy).

Per-ticker implementation of the cross-domain method from Tsallis (1988).
Computes rolling 80-day Tsallis entropy with q=2 over a 10-bin rolling-quantile
histogram of daily log-returns. Because the histogram is built from only the
current stock's return history (not cross-sectionally), this is a faithful
per-ticker proxy of the intended signal.

S_q = (1 - sum(p_i^q)) / (q - 1)   for q=2
    = (1 - sum(p_i^2)) / 1
    = 1 - sum(p_i^2)

This nonextensive entropy is more sensitive to fat tails than Shannon entropy;
paired with Renyi entropy it provides an orthogonal tail-shape probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_tsallis",
    "description": (
        "Rolling 80-day Tsallis entropy (q=2) of log-returns using a 10-bin "
        "rolling-quantile histogram per ticker. S_q = 1 - sum(p_i^2). "
        "Nonextensive entropy more sensitive to fat tails than Shannon; "
        "orthogonal tail-shape probe alongside Renyi. Per-ticker proxy "
        "(not cross-sectional); slope variant captures entropy trend."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_tsallis_80",        # level: rolling 80-day Tsallis q=2 entropy
        "xdom2_tsallis_80_slope",  # slope: 10-day slope of the entropy series
        "xdom2_tsallis_80_z",      # z-score: 60-day normalised entropy level
    ],
    "tags": ["entropy", "tail-risk", "nonextensive", "returns", "cross-domain"],
    "version": "1.0",
    "author": "Tsallis entropy q=2 of returns (Tsallis 1988); cross-domain batch 2",
}

# ── constants ──────────────────────────────────────────────────────────────────
_WINDOW = 80      # rolling window for entropy
_N_BINS = 10      # number of histogram bins (quantile-based)
_SLOPE_WIN = 10   # window for linear slope of entropy
_Z_WIN = 60       # window for z-normalising the entropy


def _tsallis2_from_window(returns_window: np.ndarray) -> float:
    """
    Compute Tsallis q=2 entropy for a 1-D array of returns.
    Uses a 10-bin quantile histogram so bins adapt to the distribution shape.
    Returns NaN if there are fewer than _N_BINS valid observations.
    """
    valid = returns_window[np.isfinite(returns_window)]
    if len(valid) < _N_BINS:
        return np.nan

    # Build quantile bin edges (adaptive to the empirical distribution)
    quantiles = np.linspace(0.0, 1.0, _N_BINS + 1)
    edges = np.quantile(valid, quantiles)

    # Ensure strictly increasing edges (duplicate edges collapse bins)
    # np.digitize handles this naturally; count falls into each bin
    counts, _ = np.histogram(valid, bins=edges)

    total = counts.sum()
    if total == 0:
        return np.nan

    p = counts / total           # probability mass per bin
    # Tsallis q=2: S_q = (1 - sum(p_i^2)) / (q-1) = 1 - sum(p_i^2)
    entropy = 1.0 - float(np.sum(p ** 2))
    return entropy


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add xdom2_tsallis_80, xdom2_tsallis_80_slope, xdom2_tsallis_80_z to df.

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock OHLCV frame, ascending by Date.

    Returns
    -------
    pd.DataFrame
        Original df with three new columns appended.
    """
    # Daily log-returns; NaN for first row
    close = df["Close"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.empty(len(close), dtype=np.float64)
        log_rets[0] = np.nan
        mask = (close[:-1] > 0) & (close[1:] > 0)
        log_rets[1:] = np.where(mask, np.log(close[1:] / close[:-1]), np.nan)

    n = len(log_rets)

    # ── rolling Tsallis entropy ────────────────────────────────────────────────
    tsallis = np.full(n, np.nan, dtype=np.float64)
    for i in range(_WINDOW - 1, n):
        window = log_rets[i - _WINDOW + 1 : i + 1]
        tsallis[i] = _tsallis2_from_window(window)

    df["xdom2_tsallis_80"] = tsallis

    # ── slope of entropy (linear regression over _SLOPE_WIN days) ─────────────
    slope_arr = np.full(n, np.nan, dtype=np.float64)
    if n >= _SLOPE_WIN:
        x = np.arange(_SLOPE_WIN, dtype=np.float64)
        x_mean = x.mean()
        x_dev = x - x_mean
        ss_x = float(np.dot(x_dev, x_dev))
        if ss_x > 0:
            for i in range(_SLOPE_WIN - 1, n):
                y = tsallis[i - _SLOPE_WIN + 1 : i + 1]
                if np.any(np.isnan(y)):
                    continue
                y_dev = y - y.mean()
                slope_arr[i] = float(np.dot(x_dev, y_dev)) / ss_x

    df["xdom2_tsallis_80_slope"] = slope_arr

    # ── z-score of entropy (rolling mean/std over _Z_WIN days) ────────────────
    tsallis_s = pd.Series(tsallis)
    roll_mean = tsallis_s.rolling(_Z_WIN, min_periods=_Z_WIN // 2).mean()
    roll_std = tsallis_s.rolling(_Z_WIN, min_periods=_Z_WIN // 2).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z_vals = np.where(
            roll_std > 0,
            (tsallis_s - roll_mean) / roll_std,
            np.nan,
        )
    df["xdom2_tsallis_80_z"] = z_vals

    return df
