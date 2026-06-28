"""
_paper_2605_16324_chaotic_interval_width.py
============================================
Per-ticker OHLCV proxy for prediction-interval / uncertainty features inspired
by arXiv 2605.16324 "Bi-Level Chaotic Fusion Based GCN for Stock Market
Prediction Interval", which produces separate CENTER / WIDTH models with a
volatility-aware gating mechanism and a LUBE objective.

No neural network, no graph — this is a pure causal OHLCV approximation of
the same structural ideas:

  - CENTER proxy   : EWMA-trend of log-returns (causal, short-horizon)
  - WIDTH proxy    : rolling realized vol / ATR-style range
  - GATED width    : width × vol-regime percentile rank (calm vs turbulent)
  - ASYMMETRY      : separate upside vs downside range components
  - PRICE POSITION : where close sits within its own expected band
  - SHARPNESS      : current width relative to its own history
  - CHAOTIC proxy  : rolling permutation entropy of recent returns
    (high entropy → less predictable → wider interval implied)

All features are causal: value at row t depends only on rows ≤ t.
Leading NaNs during the warmup window are expected and left as-is.
No inf values are emitted.
"""

import math
import warnings
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2605_16324_chaotic_interval_width",
    "description": (
        "Per-ticker OHLCV prediction-interval/uncertainty proxy: causal interval "
        "center (EWMA trend), width (realized vol), volatility-gated width, "
        "upside/downside asymmetry, price-in-band position, sharpness, and "
        "permutation-entropy 'chaoticity' — inspired by arXiv 2605.16324."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "civ_center",           # interval-center proxy: EWMA log-return trend
        "civ_width",            # interval-width proxy: rolling realized vol (ann.)
        "civ_gated_width",      # vol-regime-gated width (wider in turbulent regimes)
        "civ_up_width",         # upside asymmetry: EWMA of positive daily ranges
        "civ_down_width",       # downside asymmetry: EWMA of negative daily ranges
        "civ_asymmetry",        # up_width / (up_width + down_width) — skew [0,1]
        "civ_position",         # price-in-band: z-score of close vs rolling band
        "civ_sharpness",        # width / ewma(width) — relative sharpness
        "civ_entropy",          # rolling permutation entropy (chaoticity proxy)
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper proxy — arXiv 2605.16324 (no neural net, OHLCV only)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perm_entropy(x: np.ndarray, order: int = 3) -> float:
    """
    Sample permutation entropy of a 1-D array of length >= order.
    Returns a value in [0, 1] (normalised by log2(order!)).
    Returns NaN when the array is too short or all-equal.
    """
    n = len(x)
    if n < order or np.all(x == x[0]):
        return np.nan

    # Count ordinal patterns
    counts: dict = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i : i + order]))
        counts[pattern] = counts.get(pattern, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return np.nan

    probs = np.array(list(counts.values()), dtype=float) / total
    # Shannon entropy
    h = -np.sum(probs * np.log2(probs + 1e-12))
    # Normalise: max entropy = log2(order!)
    max_h = math.log2(math.factorial(order))
    if max_h == 0:
        return np.nan
    return h / max_h


def _rolling_perm_entropy(returns: np.ndarray, window: int, order: int = 3) -> np.ndarray:
    """
    Compute rolling permutation entropy using a Python loop.
    Rows [0 .. window-2] are NaN (insufficient history).
    """
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = returns[i - window + 1 : i + 1]
        # skip if all NaN in segment
        valid = segment[~np.isnan(segment)]
        if len(valid) < order:
            continue
        out[i] = _perm_entropy(valid, order=order)
    return out


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 9 causal interval-width / uncertainty features prefixed civ_.

    All computations are causal: row i only uses rows <= i.
    Leading NaN rows during warmup are intentional.
    """
    close = df["Close"].to_numpy(dtype=np.float64)
    high  = df["High"].to_numpy(dtype=np.float64)
    low   = df["Low"].to_numpy(dtype=np.float64)

    n = len(close)

    # -----------------------------------------------------------------------
    # Log-returns (causal: diff with past)
    # -----------------------------------------------------------------------
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ret[1:] = np.log(close[1:] / close[:-1])
    log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

    # -----------------------------------------------------------------------
    # 1. CENTER proxy — EWMA of log-returns (short span, causal)
    #    Represents the "forecast center" of the next interval.
    # -----------------------------------------------------------------------
    center_span = 10
    civ_center = (
        pd.Series(log_ret)
        .ewm(span=center_span, min_periods=center_span, adjust=False)
        .mean()
        .to_numpy()
    )

    # -----------------------------------------------------------------------
    # 2. WIDTH proxy — rolling realized vol (annualised), window=21 days
    #    Represents interval half-width driven by recent price turbulence.
    # -----------------------------------------------------------------------
    vol_window = 21
    ann_factor = math.sqrt(252)
    civ_width = (
        pd.Series(log_ret)
        .rolling(vol_window, min_periods=vol_window)
        .std()
        .multiply(ann_factor)
        .to_numpy()
    )

    # -----------------------------------------------------------------------
    # 3. GATED WIDTH — scale width by the vol-percentile rank over 63 days
    #    Calm regime  (pct ~ 0) → gated_width ≈ 0
    #    Turbulent    (pct ~ 1) → gated_width ≈ width
    #    This mirrors the paper's volatility-aware gating mechanism.
    # -----------------------------------------------------------------------
    pct_window = 63
    width_series = pd.Series(civ_width)
    # percentile rank uses only PAST values (shift(1) then rolling)
    vol_pct_rank = (
        width_series.shift(1)
        .rolling(pct_window, min_periods=pct_window)
        .apply(lambda x: (x[:-1] <= x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
    )
    # Re-attach current width (not shifted) but gate by past-based rank
    civ_gated_width = (width_series * vol_pct_rank).to_numpy()

    # -----------------------------------------------------------------------
    # 4. ASYMMETRY — upside vs downside range components
    #    Up-width  : EWMA of (High - Close_prev) when positive (upside range)
    #    Down-width: EWMA of (Close_prev - Low)  when positive (downside range)
    #    Both are causal: prev close is already available.
    # -----------------------------------------------------------------------
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    up_range   = np.where(np.isfinite(prev_close), np.maximum(high - prev_close, 0.0), np.nan)
    down_range = np.where(np.isfinite(prev_close), np.maximum(prev_close - low,  0.0), np.nan)

    ewm_span = 21
    civ_up_width = (
        pd.Series(up_range)
        .ewm(span=ewm_span, min_periods=ewm_span, adjust=False)
        .mean()
        .to_numpy()
    )
    civ_down_width = (
        pd.Series(down_range)
        .ewm(span=ewm_span, min_periods=ewm_span, adjust=False)
        .mean()
        .to_numpy()
    )

    # Asymmetry ratio: fraction attributable to upside, in [0, 1]
    total_width = civ_up_width + civ_down_width
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        civ_asymmetry = np.where(total_width > 0, civ_up_width / total_width, np.nan)

    # -----------------------------------------------------------------------
    # 5. PRICE POSITION — where close sits within its expected band
    #    Band = rolling mean ± rolling std of close (causal window=21)
    #    z-score > 0 means price is above its expected center.
    # -----------------------------------------------------------------------
    band_window = 21
    close_series = pd.Series(close)
    roll_mean = close_series.rolling(band_window, min_periods=band_window).mean()
    roll_std  = close_series.rolling(band_window, min_periods=band_window).std()

    mean_arr = roll_mean.to_numpy()
    std_arr  = roll_std.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        civ_position = np.where(
            (std_arr > 0) & np.isfinite(mean_arr),
            (close - mean_arr) / std_arr,
            np.nan,
        )

    # -----------------------------------------------------------------------
    # 6. SHARPNESS — current width relative to its recent EWMA
    #    > 1 means width is expanding (less sharp / more uncertain)
    #    < 1 means width is contracting (sharper / more certain)
    # -----------------------------------------------------------------------
    sharpness_span = 42
    width_ewma = (
        width_series.ewm(span=sharpness_span, min_periods=sharpness_span, adjust=False)
        .mean()
        .to_numpy()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        civ_sharpness = np.where(
            (width_ewma > 0) & np.isfinite(civ_width),
            civ_width / width_ewma,
            np.nan,
        )

    # -----------------------------------------------------------------------
    # 7. CHAOTIC PROXY — rolling permutation entropy of log-returns
    #    Higher entropy → more complex / less predictable dynamics →
    #    the paper's "chaotic fusion" would imply a wider interval.
    #    Window=20, order=3 (6 ordinal patterns, fast loop).
    # -----------------------------------------------------------------------
    ent_window = 20
    ent_order  = 3
    civ_entropy = _rolling_perm_entropy(log_ret, window=ent_window, order=ent_order)

    # -----------------------------------------------------------------------
    # Assemble and append — never modify existing columns
    # -----------------------------------------------------------------------
    new_cols = {
        "civ_center":      civ_center,
        "civ_width":       civ_width,
        "civ_gated_width": civ_gated_width,
        "civ_up_width":    civ_up_width,
        "civ_down_width":  civ_down_width,
        "civ_asymmetry":   civ_asymmetry,
        "civ_position":    civ_position,
        "civ_sharpness":   civ_sharpness,
        "civ_entropy":     civ_entropy,
    }

    # Replace any inf values (safety net — should not arise under normal data)
    for col_name, arr in new_cols.items():
        arr_safe = np.where(np.isinf(arr), np.nan, arr)
        df[col_name] = arr_safe

    return df
