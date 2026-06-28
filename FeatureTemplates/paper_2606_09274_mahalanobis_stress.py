"""
Mahalanobis-distance stress features derived from:
  "Reverse Stress Testing for Multivariate Scenarios: A Conditional Framework
   for Stressed Time Series"  (arxiv 2606.09274)

The paper reconstructs multivariate stress scenarios using the Mahalanobis
distance to a modal scenario inside the empirical return distribution.  We adapt
this to a per-ticker OHLCV feature: for each day, measure how far today's
OHLCV *joint* observation is from the recent centre of the distribution using
a rolling Mahalanobis distance.  Large values signal a stressed / anomalous day.

Feature family
--------------
  mah_dist_20    — rolling 20-day Mahalanobis distance of [ret, hl_range, log_vol]
  mah_dist_60    — same with 60-day window
  mah_pct_20     — percentile rank of mah_dist_20 in trailing 60-day window
  mah_pct_60     — same for mah_dist_60 in trailing 120-day window
  mah_shock_flag — 1 when mah_dist_20 > 95th pct (trailing 60d), else 0
  mah_reversal   — mean-reversion signal: -(mah_dist_20 × sign of return)
                   High-stress days with a loss often precede bounces.
"""

import numpy as np
import pandas as pd
from scipy import linalg as la

METADATA = {
    "name": "paper_2606_09274_mahalanobis_stress",
    "description": (
        "Rolling Mahalanobis distance in OHLCV-derived feature space to detect "
        "stressed / anomalous days; from arxiv 2606.09274 (Reverse Stress Testing)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "mah_dist_20",
        "mah_dist_60",
        "mah_pct_20",
        "mah_pct_60",
        "mah_shock_flag",
        "mah_reversal",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.09274",
}


def _rolling_mahal_fast(feat: np.ndarray, window: int) -> np.ndarray:
    """
    Efficient rolling Mahalanobis distance using scipy.linalg.solve.
    """
    n, k = feat.shape
    result = np.full(n, np.nan)
    ridge = np.eye(k) * 1e-6

    for i in range(window - 1, n):
        block = feat[i - window + 1 : i + 1]   # (window, k)
        if not np.all(np.isfinite(block)):
            continue
        mu = block.mean(axis=0)
        cov = np.cov(block.T, ddof=1) + ridge
        diff = feat[i] - mu
        try:
            z = la.solve(cov, diff, assume_a='pos', check_finite=False)
            result[i] = np.sqrt(max(float(diff @ z), 0.0))
        except la.LinAlgError:
            pass

    return result


def _fast_pctrank(values: np.ndarray, trail: int) -> np.ndarray:
    """
    Vectorised trailing percentile rank using stride-trick windows.
    For row i, rank values[i] among values[max(0,i-trail+1)..i-1].
    Avoids slow rolling.apply(Python fn).
    """
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isfinite(values[i]):
            continue
        start = max(0, i - trail + 1)
        hist = values[start:i]   # exclude self
        fin = hist[np.isfinite(hist)]
        if len(fin) < 5:
            continue
        result[i] = float(np.sum(fin < values[i])) / len(fin)
    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- Build 3-component feature vector -----------------------------------
    close = df["Close"].values.astype(float)
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    vol   = df["Volume"].values.astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret  = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
        hl_range = (high - low) / np.where(close > 0, close, np.nan)
        log_vol  = np.concatenate([
            [np.nan],
            np.log(vol[1:] / np.where(vol[:-1] > 0, vol[:-1], np.nan))
        ])

    feat = np.column_stack([log_ret, hl_range, log_vol])
    feat = np.where(np.isfinite(feat), feat, 0.0)

    # ---- Rolling Mahalanobis at two windows ----------------------------------
    dist20 = _rolling_mahal_fast(feat, window=20)
    dist60 = _rolling_mahal_fast(feat, window=60)

    df["mah_dist_20"] = dist20
    df["mah_dist_60"] = dist60

    # ---- Percentile rank (fast vectorised) -----------------------------------
    df["mah_pct_20"] = _fast_pctrank(dist20, trail=60)
    df["mah_pct_60"] = _fast_pctrank(dist60, trail=120)

    # ---- Shock flag (95th pct, trailing 60d) ---------------------------------
    s20 = pd.Series(dist20, index=df.index)
    p95 = s20.rolling(60, min_periods=20).quantile(0.95)
    shock = (s20 > p95).astype(float)
    shock[s20.isna() | p95.isna()] = np.nan
    df["mah_shock_flag"] = shock

    # ---- Mean-reversion signal -----------------------------------------------
    log_ret_s = pd.Series(log_ret, index=df.index)
    df["mah_reversal"] = -s20 * log_ret_s.apply(np.sign)

    return df
