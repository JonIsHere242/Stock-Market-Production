"""
Equilibrium State Estimation (ESE) deviation features derived from:
  "Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation"
  arXiv 2606.13285

The paper forecasts multiple interacting systems by first estimating a shared
equilibrium state, then measuring each system's deviation from that equilibrium.
We adapt this idea to per-ticker OHLCV features:

  1.  Treat each price-derived signal (log-return, HL-range, volume-ratio) as a
      "system".  Their rolling joint mean is the inferred equilibrium vector.
  2.  Measure how far each signal is from equilibrium (z-score).
  3.  Construct a scalar equilibrium-deviation index (Euclidean distance in
      standardised feature space) analogous to ESE's deviation-before-forecast.
  4.  Track how quickly the deviation is being resolved (convergence speed).

Feature family (8 columns):
  ese_ret_dev_20     log-return deviation from 20-day rolling equilibrium (z-score)
  ese_ret_dev_60     same, 60-day window
  ese_vol_dev_20     volume ratio deviation from 20-day equilibrium (z-score)
  ese_vol_dev_60     same, 60-day window
  ese_range_dev_20   HL-range deviation from 20-day equilibrium (z-score)
  ese_dist_20        Euclidean distance in joint standardised space, 20-day window
  ese_dist_60        same, 60-day window
  ese_convergence    rolling rate at which ese_dist_20 is decreasing (mean-reversion speed)
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13285_ese_equilibrium_deviation",
    "description": (
        "Equilibrium-state deviation features: rolling z-scores and Euclidean distance "
        "measuring departure from joint OHLCV equilibrium; from arXiv 2606.13285 (ESE)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ese_ret_dev_20",
        "ese_ret_dev_60",
        "ese_vol_dev_20",
        "ese_vol_dev_60",
        "ese_range_dev_20",
        "ese_dist_20",
        "ese_dist_60",
        "ese_convergence",
    ],
    "tags": ["mean_reversion", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13285",
}


def _rolling_zscore(s: pd.Series, window: int, min_periods: int = 10) -> pd.Series:
    """z-score of current value relative to rolling mean and std."""
    mu = s.rolling(window, min_periods=min_periods).mean()
    sigma = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sigma.replace(0.0, np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── raw signals ────────────────────────────────────────────────────────────
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    hl_range = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    log_vol = np.log(df["Volume"].replace(0, np.nan))
    vol_ratio = log_vol - log_vol.rolling(20, min_periods=5).mean()

    # ── per-signal z-scores (deviation from rolling equilibrium) ──────────────
    df["ese_ret_dev_20"]   = _rolling_zscore(log_ret,   20)
    df["ese_ret_dev_60"]   = _rolling_zscore(log_ret,   60)
    df["ese_vol_dev_20"]   = _rolling_zscore(vol_ratio, 20)
    df["ese_vol_dev_60"]   = _rolling_zscore(vol_ratio, 60)
    df["ese_range_dev_20"] = _rolling_zscore(hl_range,  20)

    # ── joint equilibrium distance (ESE's scalar deviation measure) ───────────
    # Each component is already a z-score ⇒ unit-comparable; Euclidean norm.
    ret_z20   = df["ese_ret_dev_20"].values
    vol_z20   = df["ese_vol_dev_20"].values
    range_z20 = df["ese_range_dev_20"].values

    ret_z60  = df["ese_ret_dev_60"].values
    vol_z60  = df["ese_vol_dev_60"].values
    range_z60 = _rolling_zscore(hl_range, 60).values

    def joint_dist(a, b, c):
        """Euclidean distance; NaN if any component is NaN."""
        out = np.sqrt(np.where(
            np.isfinite(a) & np.isfinite(b) & np.isfinite(c),
            a ** 2 + b ** 2 + c ** 2,
            np.nan,
        ))
        return out

    dist20 = joint_dist(ret_z20, vol_z20, range_z20)
    dist60 = joint_dist(ret_z60, vol_z60, range_z60)

    df["ese_dist_20"] = dist20
    df["ese_dist_60"] = dist60

    # ── convergence speed: negative 5-day change in dist20 (positive = contracting) ─
    dist20_s = pd.Series(dist20, index=df.index)
    df["ese_convergence"] = -(dist20_s.diff(5))   # positive when distance is shrinking

    return df
