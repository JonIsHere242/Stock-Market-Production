"""
Temporal-variance-based region-of-interest (ROI) features derived from:
  "Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period
   Estimation from Video" (arXiv 2606.13302 [cs.AI]).

The paper uses automated temporal-variance-based ROI detection to locate
the hydrodynamically active surf-zone: regions of HIGH local variance are
"active" and carry the signal.

OHLCV interpretation:
  - A "turbulent ROI" day is one where the short-window variance spikes relative
    to the longer background variance (wave is breaking = market is turbulent).
  - Key features: local/global variance ratio, variance spike detection, and
    multi-scale variance regime signals.
  - New angle: HIGH-LOW range variance as a distinct "surf zone" proxy,
    plus temporal second-order differences of variance (acceleration of turbulence).

Features:
  - tvr_var_ratio_5_20: 5d vs 20d return-variance ratio
  - tvr_var_ratio_5_60: 5d vs 60d return-variance ratio
  - tvr_range_var_10d: variance of (High-Low)/Close over 10d (intraday range vol)
  - tvr_range_var_ratio_5_20: range-variance ratio 5d/20d
  - tvr_var_accel_20d: first difference of 5d variance z-score (vol acceleration)
  - tvr_roi_score: z-score of log(var_5/var_60) over 60d lookback
  - tvr_var_regime_pct_40d: rolling percentile of 5d variance
  - tvr_hl_expansion_10d: rolling mean of (High/Low - 1), normalised
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13302b_temporal_variance_roi",
    "description": (
        "Multi-scale temporal variance ROI detection for turbulence localization; "
        "adapted from physics-guided temporal-variance ROI (arXiv 2606.13302 cs.AI)."
    ),
    "requires":    ["Close", "High", "Low"],
    "produces": [
        "tvr_var_ratio_5_20",
        "tvr_var_ratio_5_60",
        "tvr_range_var_10d",
        "tvr_range_var_ratio_5_20",
        "tvr_var_accel_20d",
        "tvr_roi_score",
        "tvr_var_regime_pct_40d",
        "tvr_hl_expansion_10d",
    ],
    "tags":        ["volatility", "market_regime", "experimental"],
    "version":     "1.1",
    "author":      "paper:2606.13302b",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    hl_ratio = (df["High"] / df["Low"]) - 1.0   # intraday range (always >= 0)

    eps = 1e-12

    # Rolling variances at multiple scales
    var_5  = log_ret.rolling(5,  min_periods=3).var()
    var_10 = log_ret.rolling(10, min_periods=5).var()
    var_20 = log_ret.rolling(20, min_periods=10).var()
    var_60 = log_ret.rolling(60, min_periods=30).var()

    # Range-based variances (High-Low spread volatility)
    range_var_5  = hl_ratio.rolling(5,  min_periods=3).var()
    range_var_20 = hl_ratio.rolling(20, min_periods=10).var()
    range_mean_10 = hl_ratio.rolling(10, min_periods=5).mean()

    # Variance ratios: short / long (turbulence ROI)
    df["tvr_var_ratio_5_20"]  = var_5  / (var_20 + eps)
    df["tvr_var_ratio_5_60"]  = var_5  / (var_60 + eps)

    # Range variance features
    df["tvr_range_var_10d"]         = range_var_5   # 5d of HL range variance
    df["tvr_range_var_ratio_5_20"]  = range_var_5 / (range_var_20 + eps)

    # Variance acceleration: change in the log of 5d variance (rate of vol change)
    log_var5 = np.log(var_5.clip(lower=eps))
    log_var5_mean = log_var5.rolling(20, min_periods=10).mean()
    log_var5_std  = log_var5.rolling(20, min_periods=10).std()
    log_var5_z    = (log_var5 - log_var5_mean) / (log_var5_std + eps)
    df["tvr_var_accel_20d"] = log_var5_z.diff(3)   # 3-day change in vol z-score

    # ROI composite score: z-score of log(var_5/var_60) over 60d lookback
    log_ratio = np.log((var_5 / (var_60 + eps)).clip(lower=eps))
    lr_mean = log_ratio.rolling(60, min_periods=20).mean()
    lr_std  = log_ratio.rolling(60, min_periods=20).std()
    df["tvr_roi_score"] = (log_ratio - lr_mean) / (lr_std + eps)

    # Rolling percentile of 5d variance (turbulence regime indicator)
    df["tvr_var_regime_pct_40d"] = var_5.rolling(40, min_periods=10).rank(pct=True)

    # Intraday range expansion: rolling mean of HL/Close-1
    range_norm = hl_ratio / (df["Close"] * 0 + 1)   # already normalised
    df["tvr_hl_expansion_10d"] = range_mean_10

    return df
