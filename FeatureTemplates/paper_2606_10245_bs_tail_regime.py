"""
Black-Scholes Tail Regime Features
Paper: "A Fast Implied Volatility Method with Expansions" (arXiv 2606.10245)

The paper's core contribution is regime-split seed construction using the
asymptotic structure of the Black-Scholes price function:
  - ATM regime: series reversion of exact Gaussian identity
  - Moderate OTM: successive Gaussian CDF approximations
  - Deep OTM: Mills ratio (Gaussian tail cancellation identity)

OHLCV adaptation: we replicate the REGIME-SPLIT philosophy by measuring
where the current price sits within its distributional tails.
  - Compute a rolling z-score (standardized distance from rolling mean)
  - Classify into ATM / moderate-tail / deep-tail regimes using quantile thresholds
    derived analytically from Gaussian CDF (mirroring the paper's regime boundaries)
  - Apply the Mills ratio approximation: φ(z)/Φ(-z) ≈ z + 1/z for large |z|
    (ratio of pdf to survival function) as a tail-amplified signal
  - Produce regime-conditional z-scores and Mills-ratio corrected signals across
    multiple horizons
"""

import pandas as pd
import numpy as np
from typing import List

METADATA = {
    "name":        "paper_2606_10245_bs_tail_regime",
    "description": (
        "Gaussian-regime-split price tail features inspired by "
        "Black-Scholes IV seed construction with Mills ratio (arXiv 2606.10245)"
    ),
    "requires":    ["Close", "High", "Low"],
    "produces":    [
        # Rolling z-score of log-price relative to rolling mean / std
        "bstr_zscore_20d",
        "bstr_zscore_60d",
        # Regime flag: 0=ATM (|z|<0.67), 1=moderate-tail (0.67<=|z|<1.96), 2=deep-tail (|z|>=1.96)
        "bstr_regime_20d",
        "bstr_regime_60d",
        # Mills-ratio-corrected tail signal: sign(z) * |z| / (1 + 1/z^2) for deep tail
        # (approximates phi(z)/(1-Phi(z)) behaviour), else raw z-score
        "bstr_mills_signal_20d",
        "bstr_mills_signal_60d",
        # Regime-conditional mean-reversion score: z * (-1) in deep tail (expect reversion)
        "bstr_tail_reversion_20d",
        "bstr_tail_reversion_60d",
        # ATM velocity: rate of change of z-score (momentum near centre is informative)
        "bstr_zscore_velocity_20d",
    ],
    "tags":        ["volatility", "market_regime", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.10245",
}


def _mills_signal(z: pd.Series) -> pd.Series:
    """
    Mills-ratio-corrected tail signal.
    For |z| > 1.96 (deep tail): sign(z) * (|z| + 1/|z|)  -- Mills ratio approx
    For |z| in [0.67, 1.96] (moderate): raw z
    For |z| < 0.67 (ATM): z (near-ATM, no amplification)
    """
    az = z.abs()
    # Mills amplification in the deep tail: phi(z)/Phi(-z) ~ z + 1/z
    mills = np.sign(z) * (az + 1.0 / az.clip(lower=1e-6))
    result = z.copy()
    deep_mask = az >= 1.96
    result = result.where(~deep_mask, mills)
    return result


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_close = np.log(close.clip(lower=1e-8))

    results = {}

    for w in [20, 60]:
        # Rolling mean and std of log-price
        rm = log_close.rolling(w, min_periods=w // 2).mean()
        rs = log_close.rolling(w, min_periods=w // 2).std()

        z = (log_close - rm) / rs.clip(lower=1e-8)
        results[f"bstr_zscore_{w}d"] = z

        # Regime classification based on Gaussian quantiles
        # |z| < 0.674 ~ central 50% (ATM-equivalent)
        # 0.674 <= |z| < 1.96 ~ moderate tail
        # |z| >= 1.96 ~ deep tail (2.5% each side)
        az = z.abs()
        regime = pd.Series(0.0, index=df.index)
        regime = regime.where(az < 0.674, 1.0)
        regime = regime.where(az < 1.96, regime)
        regime = regime.where(az < 1.96, 2.0)
        # Correct: regime=2 when az>=1.96, regime=1 when 0.674<=az<1.96, regime=0 when az<0.674
        regime = az.apply(lambda x: 0.0 if x < 0.674 else (1.0 if x < 1.96 else 2.0))
        results[f"bstr_regime_{w}d"] = regime

        # Mills-ratio corrected signal
        results[f"bstr_mills_signal_{w}d"] = _mills_signal(z)

        # Tail reversion score: in deep tail, expect mean-reversion → negative z is bullish
        az_ser = z.abs()
        tail_mask = az_ser >= 1.96
        # In deep tail: reversion signal = -z (positive when price is depressed)
        # Outside deep tail: 0 (no strong reversion expectation)
        reversion = pd.Series(0.0, index=df.index)
        reversion[tail_mask] = -z[tail_mask]
        results[f"bstr_tail_reversion_{w}d"] = reversion

    # ATM velocity (rate of change of 20d z-score)
    z20 = results["bstr_zscore_20d"]
    results["bstr_zscore_velocity_20d"] = z20.diff(3)

    # Assign all new columns at once
    for col, series in results.items():
        df[col] = series

    return df
