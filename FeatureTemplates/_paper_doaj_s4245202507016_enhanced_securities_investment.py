"""
Adaptive Moving Average (AMA / Kaufman AMA) Features  —  DOAJ:s4245202507016
"Enhanced securities investment strategy using ISSA–SVM: a hybrid model
combining adaptive moving average, support vector machine, and multi-strategy
sparrow search algorithm for improved trend tracking and risk adjustment"
(Discover Applied Sciences, 2025)

The paper's core innovation for trend tracking is the AMA (Adaptive Moving
Average) that dynamically adjusts smoothing coefficients to market conditions,
improving response speed in trending environments and dampening in sideways markets.

PROXY APPROACH (per-ticker, causal):
  The AMA used here follows Kaufman's KAMA (Perry Kaufman, 1998) which the ISSA-SVM
  paper's AMA closely resembles.  Key formula:
    ER  = |Close_t - Close_{t-n}| / sum(|Close_i - Close_{i-1}|, i=t-n+1..t)
    SC  = (ER * (fast_sc - slow_sc) + slow_sc)^2       # smoothing constant
    AMA = AMA_{t-1} + SC * (Close_t - AMA_{t-1})       # update

  Features derived:
  1. ama_10_2_30   : KAMA(10, 2, 30) — standard params
  2. ama_21_2_30   : KAMA(21, 2, 30) — longer efficiency window
  3. ama_er_10     : Efficiency ratio (ER) over 10 bars — measures trendiness
  4. ama_er_21     : ER over 21 bars
  5. ama_dist_10   : Close / AMA(10,2,30) - 1 — price distance from AMA
  6. ama_dist_21   : Close / AMA(21,2,30) - 1
  7. ama_slope_10  : 5-bar rate of change of AMA(10,2,30) — trend acceleration
  8. ama_cross_sig : Sign of (AMA_10 - AMA_21) — fast/slow AMA crossover signal

DROPPED: SVM classifier, Sparrow Search Algorithm hyper-parameter optimization,
multi-strategy meta-optimizer — none implementable as a stateless OHLCV compute().
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_doaj_s4245202507016_enhanced_securities_investment",
    "description": (
        "Kaufman Adaptive Moving Average (KAMA) features as proxy for the AMA-based "
        "CTA trend tracker in the ISSA-SVM paper (DOAJ s4245202507016): AMA values "
        "at two efficiency windows, efficiency ratios, AMA-distance from price, trend "
        "slope, and fast/slow AMA crossover.  SVM + sparrow-search components omitted."
    ),
    "requires": ["Close"],
    "produces": [
        "ama_kama_10_2_30",
        "ama_kama_21_2_30",
        "ama_er_10",
        "ama_er_21",
        "ama_dist_10",
        "ama_dist_21",
        "ama_slope_10",
        "ama_cross_sig",
    ],
    "tags": ["trend", "momentum", "experimental"],
    "version": "1.0",
    "author": (
        "proxy:doaj-s4245202507016 — Kaufman KAMA as faithful per-ticker proxy "
        "for the paper's adaptive moving average trend tracker; SVM/ISSA omitted."
    ),
}


def _kama(close: np.ndarray, n: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Compute Kaufman Adaptive Moving Average (KAMA) on a 1-D close array.

    Parameters
    ----------
    close : np.ndarray  shape (T,)
    n     : efficiency-ratio look-back window
    fast  : fast EMA period (default 2)
    slow  : slow EMA period (default 30)

    Returns
    -------
    kama  : np.ndarray shape (T,) with NaN for the first n-1 bars
    """
    T = len(close)
    kama = np.full(T, np.nan)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    # First valid bar is at index n-1 (need n bars to form the first ER)
    if T < n:
        return kama

    # Initialise KAMA at the first valid bar
    kama[n - 1] = close[n - 1]

    for i in range(n, T):
        # Efficiency ratio
        direction = abs(close[i] - close[i - n])
        volatility = np.sum(np.abs(np.diff(close[i - n: i + 1])))
        er = direction / volatility if volatility > 1e-12 else 0.0

        # Smoothing constant (squared to de-amplify noisy markets)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

    return kama


def _efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorised Efficiency Ratio over window n (causal).

    ER_t = |Close_t - Close_{t-n}| / sum(|delta_i|, i=t-n+1..t)
    """
    T = len(close)
    er = np.full(T, np.nan)
    if T < n + 1:
        return er

    close_s = pd.Series(close)
    direction = (close_s - close_s.shift(n)).abs()
    noise = close_s.diff().abs().rolling(window=n, min_periods=n).sum()
    valid = noise > 1e-12
    er_s = np.where(valid, direction / noise, np.nan)
    # Leading n bars are NaN automatically from shift/rolling
    er[:] = er_s
    return er


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KAMA (Adaptive Moving Average) features.

    All operations are causal. Leading NaNs expected during warm-up.
    """
    close = df["Close"].values.astype(np.float64)
    close_s = df["Close"]

    # ---- KAMA at two efficiency windows ------------------------------------
    kama_10 = _kama(close, n=10, fast=2, slow=30)
    kama_21 = _kama(close, n=21, fast=2, slow=30)

    df["ama_kama_10_2_30"] = kama_10
    df["ama_kama_21_2_30"] = kama_21

    # ---- Efficiency ratios -------------------------------------------------
    er_10 = _efficiency_ratio(close, n=10)
    er_21 = _efficiency_ratio(close, n=21)

    df["ama_er_10"] = er_10
    df["ama_er_21"] = er_21

    # ---- Price distance from AMA (signed) ----------------------------------
    # Positive = price above AMA (uptrend); negative = below (downtrend)
    kama_10_s = pd.Series(kama_10, index=df.index)
    kama_21_s = pd.Series(kama_21, index=df.index)

    df["ama_dist_10"] = close_s / kama_10_s.replace(0, np.nan) - 1.0
    df["ama_dist_21"] = close_s / kama_21_s.replace(0, np.nan) - 1.0

    # ---- AMA slope: 5-bar rate of change of KAMA (trend acceleration) ------
    ama_slope = kama_10_s / kama_10_s.shift(5).replace(0, np.nan) - 1.0
    df["ama_slope_10"] = ama_slope

    # ---- Fast / slow AMA crossover signal ----------------------------------
    # +1 when fast AMA > slow AMA (uptrend), -1 when below, 0 when equal
    cross = np.sign(kama_10 - kama_21).astype(float)
    cross[np.isnan(kama_10) | np.isnan(kama_21)] = np.nan
    df["ama_cross_sig"] = cross

    return df
