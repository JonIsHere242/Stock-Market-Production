"""
Forecasting the Timing of Transactions in Tehran Stock Exchange
DOI: 10.22103/jdc.2020.12002.1048
Venue: Majallah-i Tawsi'ah va Sarmayah (2020)

The paper uses a fuzzy neural network (FNN) trained on five technical indicators
(RSI, MACD, SMA, Stochastic, EMA/Signal line) to predict buy/sell timing with
~96.55% accuracy on Tehran Stock Exchange data.

PROXY IMPLEMENTED: Pure per-ticker composite of the five exact technical indicators
the paper features, weighted equally into a single directional score. No fuzzy
inference, no neural network — those require labelled training data on TSE data we
don't have. The composite signal captures the same information the FNN ingested.

DROPPED: FNN architecture, TSE-specific calibration, transaction-cost adjustment.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_jdc20201200210_timing_tehran",
    "description": (
        "Composite timing signal from 5 technical indicators (RSI, MACD, Stochastic, "
        "SMA crossover, EMA/signal crossover) as proxy for the fuzzy neural network "
        "timing model of Amiri et al. (2020) on Tehran Stock Exchange; FNN and TSE "
        "calibration dropped — indicator arithmetic only."
    ),
    "requires": ["Close", "High", "Low"],
    "produces": [
        "tehran_rsi_14",
        "tehran_stoch_k_14",
        "tehran_stoch_d_3",
        "tehran_macd_line",
        "tehran_macd_signal",
        "tehran_macd_hist",
        "tehran_sma_cross_5_20",
        "tehran_ema_cross_12_26",
        "tehran_composite_score",
    ],
    "tags": ["momentum", "trend", "technical", "experimental"],
    "version": "1.0",
    "author": "proxy: paper DOI:10.22103/jdc.2020.12002.1048 (Amiri et al., 2020); FNN dropped",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 technical indicators used by the Tehran timing paper and combine
    them into a signed composite score in [-1, +1].

    RSI normalised to [-1, +1]:  (rsi - 50) / 50
    Stochastic %K normalised:    (%K - 50) / 50
    MACD histogram sign:         sign(macd - signal)
    SMA crossover:               sign(sma5 - sma20)
    EMA crossover:               sign(ema12 - ema26)

    All five normalised to [-1, +1], averaged → tehran_composite_score in [-1, +1].
    Positive = bullish composite; negative = bearish.
    """
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # ---- RSI 14 (Wilder smoothing) ------------------------------------------
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi_14   = 100.0 - 100.0 / (1.0 + rs)
    df["tehran_rsi_14"] = rsi_14

    # ---- Stochastic %K (14) and %D (3-period SMA of %K) ----------------------
    low_14  = low.rolling(14, min_periods=14).min()
    high_14 = high.rolling(14, min_periods=14).max()
    denom   = (high_14 - low_14).replace(0.0, float("nan"))
    stoch_k = 100.0 * (close - low_14) / denom
    stoch_d = stoch_k.rolling(3, min_periods=3).mean()
    df["tehran_stoch_k_14"] = stoch_k
    df["tehran_stoch_d_3"]  = stoch_d

    # ---- MACD (12, 26, 9) ---------------------------------------------------
    ema12       = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26       = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    df["tehran_macd_line"]   = macd_line
    df["tehran_macd_signal"] = macd_signal
    df["tehran_macd_hist"]   = macd_hist

    # ---- SMA crossover: SMA(5) vs SMA(20) ------------------------------------
    sma5  = close.rolling(5,  min_periods=5).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    sma_cross = np.sign(sma5 - sma20)
    df["tehran_sma_cross_5_20"] = sma_cross

    # ---- EMA crossover: EMA(12) vs EMA(26) (Signal line) --------------------
    ema_cross = np.sign(ema12 - ema26)
    df["tehran_ema_cross_12_26"] = ema_cross

    # ---- Composite score: average of 5 normalised indicator values ----------
    #  RSI → (rsi - 50) / 50   ∈ [-1, +1]
    #  Stoch %K → (%K - 50) / 50
    #  MACD histogram sign → already ∈ {-1, 0, +1}
    #  SMA cross → already ∈ {-1, 0, +1}
    #  EMA cross → already ∈ {-1, 0, +1}
    rsi_norm    = (rsi_14 - 50.0) / 50.0
    stoch_norm  = (stoch_k - 50.0) / 50.0
    macd_sign   = np.sign(macd_hist)
    composite   = (rsi_norm + stoch_norm + macd_sign + sma_cross + ema_cross) / 5.0
    df["tehran_composite_score"] = composite

    return df
