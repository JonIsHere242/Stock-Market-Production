"""
OHLCV Proxy for:
  "A Deep Network-Based Trade and Trend Analysis System to Observe Entry and Exit
   Points in the Forex Market"  —  doi:10.3390/math10193632

The paper trains LSTM variants (Vanilla, Stacked, Bidirectional, CNN-LSTM,
Conv-LSTM) to forecast closing prices, then derives trend direction from the
forecast curve, and validates those trends against ADX, ROC, Momentum, CCI,
and MACD.

PROXY IMPLEMENTATION
--------------------
The LSTM models and Forex-pair inputs are not replicable per-ticker from OHLCV
alone.  However, the paper's own validation benchmark is exactly ADX + ROC +
Momentum + CCI + MACD — the five classical technical indicators the deep network
is benchmarked against.  We implement those five families faithfully as a
per-ticker trend signal pack, which captures the paper's core contribution:
multi-indicator trend analysis for entry/exit timing.

  1. tnd_adx_14     — Average Directional Index (14-period); > 25 = trending
  2. tnd_roc_10     — Rate of Change (10-day); direction + magnitude of momentum
  3. tnd_mom_10     — Raw momentum (Close - Close[10]); price-change velocity
  4. tnd_cci_20     — Commodity Channel Index (20-period); mean-reversion / trend
  5. tnd_macd_sig   — MACD line minus Signal line (12/26/9); classic crossover
  6. tnd_macd_hist  — MACD histogram (signal confirmation)
  7. tnd_composite  — equal-weight z-scored composite of signed trend indicators

Dropped: LSTM/deep-learning forecast, Forex pair data, CNN architecture,
Friedman statistical tests.  The composite (tnd_composite) approximates the
ensemble idea of combining multiple confirming signals.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_doaj_math10193632_deep_network_trade",
    "description": (
        "Trend-analysis indicator pack (ADX, ROC, Momentum, CCI, MACD + composite) "
        "proxying the deep-LSTM trend validation benchmark from doi:10.3390/math10193632. "
        "LSTM/Forex models dropped; classical technical indicators are the paper's own "
        "validation baseline and the implementable signal core."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "tnd_adx_14",
        "tnd_roc_10",
        "tnd_mom_10",
        "tnd_cci_20",
        "tnd_macd_sig",
        "tnd_macd_hist",
        "tnd_composite_6",
    ],
    "tags": ["momentum", "trend", "technical", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.3390/math10193632 (Sharma et al., 2022). "
        "LSTM architecture and Forex data dropped; implements ADX/ROC/Momentum/CCI/MACD "
        "which are the paper's own validation indicators."
    ),
}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # ------------------------------------------------------------------
    # 1. ADX (14-period)
    # ------------------------------------------------------------------
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    # Zero out where the other direction is larger
    mask = dm_plus >= dm_minus
    dm_plus = dm_plus.where(mask, 0.0)
    dm_minus = dm_minus.where(~mask, 0.0)

    period = 14
    atr14 = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    dip14 = dm_plus.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    dim14 = dm_minus.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    di_plus = 100.0 * dip14 / atr14.replace(0, np.nan)
    di_minus = 100.0 * dim14 / atr14.replace(0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx14 = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    df["tnd_adx_14"] = adx14

    # ------------------------------------------------------------------
    # 2. ROC (10-day Rate of Change)
    # ------------------------------------------------------------------
    df["tnd_roc_10"] = close.pct_change(10) * 100.0

    # ------------------------------------------------------------------
    # 3. Momentum (10-day: Close - Close[10])
    # ------------------------------------------------------------------
    df["tnd_mom_10"] = close - close.shift(10)

    # ------------------------------------------------------------------
    # 4. CCI (20-period)
    # ------------------------------------------------------------------
    tp = (high + low + close) / 3.0
    tp_mean = tp.rolling(20, min_periods=20).mean()
    # Mean absolute deviation (factor 0.015 is the standard CCI constant)
    mad = tp.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["tnd_cci_20"] = (tp - tp_mean) / (0.015 * mad.replace(0, np.nan))

    # ------------------------------------------------------------------
    # 5. MACD (12/26/9)
    # ------------------------------------------------------------------
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    df["tnd_macd_sig"] = macd_line - signal_line   # crossover signal
    df["tnd_macd_hist"] = macd_line - signal_line  # same as macd_sig; standard histogram

    # ------------------------------------------------------------------
    # 6. Composite: z-score each signed indicator, average them
    #    ADX is unsigned (0-100) so we skip it from the composite;
    #    ROC, Momentum, CCI, MACD-sig all have natural sign
    # ------------------------------------------------------------------
    def _zscore_roll(s: pd.Series, w: int = 252) -> pd.Series:
        mu = s.rolling(w, min_periods=60).mean()
        sd = s.rolling(w, min_periods=60).std()
        return (s - mu) / sd.replace(0, np.nan)

    z_roc = _zscore_roll(df["tnd_roc_10"])
    z_mom = _zscore_roll(df["tnd_mom_10"])
    z_cci = _zscore_roll(df["tnd_cci_20"])
    z_mac = _zscore_roll(df["tnd_macd_sig"])

    df["tnd_composite_6"] = (z_roc + z_mom + z_cci + z_mac) / 4.0

    return df
