"""
Technical Indicator Empowered Trading Signals  —  DOI:10.1016/j.joitmc.2024.100398
"Technical indicator empowered intelligent strategies to predict stock trading signals"

The paper uses MACD, DMI (Directional Movement Index), and KST (Know Sure Thing) as
inputs to LSTM/GRU models. The raw indicator values are computed here per-ticker from
OHLCV — the ML model layer is dropped (we expose the indicators as features).

  * MACD: (EMA12 - EMA26), MACD signal (EMA9 of MACD), MACD histogram.
  * DMI: +DI, -DI, ADX (Average Directional Index) — standard Wilder formulas.
  * KST: sum of four smoothed ROC components (10,15,20,30 day ROCs, smoothed by SMA 10,10,10,15).

The paper found a 5-day lookback optimal for MACD/DMI and 10-day for KST strategies. We
preserve all three families as separate named columns; the downstream model chooses.

No LSTM/GRU model is trained here — only the indicator computation (the feature layer).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_doaj_jjoitmc2024100_technical_indicator_empowered",
    "description": (
        "MACD, DMI (+DI/-DI/ADX), and KST technical indicators per-ticker from OHLCV; "
        "implements the feature layer of DOI:10.1016/j.joitmc.2024.100398 (LSTM/GRU model layer dropped)."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ti_macd_line",
        "ti_macd_signal",
        "ti_macd_hist",
        "ti_dmi_plus_di",
        "ti_dmi_minus_di",
        "ti_dmi_adx",
        "ti_dmi_dx",
        "ti_kst",
        "ti_kst_signal",
    ],
    "tags": ["momentum", "trend", "technical", "experimental"],
    "version": "1.0",
    "author": (
        "paper:10.1016/j.joitmc.2024.100398 (MACD/DMI/KST indicator layer; "
        "LSTM/GRU model and cross-ticker optimization dropped)"
    ),
}


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential smoothing: alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD, DMI, and KST indicators.

    All computations are causal (rolling over past data only). NaNs during
    warm-up are expected.
    """
    high  = df["High"].astype(np.float64)
    low   = df["Low"].astype(np.float64)
    close = df["Close"].astype(np.float64)

    # ---- MACD ---------------------------------------------------------------
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    macd_hist   = macd_line - macd_signal

    df["ti_macd_line"]   = macd_line
    df["ti_macd_signal"] = macd_signal
    df["ti_macd_hist"]   = macd_hist

    # ---- DMI / ADX ----------------------------------------------------------
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)

    # Zero out where the opposite move is larger
    cond_plus_larger  = dm_plus > dm_minus
    cond_minus_larger = dm_minus > dm_plus
    dm_plus  = dm_plus.where(cond_plus_larger, 0.0)
    dm_minus = dm_minus.where(cond_minus_larger, 0.0)

    period = 14
    tr14   = _wilder_smooth(tr,       period)
    dmp14  = _wilder_smooth(dm_plus,  period)
    dmm14  = _wilder_smooth(dm_minus, period)

    plus_di  = 100.0 * dmp14 / tr14.replace(0, np.nan)
    minus_di = 100.0 * dmm14 / tr14.replace(0, np.nan)
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = _wilder_smooth(dx, period)

    df["ti_dmi_plus_di"]  = plus_di
    df["ti_dmi_minus_di"] = minus_di
    df["ti_dmi_adx"]      = adx
    df["ti_dmi_dx"]       = dx

    # ---- KST (Know Sure Thing) ----------------------------------------------
    # KST = RCMA1 * 1 + RCMA2 * 2 + RCMA3 * 3 + RCMA4 * 4
    # where RCMAn = SMA(n_smooth, ROC(n_roc))
    # Standard parameters: ROC 10,15,20,30; SMA smooth 10,10,10,15; weights 1,2,3,4
    def _roc(n: int) -> pd.Series:
        """Rate of change: (close / close[n periods ago] - 1) * 100"""
        return (close / close.shift(n) - 1.0) * 100.0

    rcma1 = _roc(10).rolling(10, min_periods=5).mean()
    rcma2 = _roc(15).rolling(10, min_periods=5).mean()
    rcma3 = _roc(20).rolling(10, min_periods=5).mean()
    rcma4 = _roc(30).rolling(15, min_periods=8).mean()

    kst = rcma1 * 1.0 + rcma2 * 2.0 + rcma3 * 3.0 + rcma4 * 4.0
    kst_signal = kst.rolling(9, min_periods=5).mean()

    df["ti_kst"]        = kst
    df["ti_kst_signal"] = kst_signal

    return df
