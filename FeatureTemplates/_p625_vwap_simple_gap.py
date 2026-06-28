import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_vwap_simple_gap",
    "description": "VWAP-minus-simple cumulative return gap: drift between volume-weighted and equal-weighted price/return paths (informed-flow tilt). Berkman, Koch, Tuttle & Zhang (2012) JFE; Brennan, Huh & Subrahmanyam (2018); Madhavan (2002).",
    "requires":    [],
    "produces":    ["vwg_gap_20", "vwg_gap_60", "vwg_vw_simple_ret_60"],
    "tags":        ["volume", "microstructure", "experimental"],
    "version":     "1.0",
    "author":      "paper:Berkman/Koch/Tuttle/Zhang 2012 JFE; Brennan/Huh/Subrahmanyam 2018; Madhavan 2002",
}


def _vwap_gap(close: pd.Series, tp: pd.Series, vol: pd.Series, window: int) -> pd.Series:
    """Trailing VWAP-vs-SMA price gap over `window`, clipped to [-1, 1]."""
    vol_sum = vol.rolling(window).sum()
    vwap = (tp * vol).rolling(window).sum() / vol_sum.replace(0, np.nan)
    sma = close.rolling(window).mean()

    vwap_rel = (close - vwap) / vwap.replace(0, np.nan)
    sma_rel = (close - sma) / sma.replace(0, np.nan)
    gap = vwap_rel - sma_rel
    return gap.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    # Typical price (HLC/3)
    tp = (high + low + close) / 3.0

    # VWAP-vs-SMA price gaps (trailing)
    df["vwg_gap_20"] = _vwap_gap(close, tp, vol, 20)
    df["vwg_gap_60"] = _vwap_gap(close, tp, vol, 60)

    # Volume-weighted minus simple mean daily return over 60 bars (trailing)
    r = close.pct_change()
    vol_sum_60 = vol.rolling(60).sum()
    vwret_60 = (r * vol).rolling(60).sum() / vol_sum_60.replace(0, np.nan)
    sret_60 = r.rolling(60).mean()
    vw_simple = (vwret_60 - sret_60).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    df["vwg_vw_simple_ret_60"] = vw_simple

    return df
