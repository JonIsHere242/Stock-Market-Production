"""
osap_skew1 — Idiosyncratic / total return skewness

Source: Open Source Asset Pricing (OSAP) — Chen & Zimmermann (2022),
        replicating Harvey & Siddique (2000) "Conditional Skewness in Asset
        Pricing Tests", Journal of Finance 55(4); and Bali, Cakici & Whitelaw
        (2011) "Maxing Out: Stocks as Lotteries and the Cross-Section of
        Expected Returns", Journal of Financial Economics 99(2).

Economic signal:
    Stocks with high positive skewness in recent returns earn LOWER subsequent
    returns (lottery-demand / overpricing of positive skewness).  High skew =>
    negative expected alpha.  This is the "skewness premium" — investors
    overpay for positively-skewed (lottery-like) stocks.

Per-ticker proxy:
    True idiosyncratic skewness requires a cross-sectional residual from a
    factor model.  Here we compute TOTAL daily-return skewness over a trailing
    window (3-month ~ 63 trading days).  This captures the same economic signal
    at the per-ticker level and is the dominant component (Harvey & Siddique
    show total ≈ idiosyncratic skewness for single-stock prediction).

Produces:
    osap_skew1_63d   — trailing 63-day skewness of daily close returns
    osap_skew1_21d   — trailing 21-day skewness (short-horizon variant)
    osap_skew1_chg   — 63d skewness minus its own 126-day rolling mean
                       (dynamic: captures shift in skewness regime)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_skew1",
    "description": (
        "Trailing return skewness (63d and 21d) plus a dynamic skewness-change "
        "signal. High positive skewness predicts lower future returns (lottery "
        "demand / overpricing). Per-ticker proxy for OSAP idiosyncratic skewness "
        "factor (Harvey & Siddique 2000; Bali, Cakici & Whitelaw 2011 MAX effect; "
        "Chen & Zimmermann 2022 OSAP replication)."
    ),
    "requires": ["Close"],
    "produces": ["osap_skew1_63d", "osap_skew1_21d", "osap_skew1_chg"],
    "tags": ["skewness", "lottery", "cross-sectional", "osap", "returns"],
    "version": "1.0.0",
    "author": (
        "Harvey & Siddique (2000) JF; Bali, Cakici & Whitelaw (2011) JFE; "
        "Chen & Zimmermann (2022) OSAP — implemented as per-ticker proxy"
    ),
}


def _rolling_skew(ret: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling skewness using the adjusted Fisher-Pearson formula,
    vectorised via pandas rolling.  Requires at least 3 non-NaN obs;
    returns NaN otherwise.
    """
    return ret.rolling(window, min_periods=max(3, window // 3)).skew()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Daily log-return (shift(1) uses only past prices — no lookahead)
    ret = np.log(df["Close"] / df["Close"].shift(1))

    # Level signals
    skew_63 = _rolling_skew(ret, 63)
    skew_21 = _rolling_skew(ret, 21)

    # Dynamic change: current 63d skewness minus its 126-day rolling mean
    # (captures shift from recent skewness history — sign reversal signal)
    skew_63_ma = skew_63.rolling(126, min_periods=42).mean()
    skew_chg = skew_63 - skew_63_ma

    df["osap_skew1_63d"] = skew_63
    df["osap_skew1_21d"] = skew_21
    df["osap_skew1_chg"] = skew_chg

    return df
