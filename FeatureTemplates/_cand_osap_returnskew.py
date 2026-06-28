"""
osap_returnskew — Rolling return skewness (idiosyncratic lottery premium proxy).

Source: Harvey & Siddique (2000) "Conditional Skewness in Asset Pricing Tests",
        Journal of Finance 55(4); Boyer, Mitton & Vorkink (2010) "Expected Idiosyncratic
        Skewness", Review of Financial Studies 23(1); and the Open Source Asset Pricing
        (Chen & Zimmermann 2022) anomaly catalogue entry "returnskew".

Economic signal: Stocks with high realized skewness of returns attract lottery-seeking
investors who overpay, depressing subsequent returns.  Negative predictive relationship
between skewness and forward returns (sell high-skew, buy low-skew).

Per-ticker proxy: We cannot remove the market component here (no cross-section), so we
compute TOTAL daily-return skewness inside rolling windows.  This is the closest
faithful single-stock proxy to the idiosyncratic skewness used in the paper.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_returnskew",
    "description": (
        "Rolling skewness of daily log-returns over 21-day (1-month) and 63-day "
        "(3-month) windows.  High skewness is associated with lower future returns "
        "(lottery-demand overpricing).  Also computes change in 21-day skewness over "
        "the prior 21 days as a momentum-of-skew signal.  Per-ticker OHLCV proxy for "
        "the OSAP/Chen-Zimmermann 'returnskew' anomaly; market component NOT removed "
        "(no cross-section available at block-compute time)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_returnskew_21d",   # rolling 21-day skewness of log returns
        "osap_returnskew_63d",   # rolling 63-day skewness of log returns
        "osap_returnskew_chg",   # change in 21-day skew over last 21 days (skew momentum)
    ],
    "tags": ["skewness", "lottery", "osap", "returns", "risk"],
    "version": "1.0.0",
    "author": "Harvey & Siddique (2000); Boyer, Mitton & Vorkink (2010); OSAP catalogue (Chen & Zimmermann 2022)",
}


def _rolling_skew(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Pandas rolling skew — vectorised, no python loop."""
    return series.rolling(window=window, min_periods=min_periods).skew()


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Daily log returns — shift(1) is strictly backward, no lookahead
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # 1-month skewness (21 trading days); require at least 10 obs
    skew_21 = _rolling_skew(log_ret, window=21, min_periods=10)

    # 3-month skewness (63 trading days); require at least 30 obs
    skew_63 = _rolling_skew(log_ret, window=63, min_periods=30)

    # Change in 21-day skew over the prior 21 days
    # skew_21.shift(21) is the value 21 bars ago — entirely in the past, safe
    skew_chg = skew_21 - skew_21.shift(21)

    df["osap_returnskew_21d"] = skew_21
    df["osap_returnskew_63d"] = skew_63
    df["osap_returnskew_chg"] = skew_chg

    return df
