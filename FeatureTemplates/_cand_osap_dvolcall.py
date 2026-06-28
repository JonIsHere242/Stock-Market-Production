"""
osap_dvolcall — Dollar-Volume-Weighted Call-Option-Like Demand (per-ticker proxy)

Spec ID   : osap_dvolcall
Source    : OpenSourceAP (Chen-Zimmermann); original signal from the demand-for-lottery
            / call-option demand literature.
            Closest academic basis: Bali, Cakici & Whitelaw (2011) "Maxing Out: Stocks
            as Lotteries and the Cross-Section of Expected Returns"; also related to
            Boyer, Mitton & Vorkink (2010) "Expected Idiosyncratic Skewness".
Predicted sign (cross-sectional): -1  (high demand-for-call-like payoffs → overpriced →
            lower future returns)

Economic signal:
    Stocks whose recent return distribution exhibits call-option-like characteristics
    (positive skewness, high maximum daily gain, high idiosyncratic volatility relative
    to its own trend) attract lottery-seeking investors.  When that attraction is
    amplified by heavy dollar-volume turnover, the stock is likely to be overpriced
    relative to fundamentals.  The signal is:
        dvolcall = log(dollar_volume_2m) × skewness_signal
    where skewness_signal = rolling positive skewness of daily returns over 2 months.

Per-ticker proxy note:
    The OSAP cross-sectional version sorts stocks by their dollar-volume-weighted
    skewness vs. peers.  This block implements the WITHIN-ticker version: it captures
    the same economic logic (lottery demand inflates prices) using only the stock's own
    history.  Cross-sectional ranking is left to the feature framework.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "osap_dvolcall",
    "description": (
        "Per-ticker proxy for dollar-volume-weighted call-option-like demand. "
        "Combines log 2-month dollar volume with rolling return skewness to capture "
        "lottery-seeking investor demand. High values → likely overpriced → negative "
        "predicted return (cross-sectional sign: -1). "
        "Inherently cross-sectional in the original OSAP spec; this is a faithful "
        "per-ticker proxy capturing the same economic signal (lottery demand × turnover)."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "osap_dvolcall_raw",      # log(dollar_vol_2m) × clipped positive skew (2m)
        "osap_dvolcall_skew21",   # standalone rolling 21-day return skewness
        "osap_dvolcall_maxret21", # max single-day return over 21 days (MAX anomaly proxy)
    ],
    "tags": ["volume", "lottery", "skewness", "liquidity", "osap"],
    "version": "1.0",
    "author": (
        "Spec: OpenSourceAP (Chen-Zimmermann); "
        "Bali, Cakici & Whitelaw (2011) 'Maxing Out'; "
        "Boyer, Mitton & Vorkink (2010) 'Expected Idiosyncratic Skewness'. "
        "Block impl: Claude (per-ticker proxy)."
    ),
}

# Window sizes (trading days)
_WIN_SHORT = 21   # ~1 month
_WIN_LONG  = 42   # ~2 months
_MIN_OBS   = 15   # minimum non-NaN observations required


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)
    n      = len(df)

    # Daily log returns (no lookahead: shift(1) uses past close)
    log_ret = np.empty(n)
    log_ret[:] = np.nan
    prev_close = np.roll(close, 1)
    # first element is undefined (wrap-around) — set to NaN
    mask_valid = (prev_close > 0) & (close > 0)
    log_ret[1:] = np.where(
        mask_valid[1:],
        np.log(close[1:] / prev_close[1:]),
        np.nan,
    )

    # Dollar volume (daily): Close × Volume
    dv = np.where((close > 0) & (volume > 0), close * volume, np.nan)

    # ---- Rolling helpers (no scipy, pure numpy/pandas) ----
    sr = pd.Series(log_ret, index=df.index)
    dv_s = pd.Series(dv, index=df.index)

    # 1. Rolling 21-day return skewness  (osap_dvolcall_skew21)
    skew21 = sr.rolling(window=_WIN_SHORT, min_periods=_MIN_OBS).skew()

    # 2. Rolling 21-day maximum daily return  (MAX anomaly proxy)
    maxret21 = sr.rolling(window=_WIN_SHORT, min_periods=_MIN_OBS).max()

    # 3. Rolling 42-day log sum of dollar volume  → log(sum DV over 2m)
    #    Use sum then log; guard zero sums.
    dv_sum42 = dv_s.rolling(window=_WIN_LONG, min_periods=_MIN_OBS).sum()
    log_dv42 = np.where(dv_sum42 > 0, np.log(dv_sum42), np.nan)
    log_dv42 = pd.Series(log_dv42, index=df.index)

    # 4. Composite: log_dv42 × clipped positive skew
    #    We clip skew to [0, inf) so the signal is only active when skew is positive
    #    (lottery-like upside), matching the economic motivation.  Negative-skew stocks
    #    are set to 0 contribution from the skew term; they get NaN via log_dv (still
    #    informative as pure dollar-volume signal when we take the product).
    #    Replace inf/-inf with NaN just in case.
    skew_pos = skew21.clip(lower=0.0)
    dvolcall_raw = log_dv42 * skew_pos
    dvolcall_raw = dvolcall_raw.replace([np.inf, -np.inf], np.nan)

    # Assign produced columns
    df["osap_dvolcall_raw"]     = dvolcall_raw.values
    df["osap_dvolcall_skew21"]  = skew21.replace([np.inf, -np.inf], np.nan).values
    df["osap_dvolcall_maxret21"] = maxret21.replace([np.inf, -np.inf], np.nan).values

    return df
