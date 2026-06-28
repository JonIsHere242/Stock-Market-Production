"""
mst_implied_spread.py — Daily-OHLCV implied bid-ask spread estimators.

Three classic, structurally DIFFERENT estimators of the effective bid-ask
spread that need only daily Open/High/Low/Close (no intraday data):

  1. Corwin-Schultz (2012) high-low spread estimator.  Uses two consecutive
     daily high-low ranges to separate the variance component from the
     bid-ask-bounce component.  Spreads are floored at 0 (negative raw
     estimates -> 0, the standard convention).
  2. Roll (1984) effective spread from the serial covariance of close-to-close
     returns: spread = 2*sqrt(-cov(r_t, r_{t-1})) when cov < 0, else 0.
  3. Abdi-Ranaldo (2017) "CHL" close-high-low estimator — a more robust
     high-low spread proxy using log mid-range vs log close.

These are spread (transaction-cost / liquidity) signals, orthogonal to plain
price momentum and to the existing dollar-volume liquidity block.  All outputs
are clipped to sane non-negative bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "mst_implied_spread",
    "description": (
        "Daily-OHLCV implied bid-ask spread estimators: Corwin-Schultz high-low "
        "(2-day + 20d mean), Roll serial-covariance effective spread (20/60d), "
        "and Abdi-Ranaldo close-high-low spread (20d)."
    ),
    "requires":    ["High", "Low", "Close"],
    "produces":    [
        "mst_corwin_schultz_2d",
        "mst_corwin_schultz_20d",
        "mst_roll_spread_20d",
        "mst_roll_spread_60d",
        "mst_abdi_ranaldo_chl_20d",
    ],
    "tags":        ["liquidity", "microstructure", "spread"],
    "version":     "1.0",
    "author":      "feature-gen",
}

# Spread fractions above this are pathological for liquid equities -> clip.
_SPREAD_CAP = 0.25  # 25% of price


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # Guard against non-positive prices before taking logs.
    high_s = high.where(high > 0)
    low_s = low.where(low > 0)
    close_s = close.where(close > 0)

    new_cols = {}

    # ------------------------------------------------------------------
    # 1. Corwin-Schultz (2012) high-low spread estimator
    #    beta = sum over 2 days of [ln(H/L)]^2
    #    gamma = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]^2
    #    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt2)
    #            - sqrt(gamma / (3 - 2*sqrt2))
    #    S = 2*(e^alpha - 1) / (1 + e^alpha)   (floored at 0)
    # ------------------------------------------------------------------
    log_hl = np.log(high_s / low_s)            # ln(H/L) per day, >= 0
    beta_t = log_hl ** 2                        # single-day squared range
    # beta over two consecutive days
    beta = beta_t + beta_t.shift(1)

    # Two-day high / low for the gamma term
    high_2 = pd.concat([high_s, high_s.shift(1)], axis=1).max(axis=1)
    low_2 = pd.concat([low_s, low_s.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high_2 / low_2) ** 2

    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)

    cs_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # Negative spread estimates are set to 0 (Corwin-Schultz convention).
    cs_spread = cs_spread.clip(lower=0.0, upper=_SPREAD_CAP)
    cs_spread = cs_spread.replace([np.inf, -np.inf], np.nan)

    new_cols["mst_corwin_schultz_2d"] = cs_spread
    # Smoothed 20-day average of the daily CS estimate (its standard use).
    new_cols["mst_corwin_schultz_20d"] = cs_spread.rolling(
        20, min_periods=10
    ).mean()

    # ------------------------------------------------------------------
    # 2. Roll (1984) effective spread from serial covariance of returns.
    #    cov = Cov(r_t, r_{t-1}); S = 2*sqrt(-cov) when cov<0, else 0.
    #    Use simple close-to-close returns; rolling covariance windows.
    # ------------------------------------------------------------------
    ret = close_s.pct_change()
    ret_lag = ret.shift(1)
    for w in (20, 60):
        cov = ret.rolling(w, min_periods=w // 2).cov(ret_lag)
        # Roll spread defined only where serial covariance is negative.
        roll = 2.0 * np.sqrt((-cov).clip(lower=0.0))
        roll = roll.clip(lower=0.0, upper=_SPREAD_CAP)
        roll = roll.replace([np.inf, -np.inf], np.nan)
        new_cols[f"mst_roll_spread_{w}d"] = roll

    # ------------------------------------------------------------------
    # 3. Abdi-Ranaldo (2017) CHL spread estimator.
    #    eta_t = (ln H_t + ln L_t) / 2  (log mid-range)
    #    S^2 = 4 * E[ (c_t - eta_t) * (c_t - eta_{t-1}) ]
    #    Lookahead-SAFE variant: uses the PRIOR-day mid-range (shift +1), never
    #    the future, so the feature is computable in real time at the close.
    #    Implemented with a 20-day rolling mean of the cross term; floored at 0.
    # ------------------------------------------------------------------
    c = np.log(close_s)
    eta = (np.log(high_s) + np.log(low_s)) / 2.0
    cross = (c - eta) * (c - eta.shift(1))      # prior-day mid-range (no lookahead)
    s2 = 4.0 * cross.rolling(20, min_periods=10).mean()
    ar_spread = np.sqrt(s2.clip(lower=0.0))
    ar_spread = ar_spread.clip(lower=0.0, upper=_SPREAD_CAP)
    ar_spread = ar_spread.replace([np.inf, -np.inf], np.nan)
    new_cols["mst_abdi_ranaldo_chl_20d"] = ar_spread

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
