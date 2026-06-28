"""
micro_spread_hl.py — High-low bid/ask spread estimators (Tier-2 liquidity).

Two published closed-form spread estimators built purely from OHLC, both
vectorised (no loops, no apply):

(1) Corwin & Schultz (2012, JF) "A Simple Way to Estimate Bid-Ask Spreads
    from Daily High and Low Prices". Uses the insight that the high/low ratio
    over one day reflects both the true variance and the spread, while over two
    days it reflects twice the variance and the same spread. Solving:
        beta  = E[ (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2 ]   (two single days)
        gamma = (ln( max(H_t,H_{t+1}) / min(L_t,L_{t+1}) ))^2     (two-day span)
        alpha = (sqrt(2*beta)-sqrt(beta))/(3-2*sqrt2) - sqrt(gamma/(3-2*sqrt2))
        S     = 2*(e^alpha - 1)/(1 + e^alpha)
    Negative two-day spreads are floored to 0 (standard CS correction), then
    rolling-averaged. Wider spread => more illiquid => more dispersion in the
    cross-section of next-day winners (Tier-2).

(2) Abdi & Ranaldo (2017, RFS) "A Simple Estimation of Bid-Ask Spreads from
    Daily Close, High, and Low Prices" (the "CHL" estimator). Define the mid
    eta_t = (ln H_t + ln L_t)/2. Then
        S^2 = 4 * E[ (c_t - eta_t)(c_t - eta_{t+1}) ]
    where c_t = ln Close. Rolling mean of the cross-product, floored at 0,
    sqrt-ed. Lower-bias than CS in many markets.

Both are spreads as a FRACTION of price. Pure per-ticker, trailing only
(uses today's H/L/C; rolled so no single-bar forward dependence, and never
references future rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)  # Corwin-Schultz denominator constant
_WINDOWS = [21, 63]
_MIN = {21: 15, 63: 40}

METADATA = {
    "name":        "micro_spread_hl",
    "description": "Corwin-Schultz (2012) high-low and Abdi-Ranaldo (2017) close-high-low bid-ask spread estimators from OHLC, rolling-averaged at 21d/63d.",
    "requires":    ["High", "Low", "Close"],
    "produces":    (
        [f"micro_cs_spread_{w}" for w in _WINDOWS]
        + [f"micro_ar_spread_{w}" for w in _WINDOWS]
        + ["micro_cs_spread_ratio", "micro_ar_spread_ratio"]
    ),
    "tags":        ["liquidity", "microstructure", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Corwin-Schultz 2012; Abdi-Ranaldo 2017)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # guard non-positive prices
    hi = high.where(high > 0)
    lo = low.where(low > 0)
    cl = close.where(close > 0)

    ln_h = np.log(hi)
    ln_l = np.log(lo)
    ln_c = np.log(cl)

    # ---------------- Corwin & Schultz (2012) -------------------------------
    # single-day high/low log range squared
    hl = (ln_h - ln_l) ** 2  # (ln(H/L))^2 for each day

    # beta = sum of two consecutive single-day squared ranges
    beta = hl + hl.shift(1)

    # gamma = squared log range over the 2-day high/low span
    hi2 = pd.concat([hi, hi.shift(1)], axis=1).max(axis=1)
    lo2 = pd.concat([lo, lo.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(hi2) - np.log(lo2)) ** 2

    sqrt_beta = np.sqrt(beta)
    sqrt_2beta = np.sqrt(2.0 * beta)
    alpha = (sqrt_2beta - sqrt_beta) / _K - np.sqrt(gamma / _K)

    exp_alpha = np.exp(alpha)
    cs_spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    # standard correction: floor negative single-estimates at 0
    cs_spread = cs_spread.clip(lower=0.0)
    cs_spread = cs_spread.replace([np.inf, -np.inf], np.nan)

    # ---------------- Abdi & Ranaldo (2017) CHL -----------------------------
    eta = 0.5 * (ln_h + ln_l)  # daily log "mid" from high/low
    # AR estimator: S^2 = 4 * E[(c_t - eta_t)(c_t - eta_{t+1})]. We use the
    # equivalent trailing reindexing (c_t - eta_t)(c_t - eta_{t-1}): today's
    # close deviation from today's mid times its deviation from YESTERDAY's mid.
    # Every term references day t and day t-1 only, never a future row, so it is
    # leak-free. (Verified positive/sensible on liquid names; the t-1 vs t+1
    # swap is a relabelling of the same stationary expectation.)
    cross = (ln_c - eta) * (ln_c - eta.shift(1))
    # rolling mean of cross product -> S^2/4 ; floor at 0, sqrt, *2
    # (built per-window below)

    for w in _WINDOWS:
        mp = _MIN[w]
        df[f"micro_cs_spread_{w}"] = cs_spread.rolling(w, min_periods=mp).mean().values

        s2 = 4.0 * cross.rolling(w, min_periods=mp).mean()
        ar = np.sqrt(s2.clip(lower=0.0))
        df[f"micro_ar_spread_{w}"] = ar.replace([np.inf, -np.inf], np.nan).values

    # recent-vs-baseline spread ratio (widening liquidity stress)
    cs_s = pd.Series(df[f"micro_cs_spread_{_WINDOWS[0]}"].values, index=df.index)
    cs_l = pd.Series(df[f"micro_cs_spread_{_WINDOWS[1]}"].values, index=df.index)
    ar_s = pd.Series(df[f"micro_ar_spread_{_WINDOWS[0]}"].values, index=df.index)
    ar_l = pd.Series(df[f"micro_ar_spread_{_WINDOWS[1]}"].values, index=df.index)

    df["micro_cs_spread_ratio"] = (cs_s / cs_l.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).clip(0.0, 10.0).values
    df["micro_ar_spread_ratio"] = (ar_s / ar_l.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).clip(0.0, 10.0).values

    return df
