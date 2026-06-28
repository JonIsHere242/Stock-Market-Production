"""
cgw_liquidity_reversion.py - Campbell-Grossman-Wang volume-conditioned reversal (Tier-2).

Campbell, Grossman & Wang (1993, QJE) "Trading Volume and Serial Correlation in Stock Returns".
Risk-averse liquidity suppliers demand a reverting price concession to absorb non-informational
(liquidity-driven) order flow, so a price move on HIGH volume reverses more than the same move on
low volume. We measure, per name, how its one-day serial-correlation payoff a_t = sign(r_{t-1})*r_t
differs between its own high- and low-volume days:

    gap = mean(a | high-vol) - mean(a | low-vol)      (< 0  => reverses on volume)
    signal = -sign(r_t) * tanh(gap / sd_a) * relvol_t (bounce expected after a high-vol move)

De-trapped per the review: relvol = Volume / 60d-median (winsorized) replaces the original hard
high-volume gate (which fired only on the high-|r| cohort = a volatility trap), and the payoff is
a SERIAL-CORRELATION quantity split by own-volume, so it is orthogonal to the return-distribution
(A) and cost-basis (B) pockets. Sign left for the model; vectorised via masked rolling means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 120
_VW = 60

METADATA = {
    "name":        "cgw_liquidity_reversion",
    "description": "Campbell-Grossman-Wang volume-conditioned reversal: gap between the one-day serial-correlation payoff sign(r_{t-1})*r_t on a name's high- vs low-volume days (< 0 => reverses on volume). De-gated (own-volume split, no hard high-vol gate). Orthogonal to return-distribution and cost-basis pockets.",
    "requires":    ["Close", "Volume"],
    "produces":    ["cgw_gap"],
    "tags":        ["mean_reversion", "volume", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 open-room build (Campbell-Grossman-Wang 1993)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    r = close.pct_change()

    a = np.sign(r.shift(1)) * r                              # serial-correlation payoff at t
    relmed = vol / vol.rolling(_VW, min_periods=20).median().replace(0, np.nan)
    hi = (relmed > 1.0).astype(float)
    lo = (relmed <= 1.0).astype(float)

    n_hi = hi.rolling(_W, min_periods=40).sum()
    n_lo = lo.rolling(_W, min_periods=40).sum()
    m_hi = (a * hi).rolling(_W, min_periods=40).sum() / n_hi.replace(0, np.nan)
    m_lo = (a * lo).rolling(_W, min_periods=40).sum() / n_lo.replace(0, np.nan)
    gap = m_hi - m_lo
    sd = a.rolling(_W, min_periods=40).std().replace(0, np.nan)

    df["cgw_gap"] = np.clip(gap.values, -0.05, 0.05)
    return df
