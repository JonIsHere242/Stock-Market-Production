"""
Yang-Zhang volatility estimator (2000) — xdom2_yang_zhang
Minimum-variance OHLC volatility estimator robust to drift and overnight jumps.

Rolling 20-day Yang-Zhang vol decomposes realized variance into:
  - overnight (close-to-open) variance
  - open-to-close variance (Rogers-Satchell component and open-to-close return component)
combined with weights that minimise total variance under geometric Brownian motion.

Also produces yz_parkinson_ratio_20 = YZ vol / Parkinson vol, which captures
the relative contribution of overnight gap risk vs. intraday range — a signal
for gap-prone names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_yang_zhang",
    "description": (
        "Rolling 20-day Yang-Zhang OHLC volatility estimator (annualised). "
        "Combines overnight variance, open-to-close variance, and Rogers-Satchell "
        "variance with minimum-variance weights, making it robust to both price drift "
        "and overnight jumps — unlike close-to-close or Parkinson estimators. "
        "xdom2_yang_zhang_vol20: annualised YZ vol (20-day rolling). "
        "xdom2_yang_zhang_park_ratio20: YZ vol divided by Parkinson vol (gap-risk ratio); "
        "values >1 indicate overnight gaps dominate intraday range. "
        "xdom2_yang_zhang_vol20_slope: 5-day log change in YZ vol (vol-of-vol / trend). "
        "Per-ticker time-series estimator — no cross-sectional dependency."
    ),
    "requires": ["Open", "High", "Low", "Close"],
    "produces": [
        "xdom2_yang_zhang_vol20",
        "xdom2_yang_zhang_park_ratio20",
        "xdom2_yang_zhang_vol20_slope",
    ],
    "tags": ["volatility", "ohlc", "yang-zhang", "gap-risk", "cross-domain"],
    "version": "1.0",
    "author": "Yang & Zhang (2000) 'Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices' — implemented as per-ticker rolling estimator",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # Log prices
    # ------------------------------------------------------------------ #
    log_o = np.log(df["Open"].replace(0, np.nan))
    log_h = np.log(df["High"].replace(0, np.nan))
    log_l = np.log(df["Low"].replace(0, np.nan))
    log_c = np.log(df["Close"].replace(0, np.nan))

    # Overnight return: log(Open_t / Close_{t-1})
    log_c_prev = log_c.shift(1)
    overnight = log_o - log_c_prev           # u in Yang-Zhang notation

    # Open-to-close return: log(Close_t / Open_t)
    oc_ret = log_c - log_o                   # d in Yang-Zhang notation

    # Rogers-Satchell variance (per bar) — drift-independent intraday estimator
    # RS = log(H/C)*log(H/O) + log(L/C)*log(L/O)
    rs = (log_h - log_c) * (log_h - log_o) + (log_l - log_c) * (log_l - log_o)
    # RS can be negative due to numerical noise; clip at 0
    rs = rs.clip(lower=0)

    # ------------------------------------------------------------------ #
    # Yang-Zhang 20-day rolling estimator
    # ------------------------------------------------------------------ #
    N = 20

    # Overnight variance: sample variance of overnight returns
    # (mean-corrected to remove drift)
    ov_var = overnight.rolling(N).var()          # pandas var uses ddof=1

    # Open-to-close variance: sample variance of OC returns
    oc_var = oc_ret.rolling(N).var()

    # Rogers-Satchell mean (not variance — RS is already a squared quantity)
    rs_mean = rs.rolling(N).mean()

    # Yang-Zhang optimal weight k for Rogers-Satchell component
    # k = 0.34 / (1.34 + (N+1)/(N-1))   — from Yang & Zhang (2000) eq. 29
    k = 0.34 / (1.34 + (N + 1) / max(N - 1, 1))

    # YZ variance estimate (daily)
    yz_var = ov_var + k * oc_var + (1.0 - k) * rs_mean

    # Annualise: multiply daily variance by 252, then sqrt
    yz_vol = np.sqrt(yz_var * 252).replace([np.inf, -np.inf], np.nan)
    df["xdom2_yang_zhang_vol20"] = yz_vol

    # ------------------------------------------------------------------ #
    # Parkinson 20-day rolling estimator for the ratio
    # Parkinson = sqrt( (1/(4*ln2)) * mean( (log(H/L))^2 ) * 252 )
    # ------------------------------------------------------------------ #
    hl_sq = (log_h - log_l) ** 2
    park_var_daily = hl_sq.rolling(N).mean() / (4.0 * np.log(2))
    park_vol = np.sqrt(park_var_daily * 252).replace([np.inf, -np.inf], np.nan)

    # Gap-risk ratio: YZ / Parkinson
    # YZ absorbs overnight gaps; Parkinson uses only H-L range.
    # Ratio > 1 means overnight gaps add variance beyond what intraday range captures.
    park_safe = park_vol.where(park_vol > 0, np.nan)
    df["xdom2_yang_zhang_park_ratio20"] = (yz_vol / park_safe).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------ #
    # Vol slope: 5-day log change in YZ vol (momentum/trend of volatility)
    # ------------------------------------------------------------------ #
    yz_vol_safe = yz_vol.where(yz_vol > 0, np.nan)
    df["xdom2_yang_zhang_vol20_slope"] = np.log(yz_vol_safe / yz_vol_safe.shift(5)).replace(
        [np.inf, -np.inf], np.nan
    )

    return df
