"""
_p625_range_vol_termstructure.py — Range-vol term-structure slope (candidate).

Idea: the *term structure* of realized range volatility. A short-window Yang-Zhang
vol vs a long-window Yang-Zhang vol, expressed as a dimensionless log-ratio, tells
you whether near-term vol is in contango (short > long, log-ratio > 0, vol picking
up) or backwardation (short < long, log-ratio < 0, vol calming). Plus the 5-day
change of that slope to capture *steepening velocity*.

Yang-Zhang per-bar variance pieces are exactly as in range_volatility.py:
    sigma_yz^2(w) = var_overnight + k_w * var_openclose + (1 - k_w) * mean(rs_d)
    k_w = 0.34 / (1.34 + (w+1)/(w-1))
all on a rolling window of length w, min_periods = w//2.

Slope (dimensionless, contango/backwardation):
    rvts_yz_slope_10_63    = log((yz(10)+1e-6)/(yz(63)+1e-6)).clip(-3, 3)
    rvts_yz_slope_21_126   = log((yz(21)+1e-6)/(yz(126)+1e-6)).clip(-3, 3)
Steepening velocity (5-day change of the 10/63 slope):
    rvts_yz_slope_chg5_10_63 = slope_t - slope_{t-5}

The long leg GATES the short: a slope is only emitted where the long-window YZ is
defined (i.e. where the long rolling window has reached min_periods), so a short-only
warmup value never leaks through as if the term structure were measurable.

Strictly causal (rolling/shift-positive only), no cross-sectional ops, no helpers.
Refs: Johnson (2017) JFQA; Bollerslev, Tauchen & Zhou (2009) RFS; Yang-Zhang (2000).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_range_vol_termstructure",
    "description": "Yang-Zhang realized-vol term-structure slope (short vs long log-ratio = contango/backwardation) plus 5d steepening velocity. Johnson (2017) JFQA; Bollerslev-Tauchen-Zhou (2009) RFS; Yang-Zhang (2000).",
    "requires":    [],
    "produces":    [
        "rvts_yz_slope_10_63",
        "rvts_yz_slope_21_126",
        "rvts_yz_slope_chg5_10_63",
    ],
    "tags":        ["volatility", "range", "term_structure", "experimental"],
    "version":     "1.0",
    "author":      "paper:Johnson (2017) JFQA; Bollerslev-Tauchen-Zhou (2009) RFS; Yang-Zhang (2000)",
}

_EPS = 1e-6
_CLIP = 3.0


def _yang_zhang_vol(o, h, l, c, c_prev, w: int) -> pd.Series:
    """Rolling Yang-Zhang volatility over window w (log-price inputs). Causal."""
    mp = max(w // 2, 2)
    co = c - o
    rs_d = (h - c) * (h - o) + (l - c) * (l - o)        # Rogers-Satchell per-bar
    overnight = (o - c_prev) ** 2                        # close-to-open
    openclose = co ** 2                                  # open-to-close

    var_o = overnight.rolling(w, min_periods=mp).var()
    var_c = openclose.rolling(w, min_periods=mp).var()
    rs_mean = rs_d.rolling(w, min_periods=mp).mean()

    k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))
    yz_var = var_o + k * var_c + (1.0 - k) * rs_mean
    return np.sqrt(yz_var.clip(lower=0))


def _slope(short_yz: pd.Series, long_yz: pd.Series) -> pd.Series:
    """Dimensionless log-ratio of short vs long YZ vol; long leg gates the short."""
    ratio = (short_yz + _EPS) / (long_yz + _EPS)
    slope = np.log(ratio.where(ratio > 0)).clip(-_CLIP, _CLIP)
    # Gate on the long leg: only measurable where long-window vol is defined.
    return slope.where(long_yz.notna())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Guard non-positive prices before taking logs (-> NaN, never inf).
    o = np.log(df["Open"].astype(float).where(df["Open"] > 0))
    h = np.log(df["High"].astype(float).where(df["High"] > 0))
    l = np.log(df["Low"].astype(float).where(df["Low"] > 0))
    c = np.log(df["Close"].astype(float).where(df["Close"] > 0))
    c_prev = c.shift(1)

    yz10 = _yang_zhang_vol(o, h, l, c, c_prev, 10)
    yz63 = _yang_zhang_vol(o, h, l, c, c_prev, 63)
    yz21 = _yang_zhang_vol(o, h, l, c, c_prev, 21)
    yz126 = _yang_zhang_vol(o, h, l, c, c_prev, 126)

    slope_10_63 = _slope(yz10, yz63)
    slope_21_126 = _slope(yz21, yz126)

    df["rvts_yz_slope_10_63"] = slope_10_63.to_numpy()
    df["rvts_yz_slope_21_126"] = slope_21_126.to_numpy()
    df["rvts_yz_slope_chg5_10_63"] = (slope_10_63 - slope_10_63.shift(5)).to_numpy()

    return df
