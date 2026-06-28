"""
Adaptive Conformal Interval Width Features
Paper: "SPACR: Single-Pass Adaptive Training of Uncertainty-Aware Conformal Regressors"
arXiv: 2606.10734

SPACR is a neural network training method that jointly optimizes prediction interval
efficiency (width) and validity across multiple confidence levels without batch-splitting.
Its key insight: tighter intervals at EQUAL COVERAGE signal LOWER uncertainty, and the
efficiency of the interval (coverage / width) is the key regime metric.

The paper has no directly computable OHLCV method (pure NN architecture).  SKIPPED.

INSTEAD: We implement a SPACR-inspired OHLCV feature family that captures the SAME CORE
IDEAS — interval width, coverage efficiency, and INTRADAY RESOLUTION of uncertainty.

Key design insight:
  A daily OHLCV bar is itself a conformal interval: [Low, High] is the realized
  interval for that day's price path.  Where Close falls within [Low, High]
  (the "bar position") tells us how uncertainty was RESOLVED — bullish (Close near High)
  or bearish (Close near Low).  The efficiency of uncertainty resolution, how much
  of the bar's range the Close traversed, and how this compares to rolling history,
  are SPACR's efficiency concept applied directly to intraday price structure.

Features (8 columns prefixed "aiw_"):
  aiw_bar_pos_z20        Z-score of (C-L)/(H-L) vs 20d rolling history
  aiw_bar_pos_z60        Same, 60d window
  aiw_gk_vol_z20         Garman-Klass vol estimator z-score (20d)
  aiw_gk_vol_ratio       GK vol ratio: 5d vs 20d (regime shift direction)
  aiw_interval_eff_20    Interval efficiency: bar_pos vs interval width (GK vol)
  aiw_dev_typical_z20    Z-score of (Close - typical_price) / Close; typical=(H+L+C+O)/4
  aiw_vol_surprise       HL-range vs expected rolling HL range (SPACR coverage error)
  aiw_width_accel        Second derivative of GK vol (acceleration of uncertainty)
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_10734_adaptive_interval_width",
    "description": (
        "Adaptive interval efficiency features inspired by arXiv:2606.10734 (SPACR). "
        "Treats each OHLCV bar as a realized conformal interval: bar position z-score, "
        "Garman-Klass volatility regime, interval efficiency, close-vs-typical deviation, "
        "and volatility acceleration capture intraday uncertainty resolution quality."
    ),
    "requires": ["Open", "Close", "High", "Low", "Volume"],
    "produces": [
        "aiw_bar_pos_z20",
        "aiw_bar_pos_z60",
        "aiw_gk_vol_z20",
        "aiw_gk_vol_ratio",
        "aiw_interval_eff_20",
        "aiw_dev_typical_z20",
        "aiw_vol_surprise",
        "aiw_width_accel",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.10734",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)
    O = df["Open"].astype(float)

    # ── Bar position: (Close - Low) / (High - Low) — where did price close in range?
    hl = (H - L).replace(0, np.nan)
    bar_pos = (C - L) / hl  # [0,1]; 1=closed at high (bullish resolution)

    # Z-score vs rolling history (SPACR: deviation from expected efficiency)
    def _z(s, w, mp=None):
        mp = mp or max(5, w // 4)
        mu  = s.rolling(w, min_periods=mp).mean()
        std = s.rolling(w, min_periods=mp).std()
        return (s - mu) / std.replace(0, np.nan)

    df["aiw_bar_pos_z20"] = _z(bar_pos, 20)
    df["aiw_bar_pos_z60"] = _z(bar_pos, 60)

    # ── Garman-Klass volatility estimator (uses O, H, L, C)
    # GK = sqrt(0.5 * ln(H/L)^2 - (2*ln2-1) * ln(C/O)^2)
    log_hl = np.log((H / L.replace(0, np.nan)).replace(0, np.nan))
    log_co = np.log((C / O.replace(0, np.nan)).replace(0, np.nan))
    gk_sq = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk_vol = np.sqrt(gk_sq.clip(lower=0))

    gk_s = gk_vol.copy()
    df["aiw_gk_vol_z20"] = _z(gk_s, 20)

    # Ratio of 5d vs 20d GK vol — recent vol expansion/contraction
    gk_5  = gk_s.rolling(5, min_periods=2).mean()
    gk_20 = gk_s.rolling(20, min_periods=8).mean()
    df["aiw_gk_vol_ratio"] = gk_5 / gk_20.replace(0, np.nan)

    # ── Interval efficiency: bar_pos / GK vol (high = closed bullishly in low-vol day)
    # SPACR's efficiency = coverage / width; here bar_pos ~ coverage direction, GK vol ~ width
    eff = bar_pos / gk_vol.replace(0, np.nan)
    eff_med = eff.rolling(20, min_periods=8).median()
    df["aiw_interval_eff_20"] = (eff - eff_med) / eff.rolling(20, min_periods=8).std().replace(0, np.nan)

    # ── Close vs typical price deviation (OHLCV version of conformity score)
    # typical = (O + H + L + C) / 4; dev = (C - typical) / C
    typical = (O + H + L + C) / 4.0
    dev_typical = (C - typical) / C.replace(0, np.nan)
    df["aiw_dev_typical_z20"] = _z(dev_typical, 20)

    # ── Volume surprise: realized HL-range vs rolling expected (like SPACR coverage error)
    # Positive surprise = the bar was wider than expected (uncertainty underestimated)
    hl_abs = (H - L).astype(float)
    expected_hl = hl_abs.rolling(10, min_periods=4).mean().shift(1)
    df["aiw_vol_surprise"] = (hl_abs - expected_hl) / expected_hl.replace(0, np.nan)

    # ── Width acceleration: 2nd derivative of GK vol (is volatility increasing faster?)
    gk_d1 = gk_s.diff(3)     # first derivative (3d)
    gk_d2 = gk_d1.diff(3)    # second derivative
    df["aiw_width_accel"] = _z(gk_d2, 20)

    return df
