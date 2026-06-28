"""
size_liquidity_turnover.py  --  Turnover and trading activity RELATIVE to firm size.

Theme: dollar volume is mechanically larger for large firms, so raw dollar-volume features (the
model already has several) are mostly a size proxy. Dividing dollar volume by market cap gives
TURNOVER -- the fraction of the company that changes hands per day -- which is comparable across
sizes and is a classic predictor (Datar-Naik-Radcliffe turnover effect; high turnover -> lower
future returns / attention crowding). We add multi-horizon turnover, its trend and acceleration,
and a size-adjusted volume-shock that is orthogonal to plain volume z-scores.

Lookahead-safe: cap on day d known at d's close; dollar volume = Close*Volume known at d's close.

Columns (all prefixed szc_):
  szc_turnover_5              mean daily turnover (dollar-vol / cap) over 5d
  szc_turnover_20             20d mean daily turnover
  szc_turnover_60             60d mean daily turnover
  szc_turnover_z_60           z-score of 5d turnover vs its own 60d history (turnover spike)
  szc_turnover_trend_60       turnover slope: log(20d turnover / 60d turnover)
  szc_turnover_accel          turnover acceleration: 5d-vs-20d minus 20d-vs-60d ratios
  szc_capadj_vol_shock_20     today's dollar-vol / cap, z-scored over 20d (size-normalized shock)
  szc_float_stability_60      stability (1/CV) of turnover over 60d -- steady vs erratic participation
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_spec = _ilu.spec_from_file_location("_marketcap", _Path(__file__).resolve().parent / "_marketcap.py")
_marketcap = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_marketcap)

METADATA = {
    "name":        "size_liquidity_turnover",
    "description": (
        "Cap-relative turnover (dollar volume / market cap) over 5/20/60d, plus turnover spike "
        "z-score, trend, acceleration, a size-adjusted volume shock, and turnover stability."
    ),
    "requires":    ["Date", "Ticker", "Close", "Volume"],
    "produces":    [
        "szc_turnover_5",
        "szc_turnover_20",
        "szc_turnover_60",
        "szc_turnover_z_60",
        "szc_turnover_trend_60",
        "szc_turnover_accel",
        "szc_capadj_vol_shock_20",
        "szc_float_stability_60",
    ],
    "tags":        ["size", "liquidity", "turnover", "volume"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    had_mcap = "mcap" in df.columns
    df = _marketcap.as_of_cap(df)

    mcap = pd.to_numeric(df["mcap"], errors="coerce").where(lambda x: x > 0)
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").clip(lower=0)

    # Daily turnover = traded dollars / market cap = fraction of company changing hands.
    dollar_vol = (close * volume).clip(lower=0)
    daily_turnover = (dollar_vol / mcap).clip(0, 100)  # clip absurd (cap-data gaps)

    t5 = daily_turnover.rolling(5, min_periods=3).mean()
    t20 = daily_turnover.rolling(20, min_periods=10).mean()
    t60 = daily_turnover.rolling(60, min_periods=30).mean()

    df["szc_turnover_5"] = t5.clip(0, 100).to_numpy()
    df["szc_turnover_20"] = t20.clip(0, 100).to_numpy()
    df["szc_turnover_60"] = t60.clip(0, 100).to_numpy()

    # Turnover spike: short-window turnover vs its own 60d distribution.
    t_mean = daily_turnover.rolling(60, min_periods=30).mean()
    t_std = daily_turnover.rolling(60, min_periods=30).std().replace(0.0, np.nan)
    df["szc_turnover_z_60"] = ((t5 - t_mean) / t_std).clip(-10, 10).to_numpy()

    # Turnover trend: log ratio of medium-term to long-term turnover (rising attention if >0).
    trend = np.log((t20 + 1e-12) / (t60 + 1e-12))
    df["szc_turnover_trend_60"] = trend.clip(-10, 10).to_numpy()

    # Turnover acceleration: how fast the trend itself is changing (2nd difference of ratios).
    fast = np.log((t5 + 1e-12) / (t20 + 1e-12))
    slow = np.log((t20 + 1e-12) / (t60 + 1e-12))
    df["szc_turnover_accel"] = (fast - slow).clip(-10, 10).to_numpy()

    # Size-adjusted single-day volume shock: today's turnover z-scored over 20d
    # (orthogonal to raw volume z because it is normalized by CAP, not by volume).
    m20 = daily_turnover.rolling(20, min_periods=10).mean()
    s20 = daily_turnover.rolling(20, min_periods=10).std().replace(0.0, np.nan)
    df["szc_capadj_vol_shock_20"] = ((daily_turnover - m20) / s20).clip(-10, 10).to_numpy()

    # Float / participation stability: inverse coefficient of variation of turnover over 60d.
    # High = steady, predictable participation; low = erratic (event-driven) participation.
    cv = (t_std / t_mean).replace([np.inf, -np.inf], np.nan)
    df["szc_float_stability_60"] = (1.0 / (cv + 1e-6)).clip(0, 50).to_numpy()

    if not had_mcap and "mcap" in df.columns:
        df = df.drop(columns=["mcap"])

    return df
