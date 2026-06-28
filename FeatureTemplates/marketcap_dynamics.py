"""
marketcap_dynamics.py  --  Within-ticker market-cap level, momentum, and cap-vs-price divergence.

Theme: SIZE dynamics that are orthogonal to plain price momentum/volatility. Everything is
per-ticker (NO cross-sectional decile -- that is forbidden here). The interesting, model-novel
signal is the SHARE-COUNT channel: cap moves are price moves PLUS share-count moves (buybacks,
secondary issuance, SBC dilution). Isolating "cap growth minus price return" gives a clean
proxy for net share-count change that the price-only OHLCV features cannot see.

Lookahead-safe: cap on day d is known at the day-d close (price_d x shares_d); we backward-asof.

Columns (all prefixed szc_):
  szc_log_mcap                 natural log of market cap (level, slowly varying)
  szc_mcap_z_60                within-ticker z-score of log-cap over 60d
  szc_mcap_z_120               within-ticker z-score of log-cap over 120d
  szc_mcap_chg_5               5-day log-cap change (cap momentum)
  szc_mcap_chg_20              20-day log-cap change
  szc_mcap_chg_60              60-day log-cap change
  szc_sharecount_chg_20        20d cap log-change MINUS 20d price log-return = dilution/buyback proxy
  szc_sharecount_chg_60        60d version
  szc_dilution_trend_60        sign-aware persistence of the share-count drift over 60d
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---- shared PIT market-cap helper (imported by file path) -----------------
_spec = _ilu.spec_from_file_location("_marketcap", _Path(__file__).resolve().parent / "_marketcap.py")
_marketcap = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_marketcap)

METADATA = {
    "name":        "marketcap_dynamics",
    "description": (
        "Within-ticker log market-cap level, multi-horizon cap momentum, within-ticker cap "
        "z-scores, and a cap-growth-minus-price-return share-count/dilution proxy."
    ),
    "requires":    ["Date", "Ticker", "Close"],
    "produces":    [
        "szc_log_mcap",
        "szc_mcap_z_60",
        "szc_mcap_z_120",
        "szc_mcap_chg_5",
        "szc_mcap_chg_20",
        "szc_mcap_chg_60",
        "szc_sharecount_chg_20",
        "szc_sharecount_chg_60",
        "szc_dilution_trend_60",
    ],
    "tags":        ["size", "market_cap", "dilution", "fundamental"],
    "version":     "1.0",
    "author":      "feature-gen",
}

_CLIP = 1e6  # generous clip for log-change products


def _zscore(s: pd.Series, win: int, mp: int) -> pd.Series:
    mean = s.rolling(win, min_periods=mp).mean()
    std = s.rolling(win, min_periods=mp).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.clip(-10, 10)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Attach PIT market cap as a temporary 'mcap' column, then drop it before returning.
    had_mcap = "mcap" in df.columns
    df = _marketcap.as_of_cap(df)

    mcap = pd.to_numeric(df["mcap"], errors="coerce")
    # log cap; non-positive/NaN caps -> NaN (helper already filters <=0, but be safe)
    log_mcap = np.log(mcap.where(mcap > 0))

    df["szc_log_mcap"] = log_mcap.clip(0, 40).to_numpy()

    # Within-ticker level z-scores (relative cheapness/expensiveness of the firm's OWN size)
    df["szc_mcap_z_60"] = _zscore(log_mcap, 60, 30).to_numpy()
    df["szc_mcap_z_120"] = _zscore(log_mcap, 120, 60).to_numpy()

    # Cap momentum = multi-horizon log-cap change
    chg5 = (log_mcap - log_mcap.shift(5)).clip(-_CLIP, _CLIP)
    chg20 = (log_mcap - log_mcap.shift(20)).clip(-_CLIP, _CLIP)
    chg60 = (log_mcap - log_mcap.shift(60)).clip(-_CLIP, _CLIP)
    df["szc_mcap_chg_5"] = chg5.to_numpy()
    df["szc_mcap_chg_20"] = chg20.to_numpy()
    df["szc_mcap_chg_60"] = chg60.to_numpy()

    # Share-count / dilution proxy: cap log-change MINUS price log-return over the same horizon.
    # If cap grew faster than price -> shares issued (dilution / secondary); if slower -> buyback.
    log_close = np.log(pd.to_numeric(df["Close"], errors="coerce").where(lambda x: x > 0))
    pret20 = (log_close - log_close.shift(20))
    pret60 = (log_close - log_close.shift(60))
    sc20 = (chg20 - pret20).clip(-5, 5)
    sc60 = (chg60 - pret60).clip(-5, 5)
    df["szc_sharecount_chg_20"] = sc20.to_numpy()
    df["szc_sharecount_chg_60"] = sc60.to_numpy()

    # Dilution trend: persistence of the daily share-count drift over 60d.
    # daily share-count drift = daily log-cap change - daily log-return
    daily_sc = (log_mcap.diff() - log_close.diff())
    # mean drift scaled by its own volatility -> a t-stat-like steadiness score
    drift_mean = daily_sc.rolling(60, min_periods=30).mean()
    drift_std = daily_sc.rolling(60, min_periods=30).std().replace(0.0, np.nan)
    df["szc_dilution_trend_60"] = (drift_mean / drift_std).clip(-10, 10).to_numpy()

    # Clean up the temporary mcap column unless the caller already had one.
    if not had_mcap and "mcap" in df.columns:
        df = df.drop(columns=["mcap"])

    return df
