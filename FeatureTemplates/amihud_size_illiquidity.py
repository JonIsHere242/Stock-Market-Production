"""
amihud_size_illiquidity.py  --  Amihud price-impact illiquidity, size-aware and self-relative.

Theme: the Amihud (2002) illiquidity ratio = average of |daily return| / daily dollar volume,
i.e. the price move per dollar traded. Higher = more fragile / illiquid -> illiquidity premium.
The model already has raw amihud (ilq_amihud_*). What it does NOT have is:
  (1) Amihud scaled by market cap (impact-per-dollar is mechanically size-dependent; cap-scaling
      makes the ratio comparable across the firm's own size regime and across the universe),
  (2) Amihud RELATIVE to the ticker's own 60d median (an illiquidity REGIME-SHIFT signal that is
      stationary per-ticker and orthogonal to the level), and
  (3) an asymmetry: is illiquidity worse on down days than up days (fragility under selling)?

Lookahead-safe: |return| and dollar volume are both known at the day's close; cap known at close.

Columns (all prefixed szc_):
  szc_amihud_capadj_21        21d mean of |ret| / dollar-vol, x cap  (size-neutralized impact)
  szc_amihud_capadj_63        63d version
  szc_amihud_rel_median_60    21d amihud / its own 60d median  (illiquidity regime shift)
  szc_amihud_log_21           log of raw 21d amihud (heavy-tailed; log stabilizes)
  szc_amihud_trend_63         log(21d amihud / 63d amihud)  -- drying-up vs deepening liquidity
  szc_amihud_down_up_asym_60  illiquidity on down days minus up days, scaled  (fragility asymmetry)
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
    "name":        "amihud_size_illiquidity",
    "description": (
        "Amihud |return|/dollar-volume illiquidity, cap-scaled (size-neutralized), relative to "
        "its own 60d median (regime shift), log-stabilized, trend, and down-vs-up asymmetry."
    ),
    "requires":    ["Date", "Ticker", "Close", "Volume"],
    "produces":    [
        "szc_amihud_capadj_21",
        "szc_amihud_capadj_63",
        "szc_amihud_rel_median_60",
        "szc_amihud_log_21",
        "szc_amihud_trend_63",
        "szc_amihud_down_up_asym_60",
    ],
    "tags":        ["liquidity", "size", "amihud", "illiquidity"],
    "version":     "1.0",
    "author":      "feature-gen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    had_mcap = "mcap" in df.columns
    df = _marketcap.as_of_cap(df)

    mcap = pd.to_numeric(df["mcap"], errors="coerce").where(lambda x: x > 0)
    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").clip(lower=0)

    ret = close.pct_change()
    dollar_vol = (close * volume)
    # Daily Amihud (price move per dollar traded). Guard zero-volume days.
    daily_amihud = (ret.abs() / dollar_vol.replace(0.0, np.nan))
    daily_amihud = daily_amihud.replace([np.inf, -np.inf], np.nan)

    a21 = daily_amihud.rolling(21, min_periods=10).mean()
    a63 = daily_amihud.rolling(63, min_periods=30).mean()

    # (1) Cap-adjusted: multiply by cap so the impact-per-dollar is normalized to a common
    # size scale. Raw amihud is ~1e-12 for mega-caps; xcap brings it to an O(1)-ish, comparable
    # quantity. Then log for the heavy tail.
    capadj21 = (a21 * mcap)
    capadj63 = (a63 * mcap)
    df["szc_amihud_capadj_21"] = np.log1p(capadj21.clip(lower=0)).clip(0, 50).to_numpy()
    df["szc_amihud_capadj_63"] = np.log1p(capadj63.clip(lower=0)).clip(0, 50).to_numpy()

    # (2) Self-relative regime shift: current 21d amihud vs its own 60d median of daily amihud.
    med60 = daily_amihud.rolling(60, min_periods=30).median().replace(0.0, np.nan)
    df["szc_amihud_rel_median_60"] = (a21 / med60).clip(0, 50).to_numpy()

    # (3) Log-stabilized raw level (heavy-tailed) -- a per-ticker illiquidity level signal.
    df["szc_amihud_log_21"] = np.log(a21.where(a21 > 0)).clip(-60, 10).to_numpy()

    # (4) Trend: is liquidity drying up (amihud rising) or deepening?
    trend = np.log((a21 + 1e-30) / (a63 + 1e-30))
    df["szc_amihud_trend_63"] = trend.clip(-20, 20).to_numpy()

    # (5) Down/up asymmetry: mean daily amihud on down days minus up days over 60d, scaled by
    # the overall 60d mean. Positive = price impact is worse when selling (fragile to downside).
    down_imp = daily_amihud.where(ret < 0)
    up_imp = daily_amihud.where(ret > 0)
    down_mean = down_imp.rolling(60, min_periods=15).mean()
    up_mean = up_imp.rolling(60, min_periods=15).mean()
    scale = daily_amihud.rolling(60, min_periods=30).mean().replace(0.0, np.nan)
    asym = (down_mean - up_mean) / scale
    df["szc_amihud_down_up_asym_60"] = asym.replace([np.inf, -np.inf], np.nan).clip(-20, 20).to_numpy()

    if not had_mcap and "mcap" in df.columns:
        df = df.drop(columns=["mcap"])

    return df
