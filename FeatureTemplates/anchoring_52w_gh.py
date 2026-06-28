"""
anchoring_52w_gh.py — 52-week-high anchoring (George & Hwang 2004, JF).

George & Hwang (2004) "The 52-Week High and Momentum Investing" (Journal of
Finance). The single best predictor of future returns is NOT past return but
NEARNESS TO THE 52-WEEK HIGH:

    GH ratio = Close / (rolling 252-day High)

Stocks trading near their 52w high keep winning; investors anchor on the salient
high and under-react to good news that would push price through it. This is a
pure within-cross-section sorter — exactly Tier-2 top-decile discrimination.

We add, per ticker (all trailing-only):
  anch_gh_ratio_252      Close / max(High, 252)              (George-Hwang, in (0,1])
  anch_low_ratio_252     (Close - min(Low,252)) / (max-min)  (proximity to 52w LOW)
  anch_nearness_asym     gh_ratio - (1 - low_ratio): asymmetry of how close the
                         price sits to its high vs its low (>0 = hugging the high)
  anch_gh_ratio_126      same GH ratio on a 6-month (126d) window (faster anchor)
  anch_new_high_60d      fraction of last 60 days that were 252d-window new highs
                         (persistence of breakout pressure)

Vectorised, per-ticker, no leakage (rolling max/min use only trailing bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_W = 252
_W_MIN = 200
_W6 = 126
_W6_MIN = 100

METADATA = {
    "name":        "anchoring_52w_gh",
    "description": "52-week-high anchoring per George-Hwang 2004: Close/252d-High ratio, distance to 52w low, high/low nearness asymmetry, 126d anchor, and new-high persistence.",
    "requires":    ["High", "Low", "Close"],
    "produces":    [
        "anch_gh_ratio_252",
        "anch_low_ratio_252",
        "anch_nearness_asym",
        "anch_gh_ratio_126",
        "anch_new_high_60d",
    ],
    "tags":        ["momentum", "trend", "tail", "anchoring", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (George & Hwang 2004)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    roll_high_252 = high.rolling(_W, min_periods=_W_MIN).max()
    roll_low_252 = low.rolling(_W, min_periods=_W_MIN).min()
    roll_high_126 = high.rolling(_W6, min_periods=_W6_MIN).max()

    # George-Hwang 52w-high ratio: Close / 252d high. In (0, 1]; near 1 = at high.
    gh = close / roll_high_252.replace(0.0, np.nan)
    df["anch_gh_ratio_252"] = gh.clip(0.0, 2.0).values

    # Position within the 52w high-low range: 0 = at low, 1 = at high.
    rng = (roll_high_252 - roll_low_252).replace(0.0, np.nan)
    low_ratio = (close - roll_low_252) / rng
    df["anch_low_ratio_252"] = low_ratio.clip(-1.0, 2.0).values

    # Nearness asymmetry: how much closer to the high than to the low.
    # gh ~ proximity to high; (1 - low_ratio) ~ remaining room down to the low.
    df["anch_nearness_asym"] = (gh - (1.0 - low_ratio)).clip(-2.0, 2.0).values

    # Faster 6-month anchor.
    gh6 = close / roll_high_126.replace(0.0, np.nan)
    df["anch_gh_ratio_126"] = gh6.clip(0.0, 2.0).values

    # New-high persistence: fraction of last 60 days that set a 252d-window high.
    # A bar is a "new high" when its High equals the trailing 252d rolling max
    # (which, being trailing, already includes that bar — so == is the new high).
    is_new_high = (high >= roll_high_252).astype(float)
    df["anch_new_high_60d"] = is_new_high.rolling(60, min_periods=30).mean().values

    return df
