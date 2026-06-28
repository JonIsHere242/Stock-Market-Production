"""
micro_amihud.py — Amihud (2002) illiquidity & its dynamics (Tier-2 liquidity).

Amihud (2002, JFM) "Illiquidity and stock returns: cross-section and time-series
effects". Daily price-impact proxy:

    ILLIQ_t = |return_t| / dollar_volume_t

i.e. how much the price moves per dollar traded. High ILLIQ = illiquid =
high expected return (illiquidity premium) and, more usefully for Tier-2, a
strong cross-sectional sorter near the tail.

We compute, all trailing-only and vectorised:
  - micro_amihud_21 / micro_amihud_63 : rolling MEAN of daily ILLIQ (scaled).
  - micro_amihud_ratio               : recent (21d) vs baseline (252d) ratio —
        a sudden illiquidity spike (drying-up of depth) often precedes outsized
        moves.
  - micro_amihud_asym                : up-day vs down-day illiquidity asymmetry,
        mean(ILLIQ | up days) - mean(ILLIQ | down days), normalised by their
        sum. Negative = it costs MORE to push the price DOWN than up (buying
        pressure / supportive depth), a bullish microstructure tilt.

Dollar volume uses Close*Volume. Returns use prior-close to today-close
(pct_change); all rolling, no future rows referenced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [21, 63]
_MIN = {21: 15, 63: 40}
_BASELINE = 252
_BASELINE_MIN = 150
_SCALE = 1e6  # scale tiny |ret|/$vol numbers into a readable range

METADATA = {
    "name":        "micro_amihud",
    "description": "Amihud (2002) illiquidity |ret|/dollar-volume: rolling levels (21d/63d), recent-vs-baseline ratio, and up-day/down-day illiquidity asymmetry.",
    "requires":    ["Close", "Volume"],
    "produces":    (
        [f"micro_amihud_{w}" for w in _WINDOWS]
        + ["micro_amihud_ratio", "micro_amihud_asym"]
    ),
    "tags":        ["liquidity", "microstructure", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Amihud 2002)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    ret = close.pct_change()
    dollar_vol = (close * volume).where(lambda x: x > 0)  # NaN where no $ traded

    illiq = (ret.abs() / dollar_vol) * _SCALE
    illiq = illiq.replace([np.inf, -np.inf], np.nan)

    for w in _WINDOWS:
        df[f"micro_amihud_{w}"] = illiq.rolling(w, min_periods=_MIN[w]).mean().values

    # recent vs long baseline
    recent = illiq.rolling(_WINDOWS[0], min_periods=_MIN[_WINDOWS[0]]).mean()
    base = illiq.rolling(_BASELINE, min_periods=_BASELINE_MIN).mean()
    df["micro_amihud_ratio"] = (
        (recent / base.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .clip(0.0, 20.0)
        .values
    )

    # up-day vs down-day illiquidity asymmetry over 63d
    up = ret > 0
    dn = ret < 0
    illiq_up = illiq.where(up)
    illiq_dn = illiq.where(dn)
    w = 63
    # each side is ~half-NaN (only up or only down days), so a 63-day window
    # holds ~30 valid obs per side; require at least 10 per side to populate.
    mp = 10
    mean_up = illiq_up.rolling(w, min_periods=mp).mean()
    mean_dn = illiq_dn.rolling(w, min_periods=mp).mean()
    denom = (mean_up + mean_dn).replace(0.0, np.nan)
    asym = (mean_up - mean_dn) / denom
    df["micro_amihud_asym"] = (
        asym.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).values
    )

    return df
