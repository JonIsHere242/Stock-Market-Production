"""
Theil index of dollar volume (inequality econometrics) -- per-ticker candidate.

Rolling Theil index of daily dollar volume (Close*Volume) over a 60-day window.
Theil index = mean( (x / x_bar) * ln(x / x_bar) ) where x_bar is the window mean.
It is an inequality/concentration measure that is upper-tail sensitive (large volume
spikes drive it up), complementing entropy-based features which are lower-tail sensitive.

Per-ticker proxy: computed on each stock's own dollar-volume time series; the cross-
sectional ranking interpretation is naturally lost, but the signal captures within-
stock volume concentration/inequality regimes that are predictive in their own right.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "xdom2_theil_volume",
    "description": (
        "Per-ticker rolling Theil index of dollar volume (Close*Volume). "
        "Measures within-stock volume inequality over a 60-day rolling window: "
        "mean( (x/xbar)*ln(x/xbar) ). Upper-tail volume spikes drive the index up. "
        "Also produces a 20-day fast variant and a slope (60d - 20d spread) as a "
        "regime-shift signal. Inherently cross-sectional method adapted as per-ticker proxy."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "xdom2_theil_volume_60",   # 60-day Theil index of dollar volume
        "xdom2_theil_volume_20",   # 20-day Theil index of dollar volume (fast)
        "xdom2_theil_volume_slope",  # slow minus fast (rising = increasing concentration)
    ],
    "tags": ["volume", "inequality", "cross-domain", "concentration", "upper-tail"],
    "version": "1.0",
    "author": "Theil index of dollar volume (inequality econometrics); Cross-domain / practitioner method transfer (batch 2)",
}


def _theil_index(arr: np.ndarray) -> float:
    """Compute Theil T index for a 1-D array of positive values."""
    # Filter out non-positive entries (shouldn't happen with real dollar-vol, but guard)
    x = arr[arr > 0]
    if len(x) < 2:
        return np.nan
    xbar = x.mean()
    if xbar <= 0:
        return np.nan
    ratio = x / xbar
    # Theil T = (1/N) * sum( ratio * ln(ratio) )
    # ratio * ln(ratio) is 0 when ratio -> 0 by convention, but ln(0) is -inf
    # guard: wherever ratio <= 0, contribution is 0 (already filtered, but be safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ratio > 0, ratio * np.log(ratio), 0.0).mean()
    return float(t)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- dollar volume series ---
    dolvol = (df["Close"] * df["Volume"]).values.astype(np.float64)

    n = len(dolvol)

    theil_60 = np.full(n, np.nan)
    theil_20 = np.full(n, np.nan)

    # Rolling Theil for window=60
    win60 = 60
    for i in range(win60 - 1, n):
        theil_60[i] = _theil_index(dolvol[i - win60 + 1 : i + 1])

    # Rolling Theil for window=20
    win20 = 20
    for i in range(win20 - 1, n):
        theil_20[i] = _theil_index(dolvol[i - win20 + 1 : i + 1])

    df["xdom2_theil_volume_60"] = theil_60
    df["xdom2_theil_volume_20"] = theil_20
    # Slope: slow-window minus fast-window;
    # positive => 60d concentration exceeds 20d => recent calm vs longer spike history
    # negative => 20d concentration exceeds 60d => recent spike regime
    df["xdom2_theil_volume_slope"] = theil_60 - theil_20

    return df
