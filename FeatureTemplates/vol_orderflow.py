"""
vol_orderflow.py — Volume / order-flow proxies (Tier-2 confirmation).

Classic order-flow / volume-confirmation signals, all from OHLCV, all trailing,
all vectorised. The theme (Granville's OBV, and the volume-price confirmation
literature) is that a price move backed by volume is "real" and persists, while
a move on fading volume is weak and mean-reverts — a clean within-winner sorter.

  - vol_obv_slope_20/60 : On-Balance Volume (Granville 1963) = cumulative
        sign(return)*volume. We z-score-normalise its SLOPE (rolling linear
        trend per unit time, via the closed-form OLS slope on a fixed time
        index) and scale by recent volume so it is comparable cross-ticker.
        Rising OBV under rising price = accumulation.
  - vol_vwmom_20/60 : volume-weighted momentum = sum(return * volume) /
        sum(volume) over the window — momentum that upweights high-volume days.
  - vol_pv_divergence_20 : price-trend vs volume-trend divergence. +1*price_up
        but volume falling => weak rally (bearish divergence). Computed as
        sign(price slope) * (volume slope normalised). Negative => price up on
        DROPPING volume (weak), positive => move confirmed by volume.
  - vol_signret_corr_20/60 : rolling Pearson correlation between signed return
        and volume change. Positive => big moves come with volume surges
        (informed flow); near zero => noise.

OBV slope and the trend slopes use the analytic OLS-slope formula
cov(t, x)/var(t) over a rolling window (fully vectorised, no apply).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_WINDOWS = [20, 60]
_MIN = {20: 12, 60: 35}

METADATA = {
    "name":        "vol_orderflow",
    "description": "Order-flow / volume-confirmation proxies: OBV slope, volume-weighted momentum, price-volume trend divergence, and signed-return/volume correlation.",
    "requires":    ["Close", "Volume"],
    "produces":    (
        [f"vol_obv_slope_{w}" for w in _WINDOWS]
        + [f"vol_vwmom_{w}" for w in _WINDOWS]
        + ["vol_pv_divergence_20"]
        + [f"vol_signret_corr_{w}" for w in _WINDOWS]
    ),
    "tags":        ["volume", "order_flow", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (Granville OBV; volume-price confirmation)",
}


def _roll_slope(y: pd.Series, w: int, mp: int) -> pd.Series:
    """Rolling OLS slope of y against a 0..w-1 time index (vectorised).

    slope = cov(t, y) / var(t).  var(t) is constant for a fixed-width window,
    so we compute Sum((t - mean_t) * y) / Sum((t - mean_t)^2).
    """
    t = np.arange(w, dtype=float)
    t_mean = t.mean()
    tc = t - t_mean
    denom = (tc ** 2).sum()
    if denom == 0:
        return pd.Series(np.nan, index=y.index)
    # weighted rolling sum of tc*y: use rolling apply-free convolution via
    # rolling().apply is banned, so build with a rolling dot product through
    # numpy sliding window.
    vals = y.to_numpy(dtype=float)
    n = vals.shape[0]
    out = np.full(n, np.nan)
    if n >= w:
        # sliding windows view
        sw = np.lib.stride_tricks.sliding_window_view(vals, w)  # (n-w+1, w)
        # numerator per window = sum(tc * window)
        num = sw @ tc  # (n-w+1,)
        slope = num / denom
        # NaN guard: any window containing a NaN -> NaN
        bad = np.isnan(sw).any(axis=1)
        slope[bad] = np.nan
        out[w - 1:] = slope
    res = pd.Series(out, index=y.index)
    # enforce min_periods semantics (leading rows already NaN by construction)
    return res


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    ret = close.pct_change()
    sign = np.sign(ret).fillna(0.0)

    # ---------------- On-Balance Volume + normalised slope ------------------
    obv = (sign * volume).cumsum()
    for w in _WINDOWS:
        mp = _MIN[w]
        slope = _roll_slope(obv, w, mp)
        # normalise by typical daily volume so it is cross-ticker comparable
        vol_scale = volume.rolling(w, min_periods=mp).mean().replace(0.0, np.nan)
        df[f"vol_obv_slope_{w}"] = (
            (slope / vol_scale).replace([np.inf, -np.inf], np.nan).values
        )

    # ---------------- volume-weighted momentum ------------------------------
    rv = (ret * volume)
    for w in _WINDOWS:
        mp = _MIN[w]
        num = rv.rolling(w, min_periods=mp).sum()
        den = volume.rolling(w, min_periods=mp).sum().replace(0.0, np.nan)
        df[f"vol_vwmom_{w}"] = (
            (num / den).replace([np.inf, -np.inf], np.nan).values
        )

    # ---------------- price-volume trend divergence -------------------------
    w = 20
    mp = _MIN[w]
    price_slope = _roll_slope(close, w, mp)
    vol_slope = _roll_slope(volume, w, mp)
    # normalise each slope by its level so they are unit-free
    price_lvl = close.rolling(w, min_periods=mp).mean().replace(0.0, np.nan)
    vol_lvl = volume.rolling(w, min_periods=mp).mean().replace(0.0, np.nan)
    p_norm = price_slope / price_lvl
    v_norm = vol_slope / vol_lvl
    # divergence: sign of price trend * normalised volume trend
    # >0 : price up & volume rising (confirmed) OR price down & volume falling
    # <0 : price up & volume FALLING (weak rally) -> the bearish divergence
    div = np.sign(p_norm) * v_norm
    df["vol_pv_divergence_20"] = (
        div.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0).values
    )

    # ---------------- signed-return / volume-change correlation -------------
    dvol = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    for w in _WINDOWS:
        mp = _MIN[w]
        corr = ret.rolling(w, min_periods=mp).corr(dvol)
        df[f"vol_signret_corr_{w}"] = (
            corr.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).values
        )

    return df
