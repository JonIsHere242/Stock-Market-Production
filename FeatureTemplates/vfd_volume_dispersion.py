"""
vfd_volume_dispersion.py — Volume CONCENTRATION, coupling, and trend curvature.

Theme: the *shape* of the volume distribution over a window and how volume
co-moves with price — orthogonal to the level/momentum features the model
already has (raw `volume_entropy`, dollar-volume momentum, persistence count).

  - Volume Herfindahl concentration (20/60d): sum of squared daily volume
    shares. High = a few days carried all the activity (event-driven /
    one-sided); low = evenly spread. Complements the single existing entropy
    column with a different functional (HHI vs Shannon) at two horizons.
  - Volume entropy 20d (normalised Shannon, 0..1): diversity of the volume
    distribution; low = concentrated bursts.
  - Volume-price correlation (20/60d): rolling Pearson corr of |return| and
    volume. Positive = high-volume days are big-move days (information flow);
    near zero/negative = liquidity-driven churn.
  - Volume-weighted return skew (60d): are the high-volume days the up days or
    the down days? Sign of effort. Captures asymmetric participation.
  - Volume trend acceleration: 2nd difference of smoothed log-volume — is the
    participation trend speeding up or rolling over (curvature, not level).

All within-ticker, lookahead-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "vfd_volume_dispersion",
    "description": (
        "Volume concentration (Herfindahl 20/60d), normalised volume entropy, "
        "|return|-volume correlation (20/60d), volume-weighted return skew, "
        "and volume-trend acceleration (curvature)."
    ),
    "requires":    ["Close", "Volume"],
    "produces": [
        "vfd_vol_hhi_20",
        "vfd_vol_hhi_60",
        "vfd_vol_entropy_20",
        "vfd_volprice_corr_20",
        "vfd_volprice_corr_60",
        "vfd_vw_ret_skew_60",
        "vfd_vol_trend_accel",
    ],
    "tags":    ["volume", "flow", "concentration", "coupling"],
    "version": "1.0",
    "author":  "feature-gen",
}


def _herfindahl(arr: np.ndarray) -> float:
    """Sum of squared shares of a non-negative window. 1/n .. 1."""
    tot = arr.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    shares = arr / tot
    return float(np.square(shares).sum())


def _norm_entropy(arr: np.ndarray) -> float:
    """Shannon entropy of the volume shares, normalised to [0,1] by log(n)."""
    tot = arr.sum()
    n = len(arr)
    if not np.isfinite(tot) or tot <= 0 or n <= 1:
        return np.nan
    p = arr / tot
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    ent = -(p * np.log(p)).sum()
    return float(ent / np.log(n))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float).clip(lower=0)

    ret = close.pct_change()
    abs_ret = ret.abs()

    # ---- Volume concentration (Herfindahl) at two horizons -----------------
    for w in (20, 60):
        mp = max(5, w // 2)
        df[f"vfd_vol_hhi_{w}"] = volume.rolling(w, min_periods=mp).apply(
            _herfindahl, raw=True
        )

    # ---- Normalised volume entropy (diversity of participation) ------------
    df["vfd_vol_entropy_20"] = volume.rolling(20, min_periods=10).apply(
        _norm_entropy, raw=True
    )

    # ---- |return|-volume coupling (information vs churn) -------------------
    for w in (20, 60):
        mp = max(10, w // 2)
        df[f"vfd_volprice_corr_{w}"] = (
            abs_ret.rolling(w, min_periods=mp).corr(volume).clip(-1, 1)
        )

    # ---- Volume-weighted return skew (which side gets the effort) ----------
    # Weight each day's return by its volume share over a 60d window, then take
    # a volume-weighted standardised third moment. Vectorised via rolling
    # weighted moments.
    w = 60
    mp = 30
    vsum = volume.rolling(w, min_periods=mp).sum().replace(0, np.nan)
    wret1 = (volume * ret).rolling(w, min_periods=mp).sum() / vsum          # mean
    wret2 = (volume * ret.pow(2)).rolling(w, min_periods=mp).sum() / vsum
    wret3 = (volume * ret.pow(3)).rolling(w, min_periods=mp).sum() / vsum
    var_w = (wret2 - wret1.pow(2)).clip(lower=0)
    std_w = np.sqrt(var_w).replace(0, np.nan)
    # central third moment = E[x^3] - 3*mu*E[x^2] + 2*mu^3
    m3 = wret3 - 3.0 * wret1 * wret2 + 2.0 * wret1.pow(3)
    df["vfd_vw_ret_skew_60"] = (m3 / std_w.pow(3)).clip(-10, 10)

    # ---- Volume-trend acceleration (curvature of participation) ------------
    log_vol = np.log(volume.replace(0, np.nan) + 1.0)
    smooth = log_vol.ewm(span=10, min_periods=5, adjust=False).mean()
    accel = smooth.diff().diff()  # 2nd difference = acceleration
    # standardise by rolling dispersion so it is comparable across tickers
    accel_scale = accel.rolling(60, min_periods=20).std().replace(0, np.nan)
    df["vfd_vol_trend_accel"] = (accel / accel_scale).clip(-10, 10)

    return df
