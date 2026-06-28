"""
vfd_volume_shock.py — Volume SHOCK dynamics and signed-pressure imbalance.

Theme: surprise in trading intensity and how it persists/decays, plus the
direction of where that volume went. Distinct from existing volume momentum
(ratio of means), unexpected-volume residual (`_suv`), and persistence-count
features.

  - Volume shock = (Volume - rolling mean) / rolling std at 5d and 20d
    windows. A standardised intensity surprise.
  - Shock persistence (decay/half-life proxy): fraction of the last 10 days
    on which a 20d shock exceeded +1 sigma — measures whether the surprise is
    a one-off spike or a sustained regime of elevated participation.
  - Shock autocorrelation lag-1: do volume surprises cluster (volume itself is
    famously autocorrelated)?
  - Up/down volume ratio over 20d and 60d: total volume on up-close days vs
    down-close days. >1 = net accumulation pressure, classic effort/result.
  - Signed-volume cumulative pressure: rolling 20d sum of sign(return)*volume
    normalised by rolling total volume — a [-1,1] net order-flow proxy.

All within-ticker, lookahead-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "vfd_volume_shock",
    "description": (
        "Standardised volume shocks (5/20d), shock persistence and "
        "autocorrelation, up/down volume ratios (20/60d), and signed-volume "
        "cumulative net pressure."
    ),
    "requires":    ["Close", "Volume"],
    "produces": [
        "vfd_vol_shock_5",
        "vfd_vol_shock_20",
        "vfd_vol_shock_persist_10",
        "vfd_vol_shock_autocorr_20",
        "vfd_updown_vol_ratio_20",
        "vfd_updown_vol_ratio_60",
        "vfd_signed_vol_pressure_20",
    ],
    "tags":    ["volume", "flow", "shock", "order_flow"],
    "version": "1.0",
    "author":  "feature-gen",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    ret = close.pct_change()
    sign = np.sign(ret).fillna(0.0)

    # ---- Standardised volume shock at two horizons --------------------------
    for w in (5, 20):
        mp = max(3, w // 2)
        mean_v = volume.rolling(w, min_periods=mp).mean()
        std_v = volume.rolling(w, min_periods=mp).std().replace(0, np.nan)
        df[f"vfd_vol_shock_{w}"] = ((volume - mean_v) / std_v).clip(-10, 10)

    shock20 = df["vfd_vol_shock_20"]

    # ---- Persistence: how sticky are positive shocks (decay proxy) ----------
    # Fraction of last 10 days with a >1 sigma shock. High = sustained regime,
    # low = isolated one-day spike that decayed immediately.
    is_hot = (shock20 > 1.0).astype(float)
    df["vfd_vol_shock_persist_10"] = is_hot.rolling(10, min_periods=5).mean()

    # ---- Shock clustering: lag-1 autocorrelation of the 20d shock series ----
    df["vfd_vol_shock_autocorr_20"] = (
        shock20.rolling(60, min_periods=30)
        .corr(shock20.shift(1))
        .clip(-1, 1)
    )

    # ---- Up vs down volume ratio (effort / result) -------------------------
    up_vol = volume.where(sign > 0, 0.0)
    dn_vol = volume.where(sign < 0, 0.0)
    for w in (20, 60):
        mp = max(5, w // 2)
        up_sum = up_vol.rolling(w, min_periods=mp).sum()
        dn_sum = dn_vol.rolling(w, min_periods=mp).sum()
        ratio = up_sum / (dn_sum + 1e-9)
        # log so symmetric around 0 (equal up/down volume); clip fat tails.
        df[f"vfd_updown_vol_ratio_{w}"] = np.log(ratio.clip(lower=1e-6)).clip(-5, 5)

    # ---- Signed-volume cumulative net pressure (order-flow proxy) ----------
    signed_vol = sign * volume
    net = signed_vol.rolling(20, min_periods=10).sum()
    tot = volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    df["vfd_signed_vol_pressure_20"] = (net / tot).clip(-1, 1)

    return df
