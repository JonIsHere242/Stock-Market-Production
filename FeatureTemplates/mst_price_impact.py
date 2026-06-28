"""
mst_price_impact.py — Price-impact and trading-friction illiquidity estimators.

Daily-OHLCV illiquidity / price-impact family, each measuring a DIFFERENT facet
of how much price moves per unit of trading and how that impact is distributed:

  1. Amihud (2002) illiquidity ratio = mean( |ret| / dollar_volume ), the
     canonical price-impact-per-dollar measure (20d / 60d, scaled to a
     readable magnitude).
  2. Kyle (1985) lambda proxy = rolling slope of |ret| regressed on signed
     dollar volume — the marginal price impact of order flow.  Estimated as
     cov(|ret|, signed_dollar_vol) / var(signed_dollar_vol) over a window.
  3. Price-impact ASYMMETRY = ratio of average per-volume impact on DOWN days
     vs UP days.  Captures whether selling moves price more than buying (a
     liquidity/fragility tell), orthogonal to the level of illiquidity.
  4. Lesmond zero-return frequency = fraction of (near-)zero-return days over a
     window — a transaction-cost / thin-trading proxy that needs no volume.

All ratios are clipped to sane bounds; divide-by-zero is guarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "mst_price_impact",
    "description": (
        "Daily price-impact illiquidity: Amihud ratio (20/60d), Kyle-lambda "
        "order-flow impact proxy (20/60d), up-vs-down impact asymmetry (20d), "
        "and Lesmond zero-return-day frequency (20/60d)."
    ),
    "requires":    ["Close", "Volume"],
    "produces":    [
        "mst_amihud_illiq_20d",
        "mst_amihud_illiq_60d",
        "mst_kyle_lambda_20d",
        "mst_kyle_lambda_60d",
        "mst_impact_asymmetry_20d",
        "mst_zero_return_freq_20d",
        "mst_zero_return_freq_60d",
    ],
    "tags":        ["liquidity", "microstructure", "price_impact"],
    "version":     "1.0",
    "author":      "feature-gen",
}

# A near-zero return: |ret| below this is treated as a "zero" trading day.
_ZERO_RET_EPS = 1e-4   # 1 basis point
_EPS = 1e-12


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    close_s = close.where(close > 0)
    ret = close_s.pct_change()
    abs_ret = ret.abs()

    dollar_vol = (close_s * volume).clip(lower=0.0)

    new_cols = {}

    # ------------------------------------------------------------------
    # 1. Amihud illiquidity = |ret| / dollar_volume, rolling mean.
    #    Scaled by 1e9 so values land in a human-readable range, then clipped.
    # ------------------------------------------------------------------
    daily_illiq = abs_ret / (dollar_vol + _EPS)
    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan) * 1e9
    for w in (20, 60):
        amihud = daily_illiq.rolling(w, min_periods=w // 2).mean()
        new_cols[f"mst_amihud_illiq_{w}d"] = amihud.clip(lower=0.0, upper=1e6)

    # ------------------------------------------------------------------
    # 2. Kyle lambda proxy: marginal price impact of signed order flow.
    #    Signed dollar volume = sign(ret) * dollar_volume (return-sign as the
    #    trade-direction proxy).  lambda = cov(ret, signed_dvol)/var(signed_dvol)
    #    over a rolling window.  Sign of lambda is informative (positive =
    #    flow pushes price the expected way); magnitude = impact intensity.
    # ------------------------------------------------------------------
    signed_dvol = np.sign(ret) * dollar_vol
    # Scale flow to $millions so the regression slope is numerically stable.
    signed_dvol_m = signed_dvol / 1e6
    for w in (20, 60):
        cov = ret.rolling(w, min_periods=w // 2).cov(signed_dvol_m)
        var = signed_dvol_m.rolling(w, min_periods=w // 2).var()
        lam = cov / (var + _EPS)
        lam = lam.replace([np.inf, -np.inf], np.nan)
        # Clip to a generous symmetric bound (returns per $M of flow).
        new_cols[f"mst_kyle_lambda_{w}d"] = lam.clip(-1.0, 1.0)

    # ------------------------------------------------------------------
    # 3. Price-impact asymmetry (20d): mean per-volume impact on down days
    #    relative to up days.  >1 => down moves cost more liquidity than up
    #    moves (downside-fragile); <1 => the reverse.
    # ------------------------------------------------------------------
    per_vol_impact = abs_ret / (dollar_vol + _EPS)
    per_vol_impact = per_vol_impact.replace([np.inf, -np.inf], np.nan)
    up_impact = per_vol_impact.where(ret > 0)
    down_impact = per_vol_impact.where(ret < 0)
    up_mean = up_impact.rolling(20, min_periods=5).mean()
    down_mean = down_impact.rolling(20, min_periods=5).mean()
    asym = down_mean / (up_mean + _EPS)
    asym = asym.replace([np.inf, -np.inf], np.nan)
    new_cols["mst_impact_asymmetry_20d"] = asym.clip(lower=0.0, upper=20.0)

    # ------------------------------------------------------------------
    # 4. Lesmond zero-return frequency: fraction of near-zero-return days.
    #    High frequency => thin trading / high implicit transaction costs.
    # ------------------------------------------------------------------
    is_zero = (abs_ret <= _ZERO_RET_EPS).astype(float)
    # Mask the leading NaN return so it is not counted as a zero day.
    is_zero = is_zero.where(ret.notna())
    for w in (20, 60):
        new_cols[f"mst_zero_return_freq_{w}d"] = is_zero.rolling(
            w, min_periods=w // 2
        ).mean()

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
