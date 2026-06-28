"""
ish_return_decomposition.py — Overnight vs intraday return decomposition.

A daily bar's total return splits cleanly into two non-overlapping pieces:
  - OVERNIGHT return  = Open_t / Close_{t-1} - 1   (gap from prior close to today's open)
  - INTRADAY  return  = Close_t / Open_t   - 1   (open-to-close session move)

These two components are economically distinct: the overnight piece is driven
mostly by news/order-flow accumulated while the market is closed, the intraday
piece by the live session. Their relative MEANS, VOLS, and MOMENTUM tell us which
leg is carrying the trend and how much of the risk is borne overnight — signal
that is largely orthogonal to a plain close-to-close momentum/vol stack.

All columns are dimensionless ratios or fractions, clipped to sane ranges.
Leading rolling NaN is expected and fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "ish_return_decomposition",
    "description": (
        "Overnight (prevClose->Open) vs intraday (Open->Close) return decomposition: "
        "rolling means, vols, ratio, momentum split and overnight risk share over 20/60d."
    ),
    "requires":    ["Open", "Close"],
    "produces":    [
        "ish_overnight_ret",
        "ish_intraday_ret",
        "ish_overnight_mean_20",
        "ish_overnight_mean_60",
        "ish_intraday_mean_20",
        "ish_intraday_mean_60",
        "ish_overnight_vol_20",
        "ish_intraday_vol_20",
        "ish_on_id_vol_ratio_20",
        "ish_on_id_mom_split_60",
        "ish_overnight_var_share_60",
    ],
    "tags":    ["overnight", "intraday", "return_decomposition", "gap"],
    "version": "1.0",
    "author": "feature-gen",
}

_EPS = 1e-9


def compute(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["Open"].astype("float64")
    close = df["Close"].astype("float64")
    prev_close = close.shift(1)

    # --- the two non-overlapping legs of the daily return -------------------
    overnight = open_ / prev_close.replace(0, np.nan) - 1.0
    intraday = close / open_.replace(0, np.nan) - 1.0

    # Clip pathological single-day moves (e.g. bad ticks / splits not adjusted)
    overnight = overnight.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)
    intraday = intraday.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5)

    df["ish_overnight_ret"] = overnight.values
    df["ish_intraday_ret"] = intraday.values

    # --- rolling means: which leg carries the drift ------------------------
    on_mean_20 = overnight.rolling(20, min_periods=10).mean()
    on_mean_60 = overnight.rolling(60, min_periods=30).mean()
    id_mean_20 = intraday.rolling(20, min_periods=10).mean()
    id_mean_60 = intraday.rolling(60, min_periods=30).mean()

    df["ish_overnight_mean_20"] = on_mean_20.values
    df["ish_overnight_mean_60"] = on_mean_60.values
    df["ish_intraday_mean_20"] = id_mean_20.values
    df["ish_intraday_mean_60"] = id_mean_60.values

    # --- rolling vols and their ratio --------------------------------------
    on_vol_20 = overnight.rolling(20, min_periods=10).std()
    id_vol_20 = intraday.rolling(20, min_periods=10).std()
    df["ish_overnight_vol_20"] = on_vol_20.values
    df["ish_intraday_vol_20"] = id_vol_20.values

    # ratio > 1 => more risk borne overnight than intraday
    vol_ratio = on_vol_20 / (id_vol_20 + _EPS)
    df["ish_on_id_vol_ratio_20"] = vol_ratio.replace([np.inf, -np.inf], np.nan).clip(0, 20).values

    # --- momentum split: signed share of trend from overnight leg ----------
    # In (-1, 1): +1 => the whole 60d drift comes from overnight gaps,
    #             -1 => entirely from the intraday session, opposite signs cancel.
    denom = on_mean_60.abs() + id_mean_60.abs() + _EPS
    mom_split = (on_mean_60.abs() - id_mean_60.abs()) / denom
    # carry the sign of the dominant leg's net drift for directional context
    dominant_sign = np.sign(np.where(on_mean_60.abs() >= id_mean_60.abs(),
                                     on_mean_60, id_mean_60))
    mom_split = mom_split * pd.Series(dominant_sign, index=on_mean_60.index)
    df["ish_on_id_mom_split_60"] = mom_split.replace([np.inf, -np.inf], np.nan).clip(-1, 1).values

    # --- overnight variance share over 60d ---------------------------------
    on_var_60 = overnight.rolling(60, min_periods=30).var()
    id_var_60 = intraday.rolling(60, min_periods=30).var()
    var_share = on_var_60 / (on_var_60 + id_var_60 + _EPS)
    df["ish_overnight_var_share_60"] = var_share.replace([np.inf, -np.inf], np.nan).clip(0, 1).values

    return df
