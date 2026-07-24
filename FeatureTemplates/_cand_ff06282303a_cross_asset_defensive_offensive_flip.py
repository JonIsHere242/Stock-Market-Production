"""
ff06282303a_cross_asset_defensive_offensive_flip

Defensive/Offensive flip score:
  Over trailing 90 days, compute ticker return and SPY return each day.
  Split days into SPY-up and SPY-down buckets.
  rs_down = mean(ticker_ret - SPY_ret) on SPY-down days
  rs_up   = mean(ticker_ret - SPY_ret) on SPY-up   days
  flip    = rs_down - rs_up
    > 0 → defensive (outperforms more in down markets)
    < 0 → offensive (outperforms more in up markets)

Produced columns:
  ff06282303a_cross_asset_defensive_offensive_flip_score  : flip value (rs_down - rs_up), window=90d
  ff06282303a_cross_asset_defensive_offensive_flip_rs_down: mean excess return on SPY-down days
  ff06282303a_cross_asset_defensive_offensive_flip_rs_up  : mean excess return on SPY-up days
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282303a_cross_asset_defensive_offensive_flip",
    "description": (
        "Per-ticker defensive/offensive flip score. Over a trailing 90-day window, "
        "computes mean excess return (ticker - SPY) separately on SPY-down vs SPY-up days. "
        "flip = rs_down - rs_up: positive = defensive character (alpha in drawdowns), "
        "negative = offensive character (alpha in rallies). Requires >=5 days per bucket "
        "else the component is set to 0.0. Uses SPY daily close returns from _indexes."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282303a_cross_asset_defensive_offensive_flip_score",
        "ff06282303a_cross_asset_defensive_offensive_flip_rs_down",
        "ff06282303a_cross_asset_defensive_offensive_flip_rs_up",
    ],
    "tags": ["cross_asset", "relative_strength", "defensive", "regime", "SPY"],
    "version": "1.0.0",
    "author": "feature-factory ff06282303a",
}

_WINDOW = 90
_MIN_BUCKET = 5
_COL_SCORE = "ff06282303a_cross_asset_defensive_offensive_flip_score"
_COL_DOWN = "ff06282303a_cross_asset_defensive_offensive_flip_rs_down"
_COL_UP = "ff06282303a_cross_asset_defensive_offensive_flip_rs_up"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_SCORE] = np.nan
    df[_COL_DOWN] = np.nan
    df[_COL_UP] = np.nan

    if len(df) < 2:
        return df

    # -----------------------------------------------------------------------
    # Fetch SPY close and align to ticker dates via merge_asof
    # -----------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_series is None or len(spy_series) == 0:
        return df

    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date").reset_index(drop=True)

    spy_df = spy_df.sort_values("Date")
    merged = pd.merge_asof(work, spy_df, on="Date", direction="backward")

    # Compute daily returns (percentage, as fraction)
    ticker_ret = merged["Close"].pct_change()
    spy_ret = merged["spy_close"].pct_change()

    excess = ticker_ret - spy_ret
    spy_up_flag = spy_ret > 0  # True = up day, False = down (or flat) day

    n = len(merged)

    score_arr = np.full(n, np.nan)
    rs_down_arr = np.full(n, np.nan)
    rs_up_arr = np.full(n, np.nan)

    excess_np = excess.to_numpy(dtype=float)
    spy_up_np = spy_up_flag.to_numpy()
    spy_ret_np = spy_ret.to_numpy(dtype=float)

    # Rolling window: for each bar i, look back over [i-WINDOW+1 .. i]
    # Use cumulative sums to vectorise the rolling split-bucket means.
    # We need count/sum of excess on up-days and down-days.
    # Mark rows with valid excess (not NaN) and valid spy_ret (not NaN).
    valid = ~(np.isnan(excess_np) | np.isnan(spy_ret_np))

    # Precompute per-bar contributions
    exc_up = np.where(valid & spy_up_np, excess_np, 0.0)
    cnt_up = np.where(valid & spy_up_np, 1.0, 0.0)
    exc_dn = np.where(valid & (~spy_up_np), excess_np, 0.0)
    cnt_dn = np.where(valid & (~spy_up_np), 1.0, 0.0)

    # Cumulative sums for O(1) window queries
    cum_exc_up = np.cumsum(exc_up)
    cum_cnt_up = np.cumsum(cnt_up)
    cum_exc_dn = np.cumsum(exc_dn)
    cum_cnt_dn = np.cumsum(cnt_dn)

    for i in range(1, n):  # start at 1 (need at least 2 bars for pct_change)
        start = max(0, i - _WINDOW + 1)

        if start == 0:
            s_eu = cum_exc_up[i]
            s_cu = cum_cnt_up[i]
            s_ed = cum_exc_dn[i]
            s_cd = cum_cnt_dn[i]
        else:
            s_eu = cum_exc_up[i] - cum_exc_up[start - 1]
            s_cu = cum_cnt_up[i] - cum_cnt_up[start - 1]
            s_ed = cum_exc_dn[i] - cum_exc_dn[start - 1]
            s_cd = cum_cnt_dn[i] - cum_cnt_dn[start - 1]

        mean_up = s_eu / s_cu if s_cu >= _MIN_BUCKET else 0.0
        mean_dn = s_ed / s_cd if s_cd >= _MIN_BUCKET else 0.0

        rs_up_arr[i] = mean_up
        rs_down_arr[i] = mean_dn
        score_arr[i] = mean_dn - mean_up

    # Map results back to original df index (which may not be sorted)
    orig_dates = pd.to_datetime(df["Date"]).values
    work_dates = merged["Date"].values

    date_to_idx = {d: idx for idx, d in enumerate(work_dates)}

    score_out = np.full(len(df), np.nan)
    dn_out = np.full(len(df), np.nan)
    up_out = np.full(len(df), np.nan)

    for j, d in enumerate(orig_dates):
        idx = date_to_idx.get(d)
        if idx is not None:
            score_out[j] = score_arr[idx]
            dn_out[j] = rs_down_arr[idx]
            up_out[j] = rs_up_arr[idx]

    df[_COL_SCORE] = score_out
    df[_COL_DOWN] = dn_out
    df[_COL_UP] = up_out

    return df
