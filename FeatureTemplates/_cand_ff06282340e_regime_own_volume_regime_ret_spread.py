from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff06282340e_regime_own_volume_regime_ret_spread",
    "description": (
        "Volume-regime return spread (per-ticker proxy). "
        "Computes a trailing 60-bar volume z-score, then over a 120-bar window "
        "calculates mean return on high-volume days (z>0.5) minus mean return on "
        "low-volume days (z<-0.5). A positive spread means volume spikes precede "
        "favourable moves; negative means spikes precede adverse moves. "
        "Also emits the rolling z-score itself and a 20-bar EWM of the spread "
        "for trend-of-regime. Empty buckets (no high or no low vol days in window) "
        "produce NaN for that bar. Per-ticker only -- no cross-sectional ranking."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282340e_vol_zscore",
        "ff06282340e_vol_regime_ret_spread",
        "ff06282340e_spread_ewm20",
    ],
    "tags": ["volume", "regime", "return_spread", "activity"],
    "version": "1.0",
    "author": "feature-factory ff06282340e",
}

_VOL_WIN = 60   # z-score lookback
_RET_WIN = 120  # spread lookback
_HI_Z = 0.5
_LO_Z = -0.5
_EWM_SPAN = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front (covers empty / short series)
    df["ff06282340e_vol_zscore"] = np.nan
    df["ff06282340e_vol_regime_ret_spread"] = np.nan
    df["ff06282340e_spread_ewm20"] = np.nan

    n = len(df)
    if n < _VOL_WIN + 1:
        return df

    vol = df["Volume"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    # Guard zero close values to avoid division issues
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(close[:-1] != 0, (close[1:] - close[:-1]) / close[:-1], np.nan)
    # ret[i] = return from bar i to bar i+1 -- but we want SAME-bar return (open->close proxy)
    # Use simple bar-close return shifted by 1 (return known at bar i = (close[i]-close[i-1])/close[i-1])
    bar_ret = np.empty(n, dtype=float)
    bar_ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bar_ret[1:] = np.where(
            close[:-1] != 0,
            (close[1:] - close[:-1]) / close[:-1],
            np.nan,
        )

    # --- Rolling volume z-score (60-bar) ---
    vol_s = pd.Series(vol)
    roll_mean = vol_s.rolling(_VOL_WIN, min_periods=_VOL_WIN).mean()
    roll_std = vol_s.rolling(_VOL_WIN, min_periods=_VOL_WIN).std(ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = np.where(roll_std > 0, (vol - roll_mean.to_numpy()) / roll_std.to_numpy(), np.nan)

    df["ff06282340e_vol_zscore"] = vol_z

    # --- Rolling 120-bar spread between high-vol and low-vol mean returns ---
    # We need a causal rolling window: for each bar t, look at [t-119 .. t]
    spread = np.full(n, np.nan, dtype=float)

    # Vectorised approach: use stride-trick via a loop over each bar but only
    # materialise a 2-D matrix once (feasible for n~700, win=120 -> 700x120 float64 ~0.67MB)
    if n >= _RET_WIN:
        # Build 2-D arrays: rows = bars, cols = window positions
        from numpy.lib.stride_tricks import sliding_window_view

        # We need ret and vol_z aligned for window [t-_RET_WIN+1 .. t]
        # sliding_window_view gives shape (n - win + 1, win)
        ret_mat = sliding_window_view(bar_ret, _RET_WIN)     # shape: (n-119, 120)
        z_mat = sliding_window_view(vol_z, _RET_WIN)         # shape: (n-119, 120)

        hi_mask = z_mat > _HI_Z   # (n-119, 120) bool
        lo_mask = z_mat < _LO_Z

        # Replace NaN ret with 0 contribution; track valid counts
        ret_mat_f = np.where(np.isfinite(ret_mat), ret_mat, np.nan)

        # mean of hi-vol returns per row
        hi_ret = np.where(hi_mask, ret_mat_f, np.nan)
        lo_ret = np.where(lo_mask, ret_mat_f, np.nan)

        with np.errstate(invalid="ignore"):
            hi_mean = np.nanmean(hi_ret, axis=1)   # NaN if no valid hi bars
            lo_mean = np.nanmean(lo_ret, axis=1)   # NaN if no valid lo bars

        # nanmean returns nan when all-nan -- check counts to guard empty buckets
        hi_count = np.sum(hi_mask & np.isfinite(ret_mat_f), axis=1)
        lo_count = np.sum(lo_mask & np.isfinite(ret_mat_f), axis=1)

        spread_vals = np.where(
            (hi_count > 0) & (lo_count > 0),
            hi_mean - lo_mean,
            np.nan,
        )
        # Place at bars [_RET_WIN-1 .. n-1]
        spread[_RET_WIN - 1:] = spread_vals

    df["ff06282340e_vol_regime_ret_spread"] = spread

    # --- EWM(20) of the spread ---
    spread_ewm = (
        pd.Series(spread)
        .ewm(span=_EWM_SPAN, min_periods=_EWM_SPAN, adjust=False)
        .mean()
        .to_numpy()
    )
    df["ff06282340e_spread_ewm20"] = spread_ewm

    return df
