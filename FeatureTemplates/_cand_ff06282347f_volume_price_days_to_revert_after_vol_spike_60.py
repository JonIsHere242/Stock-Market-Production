from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff06282347f_volume_price_days_to_revert_after_vol_spike_60",
    "description": (
        "Mean reversion after volume spikes: within a rolling 60-day window, "
        "identifies volume-spike days (z-score of log1p(Volume) vs trailing "
        "40-day mean/std > 2). For each causal spike (spike day + 3 forward bars "
        "all within the 60-day window), measures the 3-day forward signed return "
        "relative to the spike-day return direction (reversion fraction). "
        "Produces: (1) mean reversion fraction across spikes in the window "
        "(positive = mean-reverting, negative = momentum), (2) spike count in "
        "the window, (3) ewm-smoothed reversion fraction for a momentum-like "
        "slope variant. Per-ticker causal proxy; no cross-sectional data needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_revert_frac",
        "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_spike_count",
        "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_revert_ewm",
    ],
    "tags": ["volume", "mean_reversion", "spike", "price_action"],
    "version": "1.0.0",
    "author": "feature-factory ff06282347f",
}

_COL_FRAC  = "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_revert_frac"
_COL_COUNT = "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_spike_count"
_COL_EWM   = "ff06282347f_volume_price_days_to_revert_after_vol_spike_60_revert_ewm"

_WINDOW     = 60   # rolling look-back window for aggregating spikes
_VOL_WINDOW = 40   # trailing window for volume z-score
_Z_THRESH   = 2.0  # spike threshold
_FWD        = 3    # forward bars to measure reversion


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path.
    df[_COL_FRAC]  = np.nan
    df[_COL_COUNT] = np.nan
    df[_COL_EWM]   = np.nan

    n = len(df)
    if n < _VOL_WINDOW + _FWD + 1:
        return df

    # --- volume z-score (trailing, causal) ---
    log_vol = np.log1p(df["Volume"].to_numpy(dtype=np.float64))
    roll_mean = (
        pd.Series(log_vol)
        .rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW)
        .mean()
        .to_numpy()
    )
    roll_std = (
        pd.Series(log_vol)
        .rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW)
        .std(ddof=1)
        .to_numpy()
    )
    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = np.where(roll_std > 0, (log_vol - roll_mean) / roll_std, np.nan)

    # --- daily close returns (causal: today vs yesterday) ---
    close = df["Close"].to_numpy(dtype=np.float64)
    ret = np.empty(n, dtype=np.float64)
    ret[:] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = np.where(
            close[:-1] != 0,
            (close[1:] - close[:-1]) / np.abs(close[:-1]),
            np.nan,
        )

    # --- spike indicator: bar i is a spike if vol_z[i] > threshold ---
    # For bar i to have 3 forward bars of room, i <= n - 1 - _FWD.
    is_spike = (vol_z > _Z_THRESH)  # bool array length n; NaN -> False

    # --- per-spike reversion fraction ---
    # revert_frac[i] = -(sum of forward 3-day return) / spike_day_return
    # i.e. if spike-day up +5%, and next 3 days return -3%, fraction = 0.6
    # Positive fraction = mean reverting.
    revert_at = np.full(n, np.nan)
    for i in range(1, n - _FWD):   # i>=1 so ret[i] is valid; i <= n-1-_FWD
        if not is_spike[i]:
            continue
        spike_ret = ret[i]
        if spike_ret == 0 or np.isnan(spike_ret):
            continue
        fwd_ret = np.sum(ret[i + 1 : i + 1 + _FWD])
        if np.isnan(fwd_ret):
            continue
        # Reversion fraction: negative forward return relative to spike direction
        revert_at[i] = -fwd_ret / np.abs(spike_ret)

    # --- rolling 60-bar aggregation (causal) ---
    # At bar t, look back over the window ending at t (exclusive of future bars).
    # We use a simple O(n) pass via pandas rolling on revert_at and spike indicator.
    revert_series = pd.Series(revert_at)
    spike_series  = pd.Series(is_spike.astype(np.float64))
    spike_series[spike_series == 0] = np.nan  # only count valid spikes (non-NaN revert)

    # Rolling mean of revert_at (only non-NaN spikes contribute)
    roll_frac = (
        revert_series
        .rolling(_WINDOW, min_periods=1)
        .mean()
    )
    # Rolling sum of spike count (bars with a reversion estimate)
    roll_count = (
        revert_series.notna().astype(np.float64)
        .rolling(_WINDOW, min_periods=1)
        .sum()
    )

    df[_COL_FRAC]  = roll_frac.to_numpy()
    df[_COL_COUNT] = roll_count.to_numpy()

    # EWM-smoothed version of the per-spike reversion signal
    # (captures recent trend in reversion behaviour)
    ewm_frac = (
        revert_series
        .fillna(method="ffill")   # carry last spike value forward
        .ewm(span=_WINDOW, min_periods=1, adjust=False)
        .mean()
    )
    # Mask leading stretch where we have no spikes yet
    first_spike = revert_series.first_valid_index()
    if first_spike is not None:
        ewm_arr = ewm_frac.to_numpy(dtype=np.float64)
        ewm_arr[:first_spike] = np.nan
        df[_COL_EWM] = ewm_arr
    # else stays NaN

    return df
