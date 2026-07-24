from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282347f_volume_price_price_move_per_abnormal_volume_40",
    "description": (
        "Price move per unit of abnormal volume over a 40-day rolling window. "
        "On days where the volume z-score exceeds 1.0, computes mean(|return| / vol_zscore). "
        "Low values = high-volume days absorbed with little price move (noise/liquidity); "
        "high values = abnormal volume is informative (informed trading). "
        "Produces a level (rolling ratio), a sign-aware directional variant (return/vol_zscore, "
        "preserving up/down directionality), and a 10-day EWM slope of the level."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282347f_volume_price_price_move_per_abnormal_volume_40_level",
        "ff06282347f_volume_price_price_move_per_abnormal_volume_40_dir",
        "ff06282347f_volume_price_price_move_per_abnormal_volume_40_slope",
    ],
    "tags": ["volume", "price", "informed_trading", "abnormal_volume", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory ff06282347f",
}

_WINDOW = 40
_ZSCORE_THRESH = 1.0


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out_level = "ff06282347f_volume_price_price_move_per_abnormal_volume_40_level"
    out_dir = "ff06282347f_volume_price_price_move_per_abnormal_volume_40_dir"
    out_slope = "ff06282347f_volume_price_price_move_per_abnormal_volume_40_slope"

    # Initialise all produced columns to NaN on every code path
    df[out_level] = np.nan
    df[out_dir] = np.nan
    df[out_slope] = np.nan

    n = len(df)
    if n < _WINDOW + 1:
        return df

    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Daily log returns (causal: shift by 1)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # Rolling volume z-score over the past _WINDOW bars
    # Use pandas rolling for clarity and speed
    vol_s = pd.Series(volume)
    roll_mean = vol_s.rolling(_WINDOW, min_periods=_WINDOW).mean()
    roll_std = vol_s.rolling(_WINDOW, min_periods=_WINDOW).std(ddof=1)

    vol_mean_arr = roll_mean.to_numpy(dtype=np.float64)
    vol_std_arr = roll_std.to_numpy(dtype=np.float64)

    # Guard against zero std
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_zscore = np.where(
            vol_std_arr > 0,
            (volume - vol_mean_arr) / vol_std_arr,
            np.nan,
        )

    # For each bar t, compute the rolling metric over [t-_WINDOW+1 .. t]
    # using a fixed-stride vectorised approach via pandas rolling apply.
    # We build per-bar |ret|/zscore and direction ret/zscore, then
    # compute rolling mean only on the subset where zscore > threshold.

    abs_ret = np.abs(ret)

    with np.errstate(invalid="ignore", divide="ignore"):
        # |return| / vol_zscore (informational price move per unit abnormal vol)
        ratio_abs = np.where(vol_zscore > 0, abs_ret / vol_zscore, np.nan)
        # signed return / vol_zscore (directional variant)
        ratio_dir = np.where(vol_zscore > 0, ret / vol_zscore, np.nan)

    # mask: only include days where zscore > threshold
    mask_high = vol_zscore > _ZSCORE_THRESH  # bool array

    # Apply mask: set to nan where condition not met
    ratio_abs_masked = np.where(mask_high, ratio_abs, np.nan)
    ratio_dir_masked = np.where(mask_high, ratio_dir, np.nan)

    # Rolling mean over _WINDOW bars of masked values
    # Default to 0 if no days qualify in the window (per spec)
    def rolling_nanmean_with_default(arr: np.ndarray, window: int) -> np.ndarray:
        s = pd.Series(arr)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) == 0:
                result[i] = 0.0  # no qualifying days -> default 0 per spec
            else:
                result[i] = valid.mean()
        return result

    # Vectorised version using pandas rolling (faster than python loop for large n):
    # We compute the count of valid (non-nan) entries and sum separately.
    s_abs = pd.Series(ratio_abs_masked)
    s_dir = pd.Series(ratio_dir_masked)

    roll_sum_abs = s_abs.rolling(_WINDOW, min_periods=1).sum()
    roll_count_abs = s_abs.rolling(_WINDOW, min_periods=1).count()
    roll_sum_dir = s_dir.rolling(_WINDOW, min_periods=1).sum()
    roll_count_dir = s_dir.rolling(_WINDOW, min_periods=1).count()

    sum_abs_arr = roll_sum_abs.to_numpy(dtype=np.float64)
    cnt_abs_arr = roll_count_abs.to_numpy(dtype=np.float64)
    sum_dir_arr = roll_sum_dir.to_numpy(dtype=np.float64)
    cnt_dir_arr = roll_count_dir.to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        level_arr = np.where(
            cnt_abs_arr > 0,
            sum_abs_arr / cnt_abs_arr,
            0.0,  # default 0 if no qualifying days
        )
        dir_arr = np.where(
            cnt_dir_arr > 0,
            sum_dir_arr / cnt_dir_arr,
            0.0,
        )

    # Only expose values from bar _WINDOW onward (need full window for vol z-score)
    level_arr[:_WINDOW] = np.nan
    dir_arr[:_WINDOW] = np.nan

    # Sanitize infinities
    level_arr = np.where(np.isfinite(level_arr), level_arr, np.nan)
    dir_arr = np.where(np.isfinite(dir_arr), dir_arr, np.nan)

    df[out_level] = level_arr
    df[out_dir] = dir_arr

    # 10-bar EWM slope of the level (change in informativeness)
    level_s = pd.Series(level_arr)
    ewm_now = level_s.ewm(span=10, min_periods=5).mean()
    ewm_lag = level_s.shift(5).ewm(span=10, min_periods=5).mean()
    slope_arr = (ewm_now - ewm_lag).to_numpy(dtype=np.float64)
    slope_arr = np.where(np.isfinite(slope_arr), slope_arr, np.nan)
    df[out_slope] = slope_arr

    return df
