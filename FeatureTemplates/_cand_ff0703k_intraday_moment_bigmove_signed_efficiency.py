from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff0703k_intraday_moment_bigmove_signed_efficiency",
    "description": (
        "Signed momentum concentrated in large-range days only. Range R_t = "
        "(High_t-Low_t)/Close_{t-1} (zero-denom guarded); close-to-close return "
        "r_t = Close_t/Close_{t-1}-1. Over a rolling W=60 window, days with "
        "R_t >= 70th percentile of R within the window are flagged as "
        "'big-range' days. LEVEL "
        "(ff0703k_intraday_moment_bigmove_signed_efficiency) = "
        "sum(r_t over big-range days) / sum(|r_t| over big-range days), a signed "
        "efficiency ratio in [-1,1] measuring the net directional drift delivered "
        "specifically by the high-range days (NaN if <5 big-range days in the "
        "window or the denominator is zero). DYNAMIC "
        "(..._drift_slope) = OLS slope of the cumulative sum of r_t over the "
        "big-range days only (ordered by date) regressed against event index "
        "within the window, capturing whether the big-move drift is accelerating "
        "up or down. Implemented fully vectorised via sliding_window_view: the "
        "per-window big-range mask, cumulative-sum-at-events, and event rank are "
        "built as (n_windows x 60) arrays and the OLS slope closed-form "
        "(n*sum_xy - sum_x*sum_y)/(n*sum_x2 - sum_x^2) is computed with the "
        "non-event columns zeroed out by the mask -- exactly equivalent to a "
        "per-window OLS on the variable-length event subsequence, without any "
        "Python-level loop over rows. Pure per-ticker OHLC; no lookahead (window "
        "t-59..t only)."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff0703k_intraday_moment_bigmove_signed_efficiency",
        "ff0703k_intraday_moment_bigmove_signed_efficiency_drift_slope",
    ],
    "tags": ["intraday_moment", "momentum", "range", "efficiency", "asymmetry"],
    "version": "1.0",
    "author": "ff0703k spec (exact match to spec definition; vectorised via sliding_window_view, no python row loop)",
}

_WINDOW = 60
_PCTL = 70.0
_MIN_EVENTS = 5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    level = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)

    close_prev = df["Close"].shift(1)
    close_prev_safe = close_prev.replace(0, np.nan)

    R = (df["High"] - df["Low"]) / close_prev_safe
    r = df["Close"] / close_prev_safe - 1.0

    R_arr = R.to_numpy(dtype=np.float64)
    r_arr = r.to_numpy(dtype=np.float64)

    if n >= _WINDOW:
        swR = np.lib.stride_tricks.sliding_window_view(R_arr, _WINDOW)
        swr = np.lib.stride_tricks.sliding_window_view(r_arr, _WINDOW)

        # 70th percentile threshold of R within each window (NaN-aware)
        thresh = np.nanpercentile(swR, _PCTL, axis=1)
        mask = swR >= thresh[:, None]
        mask = np.where(np.isnan(swR), False, mask)

        count = mask.sum(axis=1)

        r_masked = np.where(mask, swr, 0.0)
        sum_r = r_masked.sum(axis=1)
        sum_abs = np.where(mask, np.abs(swr), 0.0).sum(axis=1)

        valid = count >= _MIN_EVENTS
        denom_ok = sum_abs > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            lvl = np.where(valid & denom_ok, sum_r / np.where(sum_abs == 0, np.nan, sum_abs), np.nan)

        # OLS slope of cumulative-sum-at-events vs event rank, via masked closed-form
        cm = np.cumsum(r_masked, axis=1)  # cumsum of r over big-range days only (flat elsewhere)
        rank = np.cumsum(mask.astype(np.int64), axis=1) - 1  # 0-indexed event rank
        rank_f = rank.astype(np.float64)

        m = count.astype(np.float64)
        sum_x = (rank_f * mask).sum(axis=1)
        sum_y = (cm * mask).sum(axis=1)
        sum_xy = (rank_f * cm * mask).sum(axis=1)
        sum_x2 = (rank_f * rank_f * mask).sum(axis=1)

        denom = m * sum_x2 - sum_x ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            slp = np.where(
                valid & (denom != 0),
                (m * sum_xy - sum_x * sum_y) / np.where(denom == 0, np.nan, denom),
                np.nan,
            )

        level[_WINDOW - 1:] = lvl
        slope[_WINDOW - 1:] = slp

    df["ff0703k_intraday_moment_bigmove_signed_efficiency"] = level
    df["ff0703k_intraday_moment_bigmove_signed_efficiency_drift_slope"] = slope

    return df
