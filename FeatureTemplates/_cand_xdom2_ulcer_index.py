"""
Ulcer Index & Martin pain metric (Martin 1989).

Rolling 60-day Ulcer Index = RMS of the percentage drawdown from the
running 60-day peak close.  Penalises both depth AND duration of
drawdowns, unlike standard deviation which is symmetric.

Per-ticker implementation from OHLCV Close only -- no cross-sectional
dependency.  Produces:
  xdom2_ulcer_index_60     : Ulcer Index over 60-day rolling window
  xdom2_ulcer_index_20     : Ulcer Index over 20-day rolling window (faster)
  xdom2_ulcer_trend_60     : 20-day change in xdom2_ulcer_index_60
                             (rising = worsening drawdown pain)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_ulcer_index",
    "description": (
        "Rolling Ulcer Index (Martin 1989): RMS of percentage drawdown from "
        "the running peak close over a 60-day window.  Captures drawdown pain "
        "that penalises both depth and duration, unlike standard deviation. "
        "Also produces a 20-day window variant and a 20-day trend in the 60d UI "
        "to capture acceleration of pain.  Per-ticker OHLCV proxy -- fully "
        "faithful to the original method (no cross-sectional component required)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_ulcer_index_60",
        "xdom2_ulcer_index_20",
        "xdom2_ulcer_trend_60",
    ],
    "tags": ["risk", "drawdown", "ulcer_index", "martin", "cross_domain"],
    "version": "1.0",
    "author": "Ulcer index & Martin pain metric (Martin 1989); spec: Cross-domain / practitioner method transfer (batch 2)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    # ------------------------------------------------------------------ #
    # Helper: rolling RMS of percentage drawdown from running peak        #
    # ------------------------------------------------------------------ #
    def _ulcer(window: int) -> np.ndarray:
        result = np.full(n, np.nan)
        if n < window:
            return result

        for t in range(window - 1, n):
            window_slice = close[t - window + 1 : t + 1]
            # running peak within the window
            peaks = np.maximum.accumulate(window_slice)
            # pct drawdown from peak (always <= 0, we want the magnitude)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_dd = np.where(
                    peaks > 0,
                    (window_slice - peaks) / peaks * 100.0,
                    np.nan,
                )
            # RMS of drawdowns
            sq = pct_dd ** 2
            valid = sq[np.isfinite(sq)]
            if len(valid) > 0:
                result[t] = np.sqrt(np.mean(valid))
        return result

    # Vectorised implementation using sliding windows (fast for 700 rows)
    def _ulcer_vec(window: int) -> np.ndarray:
        """Vectorised via cummax trick; O(n) not O(n*w)."""
        result = np.full(n, np.nan)
        if n < window:
            return result

        # For each t compute: sqrt( mean( ((c - peak_in_window)/peak_in_window*100)^2 ) )
        # We use a loop over the window offset positions, which is O(window) but
        # avoids O(n^2) per-row loop.
        sq_sum = np.zeros(n, dtype=np.float64)
        sq_cnt = np.zeros(n, dtype=np.float64)

        for k in range(window):
            # position within window: k steps before current bar
            # c_k = close[t-k], peak up to that point = max(close[t-window+1 .. t-k+1])
            # We approximate running peak using a rolling max on the full series;
            # for position t and offset k: c[t-k] vs max(c[t-window+1 : t-k+1])
            # This requires a forward look into the suffix -- instead use
            # the exact formula: peak_at_offset_k_for_bar_t = max(close[t-window+1 .. t-k+1])
            # We unroll: shift close by k, then rolling max over (window-k) bars
            shifted_close = np.concatenate([np.full(k, np.nan), close[: n - k]])
            roll_len = window - k  # number of bars from window start to offset k
            if roll_len < 1:
                continue
            # rolling max of shifted_close over roll_len bars ending at each t
            peak = np.full(n, np.nan)
            for t in range(roll_len - 1, n):
                seg = shifted_close[t - roll_len + 1 : t + 1]
                valid_seg = seg[np.isfinite(seg)]
                if len(valid_seg) > 0:
                    peak[t] = np.max(valid_seg)

            with np.errstate(divide="ignore", invalid="ignore"):
                dd_pct = np.where(
                    peak > 0,
                    (shifted_close - peak) / peak * 100.0,
                    np.nan,
                )
            finite_mask = np.isfinite(dd_pct)
            sq_sum[finite_mask] += dd_pct[finite_mask] ** 2
            sq_cnt[finite_mask] += 1

        valid_mask = (sq_cnt > 0) & (np.arange(n) >= window - 1)
        result[valid_mask] = np.sqrt(sq_sum[valid_mask] / sq_cnt[valid_mask])
        return result

    # For small n (<=700 typical), a straightforward numpy-vectorised approach
    # using stride tricks is cleanest and fast enough.
    def _ulcer_strided(window: int) -> np.ndarray:
        result = np.full(n, np.nan)
        if n < window:
            return result

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(close, window)  # shape (n-window+1, window)
        # running peak along axis=1 (cummax within each window)
        peaks = np.maximum.accumulate(windows, axis=1)  # same shape
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_dd = np.where(peaks > 0, (windows - peaks) / peaks * 100.0, np.nan)
        # RMS over axis=1
        sq = pct_dd ** 2
        # nanmean per row
        with np.errstate(invalid="ignore"):
            row_mean_sq = np.nanmean(sq, axis=1)
        ui = np.sqrt(row_mean_sq)
        result[window - 1 :] = ui
        return result

    ui_60 = _ulcer_strided(60)
    ui_20 = _ulcer_strided(20)

    # 20-day change in the 60-day UI (trend / acceleration of pain)
    ui_60_series = pd.Series(ui_60, index=df.index)
    ui_trend = ui_60_series.diff(20).to_numpy()

    df["xdom2_ulcer_index_60"] = ui_60
    df["xdom2_ulcer_index_20"] = ui_20
    df["xdom2_ulcer_trend_60"] = ui_trend

    # Guard: replace any inf that slipped through
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
