from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282347f_microstructure_clv_trend_40",
    "description": (
        "Trend in Close Location Value (CLV) over a trailing 40-bar window. "
        "CLV = ((close-low)-(high-close))/(high-low) ranges in [-1,+1]; +1 means "
        "close at the high, -1 at the low. The OLS slope of the trailing 40-day CLV "
        "series captures whether intraday closing pressure is strengthening toward the "
        "high (rising slope = buy pressure) or drifting toward the low (falling = sell "
        "pressure). Also produces the raw 10-bar EMA of CLV as a level, and the ratio "
        "of the slope to its trailing standard deviation (t-stat proxy) for confidence."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ff06282347f_microstructure_clv_trend_40_clv",          # raw CLV level
        "ff06282347f_microstructure_clv_trend_40_slope",        # OLS slope of CLV over 40 bars
        "ff06282347f_microstructure_clv_trend_40_slope_tstat",  # slope / rolling std of CLV (signal strength)
    ],
    "tags": ["microstructure", "price_location", "trend", "clv", "intraday_pressure"],
    "version": "1.0.0",
    "author": "feature-factory ff06282347f",
}

_WINDOW = 40
_LEVEL_SPAN = 10


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN so every code path is covered.
    for col in METADATA["produces"]:
        df[col] = np.nan

    n = len(df)
    if n < 2:
        return df

    hi = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    cl = df["Close"].to_numpy(dtype=float)

    # --- CLV: ((close - low) - (high - close)) / (high - low) ---
    rng = hi - lo
    # Guard zero range
    valid_rng = np.where(rng > 0, rng, np.nan)
    clv = ((cl - lo) - (hi - cl)) / valid_rng  # in [-1, +1]; NaN where flat bar

    # --- EMA of CLV (level) ---
    clv_series = pd.Series(clv, index=df.index)
    clv_ema = clv_series.ewm(span=_LEVEL_SPAN, min_periods=_LEVEL_SPAN // 2, adjust=False).mean()
    df["ff06282347f_microstructure_clv_trend_40_clv"] = clv_ema.values

    # --- OLS slope of CLV over trailing 40-bar window ---
    # Use the closed-form slope: slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # with x = [0,1,...,W-1].  All needed x-sums are constants for fixed W.
    W = _WINDOW
    if n < W:
        return df

    x = np.arange(W, dtype=float)
    sx = x.sum()          # W*(W-1)/2
    sx2 = (x * x).sum()  # W*(W-1)*(2W-1)/6
    denom_x = W * sx2 - sx * sx  # constant; always > 0 for W >= 2

    # Build rolling sum(y) and sum(x*y) using cumsum trick
    # For a window ending at index t: sy = sum(clv[t-W+1..t])
    #                                 sxy = sum(k * clv[t-W+1+k]) for k in 0..W-1
    # We pre-weight by x and then use rolling sum.

    # Weighted version: w_clv[i] = i_within_window * clv[i]
    # We need rolling weighted sum where weights restart at 0,1,...,W-1 each window.
    # Easiest: sliding_window_view
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        clv_padded = clv  # length n
        windows = sliding_window_view(clv_padded, W)  # shape (n-W+1, W)

        sy = np.nansum(windows, axis=1)
        sxy = np.nansum(windows * x, axis=1)
        # count of non-nan per window (for degenerate guard)
        cnt = np.sum(~np.isnan(windows), axis=1)

        slope = np.where(
            (denom_x > 0) & (cnt >= W // 2),
            (W * sxy - sx * sy) / denom_x,
            np.nan,
        )

        # Rolling std of CLV for the same window (t-stat proxy)
        clv_std = np.nanstd(windows, axis=1, ddof=1)
        slope_tstat = np.where(clv_std > 1e-10, slope / clv_std, np.nan)

        out_start = W - 1  # first valid index
        df.iloc[out_start:, df.columns.get_loc("ff06282347f_microstructure_clv_trend_40_slope")] = slope
        df.iloc[out_start:, df.columns.get_loc("ff06282347f_microstructure_clv_trend_40_slope_tstat")] = slope_tstat

    except Exception:
        # Fallback: plain rolling — slower but safe
        slope_arr = np.full(n, np.nan)
        tstat_arr = np.full(n, np.nan)
        for t in range(W - 1, n):
            w = clv[t - W + 1 : t + 1]
            mask = ~np.isnan(w)
            if mask.sum() < W // 2:
                continue
            ww = w[mask]
            xx = x[mask]
            sx_ = xx.sum()
            sy_ = ww.sum()
            sxy_ = (xx * ww).sum()
            sx2_ = (xx * xx).sum()
            dn = len(ww) * sx2_ - sx_ * sx_
            if dn <= 0:
                continue
            s = (len(ww) * sxy_ - sx_ * sy_) / dn
            slope_arr[t] = s
            std_ = ww.std(ddof=1)
            tstat_arr[t] = s / std_ if std_ > 1e-10 else np.nan
        df["ff06282347f_microstructure_clv_trend_40_slope"] = slope_arr
        df["ff06282347f_microstructure_clv_trend_40_slope_tstat"] = tstat_arr

    return df
