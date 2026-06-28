"""
xdom_zero_crossing — Mean/zero-crossing rate & slope-sign changes (DSP activity)

Rolling 30-day zero-crossing rate of detrended price and slope-sign-change rate.
High values indicate choppy, mean-reverting price action (high DSP activity).
Low values indicate trending, directionally persistent price action.

Per-ticker proxy: faithful to the spec — no cross-sectional data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_zero_crossing",
    "description": (
        "Mean/zero-crossing rate and slope-sign-change rate computed on rolling 30-day "
        "windows of Close prices. xdom_zero_crossing_zcr: fraction of bars in the window "
        "where the linearly detrended price crosses its window mean (zero after detrend). "
        "xdom_zero_crossing_sscr: fraction of bars where the sign of the 1-day price change "
        "flips relative to the prior day. xdom_zero_crossing_chop: composite choppiness index "
        "= (zcr + sscr) / 2. High values = choppy mean-reverting tape; low = trending tape. "
        "Classic time-domain DSP activity features applied per-ticker. No cross-sectional "
        "component; fully per-stock."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_zero_crossing_zcr",    # zero-crossing rate of detrended price, 30d window
        "xdom_zero_crossing_sscr",   # slope-sign-change rate (1d change sign flips), 30d window
        "xdom_zero_crossing_chop",   # composite = (zcr + sscr) / 2
    ],
    "tags": ["dsp", "cross-domain", "activity", "mean-reversion", "choppiness"],
    "version": "1.0.0",
    "author": (
        "Spec: Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Mean/zero-crossing rate & slope-sign changes (DSP activity). "
        "Implementation: Claude Sonnet 4.6."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    window = 30

    close = df["Close"].to_numpy(dtype=np.float64)
    n = len(close)

    zcr_arr = np.full(n, np.nan)
    sscr_arr = np.full(n, np.nan)

    # 1-day price changes; need at least 2 bars to compute sign
    # sign of change: +1 up, -1 down, 0 flat
    # We treat 0 as same-sign (not a flip) to avoid spurious flips on flat days
    changes = np.diff(close, prepend=np.nan)  # length n; index 0 = NaN

    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)  # current window of length `window`
        w_close = close[sl]  # shape (window,)

        # ---- Zero-crossing rate (ZCR) of linearly detrended price ----
        # Detrend: remove linear trend fit to the window
        x = np.arange(window, dtype=np.float64)
        # Manually compute linear regression coefficients to avoid sklearn
        x_mean = (window - 1) / 2.0
        y_mean = np.mean(w_close)
        ss_xx = np.sum((x - x_mean) ** 2)
        if ss_xx == 0.0:
            detrended = w_close - y_mean
        else:
            slope = np.sum((x - x_mean) * (w_close - y_mean)) / ss_xx
            intercept = y_mean - slope * x_mean
            trend = slope * x + intercept
            detrended = w_close - trend

        # After detrending the window mean ≈ 0; count sign crossings
        # A crossing occurs where consecutive samples straddle zero
        # (sign changes from positive to negative or vice versa)
        signs = np.sign(detrended)
        # Replace zeros with the previous non-zero sign to avoid phantom crossings
        for k in range(1, window):
            if signs[k] == 0:
                signs[k] = signs[k - 1]
        sign_diff = np.diff(signs)  # non-zero where sign flips; length window-1
        cross_count = np.count_nonzero(sign_diff)
        zcr_arr[i] = cross_count / (window - 1) if window > 1 else np.nan

        # ---- Slope-sign-change rate (SSCR) ----
        # Fraction of bars where the sign of the 1d price change flips
        # Use the changes slice for this window
        w_changes = changes[sl]  # length window; first element may be NaN
        # Only use indices 1..window-1 (we need consecutive pairs)
        c_signs = np.sign(w_changes[1:])   # length window-1; index 0 = change[i-window+2]
        c_signs_prev = np.sign(w_changes[:-1])  # length window-1; change at prior bar
        # Ignore bars where either change is zero (flat) — treat as no flip
        valid = (c_signs != 0) & (c_signs_prev != 0)
        n_valid = np.sum(valid)
        if n_valid == 0:
            sscr_arr[i] = np.nan
        else:
            flip_count = np.sum((c_signs[valid] != c_signs_prev[valid]))
            sscr_arr[i] = flip_count / n_valid

    df["xdom_zero_crossing_zcr"] = zcr_arr
    df["xdom_zero_crossing_sscr"] = sscr_arr

    # Composite choppiness: mean of zcr and sscr; NaN if either is NaN
    chop = np.where(
        np.isnan(zcr_arr) | np.isnan(sscr_arr),
        np.nan,
        (zcr_arr + sscr_arr) / 2.0,
    )
    df["xdom_zero_crossing_chop"] = chop

    return df
