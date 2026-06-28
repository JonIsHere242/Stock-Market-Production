"""
CUSUM change-point intensity (Page 1954; statistical process control)
Per-ticker causal two-sided CUSUM on standardised log-returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom_cusum_changepoint",
    "description": (
        "Causal two-sided CUSUM (Page 1954) on standardised log-returns. "
        "Maintains S+ = max(0, S+ + z - k) and S- = max(0, S- - z - k) with "
        "k=0.5 and alarm threshold h=5; resets the counter on each alarm. "
        "Produces: xdom_cusum_changepoint_intensity_60 = rolling-60-bar alarm count "
        "(regime-change frequency), xdom_cusum_changepoint_run = bars since last "
        "alarm (run-length between structural breaks), and "
        "xdom_cusum_changepoint_sp = current positive CUSUM accumulator level "
        "(directional drift measure). Per-ticker proxy; faithfully captures the "
        "same persistent-small-drift signal the cross-sectional method targets."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_cusum_changepoint_intensity_60",
        "xdom_cusum_changepoint_run",
        "xdom_cusum_changepoint_sp",
    ],
    "tags": ["cusum", "changepoint", "regime", "cross-domain", "signal-processing"],
    "version": "1.0",
    "author": "CUSUM change-point intensity (Page 1954; statistical process control); cross-domain method transfer (signal processing / econophysics / HRV / DSP)",
}

# CUSUM parameters
_K = 0.5   # allowance / slack (half the detectable shift size in std units)
_H = 5.0   # decision threshold (alarm when S > h)
_Z_WINDOW = 60   # rolling window for z-score standardisation of returns
_INTENSITY_WINDOW = 60  # rolling window for counting alarms


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n < 3:
        df["xdom_cusum_changepoint_intensity_60"] = np.nan
        df["xdom_cusum_changepoint_run"] = np.nan
        df["xdom_cusum_changepoint_sp"] = np.nan
        return df

    # --- log returns ---
    close = df["Close"].to_numpy(dtype=float)
    ret = np.empty(n, dtype=float)
    ret[0] = np.nan
    ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # --- rolling z-score of returns (causal: use past _Z_WINDOW bars) ---
    ret_s = pd.Series(ret)
    roll_mean = ret_s.rolling(_Z_WINDOW, min_periods=10).mean().to_numpy()
    roll_std  = ret_s.rolling(_Z_WINDOW, min_periods=10).std().to_numpy()

    # standardise
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(roll_std > 0, (ret - roll_mean) / roll_std, np.nan)

    # --- causal CUSUM loop ---
    # S+ detects upward drift, S- detects downward drift.
    # An alarm fires when either exceeds h; both are reset to 0 on alarm.
    sp = np.empty(n, dtype=float)
    sm = np.empty(n, dtype=float)
    alarm = np.zeros(n, dtype=np.int8)

    sp_val = 0.0
    sm_val = 0.0

    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            sp[i] = np.nan
            sm[i] = np.nan
            # no alarm; let sp_val / sm_val carry forward (start at 0)
            continue
        sp_val = max(0.0, sp_val + zi - _K)
        sm_val = max(0.0, sm_val - zi - _K)
        if sp_val > _H or sm_val > _H:
            alarm[i] = 1
            sp_val = 0.0
            sm_val = 0.0
        sp[i] = sp_val
        sm[i] = sm_val

    # --- rolling 60-bar alarm count (intensity) ---
    alarm_s = pd.Series(alarm.astype(float))
    intensity = alarm_s.rolling(_INTENSITY_WINDOW, min_periods=1).sum().to_numpy()
    # mask early rows where we had no z-scores yet
    first_valid_z = np.argmax(~np.isnan(z))  # first non-nan index
    intensity[:first_valid_z] = np.nan

    # --- run-length since last alarm (bars since last structural break) ---
    run = np.empty(n, dtype=float)
    bars_since = np.nan
    for i in range(n):
        if np.isnan(z[i]):
            run[i] = np.nan
            continue
        if alarm[i]:
            bars_since = 0.0
        else:
            bars_since = bars_since + 1.0 if not np.isnan(bars_since) else np.nan
        run[i] = bars_since

    df["xdom_cusum_changepoint_intensity_60"] = intensity
    df["xdom_cusum_changepoint_run"] = run
    df["xdom_cusum_changepoint_sp"] = sp

    return df
