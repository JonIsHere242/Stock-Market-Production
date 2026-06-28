"""
Price acceleration & jerk (kinematics of trend) — xdom2_price_acceleration

Second and third finite differences of a 10-day EMA of log-close price.
Acceleration (d2/dt2) captures curvature of the trend; positive = trend speeding up
(concave up), negative = deceleration / top forming. Jerk (d3/dt3) captures change
in acceleration — a sign flip in jerk often leads reversals by 1-2 bars.

Per-ticker proxy faithful to the kinematic physics transfer idea from the spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_price_acceleration",
    "description": (
        "Kinematics of log-price trend: computes EMA-10 of log(Close), then takes "
        "rolling second differences (acceleration = d2/dt2) and third differences "
        "(jerk = d3/dt3). A rolling 5-day mean smooths each to reduce noise. "
        "Acceleration > 0 = trend speeding up; < 0 = decelerating. Jerk sign-flip "
        "can lead reversals. Per-ticker proxy; economic signal: curvature of momentum "
        "is leading vs momentum level (kinematic transfer from physics)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_price_acceleration_accel10",
        "xdom2_price_acceleration_jerk10",
        "xdom2_price_acceleration_accel10_z",
    ],
    "tags": ["cross-domain", "momentum", "kinematics", "price-structure"],
    "version": "1.0",
    "author": "Spec: Cross-domain / practitioner method transfer (batch 2) — Price acceleration & jerk (kinematics of trend)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : per-ticker DataFrame, ascending by Date, columns include Close.

    Returns
    -------
    df with three new columns appended.
    """
    close = df["Close"].values.astype(np.float64)

    # Guard: log requires positive prices
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.where(close > 0, np.log(close), np.nan)

    log_s = pd.Series(log_close, index=df.index)

    # --- Step 1: 10-day EMA of log-close (smoother than SMA, captures trend shape) ---
    ema10 = log_s.ewm(span=10, adjust=False, min_periods=10).mean()

    # --- Step 2: first difference (velocity) ---
    vel = ema10.diff(1)

    # --- Step 3: second difference (acceleration = d2/dt2) ---
    accel_raw = vel.diff(1)

    # --- Step 4: third difference (jerk = d3/dt3) ---
    jerk_raw = accel_raw.diff(1)

    # --- Step 5: rolling 5-day mean to reduce tick noise ---
    accel10 = accel_raw.rolling(5, min_periods=3).mean()
    jerk10 = jerk_raw.rolling(5, min_periods=3).mean()

    # --- Step 6: z-score of acceleration over trailing 63 bars (normalised signal) ---
    accel_mean = accel10.rolling(63, min_periods=20).mean()
    accel_std = accel10.rolling(63, min_periods=20).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        accel10_z = (accel10 - accel_mean) / accel_std.replace(0, np.nan)

    # Guard against inf (edge case: constant price run → zero std)
    accel10 = accel10.replace([np.inf, -np.inf], np.nan)
    jerk10 = jerk10.replace([np.inf, -np.inf], np.nan)
    accel10_z = accel10_z.replace([np.inf, -np.inf], np.nan)

    df["xdom2_price_acceleration_accel10"] = accel10.values
    df["xdom2_price_acceleration_jerk10"] = jerk10.values
    df["xdom2_price_acceleration_accel10_z"] = accel10_z.values

    return df
