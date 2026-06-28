"""
Count-based roughness / total-variation indices.

Rolling path-length over net-displacement (efficiency complement) and total-
variation acceleration.  Distinct from Kaufman ER in that it normalises by
the *absolute* end-to-end displacement rather than the signed one, making it
a pure chop/trend discriminator: high = choppy, low = trending.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "xdom_count_based_roughness",
    "description": (
        "Per-ticker rolling roughness indices derived from price total-variation "
        "relative to net displacement.  "
        "`xdom_count_based_roughness_40`: 40-day efficiency complement = "
        "(sum|Δclose|) / |close[t] - close[t-40]| - 1.  "
        "High values → choppy/mean-reverting; low → trending.  "
        "`xdom_count_based_roughness_tv_accel`: ratio of the most-recent 20-day "
        "total variation to the prior 20-day total variation; >1 means volatility "
        "is accelerating.  "
        "`xdom_count_based_roughness_zscore`: 63-day z-score of the 40-day "
        "roughness, centred per-ticker so it is comparable across regimes.  "
        "All values are purely per-ticker (no cross-sectional rank step).  "
        "Proxy for: cross-domain variation/roughness indices from signal "
        "processing / econophysics / HRV literature."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_count_based_roughness_40",
        "xdom_count_based_roughness_tv_accel",
        "xdom_count_based_roughness_zscore",
    ],
    "tags": ["roughness", "total_variation", "efficiency", "chop", "trend", "cross_domain"],
    "version": "1.0.0",
    "author": (
        "Cross-domain method transfer (signal processing / econophysics / HRV / DSP); "
        "Variation/roughness indices (total variation vs net move)"
    ),
}

_WIN = 40          # main roughness window
_HALF = _WIN // 2  # 20-day half for acceleration
_Z_WIN = 63        # z-score lookback


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    roughness_40 = np.full(n, np.nan)
    tv_accel = np.full(n, np.nan)

    # --- 40-day roughness (efficiency complement) ---
    # roughness = sum(|Δclose|, 40) / |close[t] - close[t-40]| - 1
    # Clamped against zero denominator -> NaN
    if n >= _WIN + 1:
        abs_diff = np.abs(np.diff(close))      # length n-1

        # rolling sum of |Δclose| over _WIN steps (requires _WIN price-diffs)
        # bar t uses diffs[t-_WIN : t]  (indices of abs_diff relative to close)
        for t in range(_WIN, n):
            tv = abs_diff[t - _WIN: t].sum()          # sum of _WIN daily moves
            net = abs(close[t] - close[t - _WIN])
            if net > 0.0:
                roughness_40[t] = tv / net - 1.0
            # else: zero displacement → undefined, leave NaN

    # Vectorised alternative for the inner loop (avoids Python loop overhead):
    # use cumsum trick
    if n >= _WIN + 1:
        cs = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(close)))])
        rolling_tv = cs[_WIN:] - cs[:n - _WIN]        # length n - _WIN
        net_disp = np.abs(close[_WIN:] - close[:n - _WIN])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(net_disp > 0.0, rolling_tv / net_disp - 1.0, np.nan)
        roughness_40[_WIN:] = r

    # --- 20/20 total-variation acceleration ---
    # tv_accel = TV(last 20 days) / TV(prior 20 days)
    # Bar t: last-20 = sum|Δclose|[t-20:t], prior-20 = sum|Δclose|[t-40:t-20]
    if n >= _WIN + 1:
        cs_inner = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(close)))])
        # last-20 TV at bar t  (t >= _WIN)
        tv_last20 = cs_inner[_WIN:] - cs_inner[_HALF: n - _HALF]   # length n-_WIN
        tv_prev20 = cs_inner[_HALF: n - _HALF] - cs_inner[:n - _WIN]
        with np.errstate(divide="ignore", invalid="ignore"):
            accel = np.where(tv_prev20 > 0.0, tv_last20 / tv_prev20, np.nan)
        tv_accel[_WIN:] = accel

    # --- 63-day z-score of roughness_40 ---
    roughness_s = pd.Series(roughness_40)
    roll_mean = roughness_s.rolling(_Z_WIN, min_periods=_Z_WIN // 2).mean()
    roll_std = roughness_s.rolling(_Z_WIN, min_periods=_Z_WIN // 2).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(roll_std > 0.0, (roughness_s - roll_mean) / roll_std, np.nan)

    df["xdom_count_based_roughness_40"] = roughness_40
    df["xdom_count_based_roughness_tv_accel"] = tv_accel
    df["xdom_count_based_roughness_zscore"] = zscore

    return df
