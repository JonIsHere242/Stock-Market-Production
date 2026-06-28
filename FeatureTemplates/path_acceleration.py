"""
path_acceleration.py — return acceleration / convexity of the price path (Tier-2).

Momentum that is ACCELERATING (the short-horizon return pulling away from the
long-horizon return, a positive second-derivative of cumulative log-return)
discriminates continuation winners from stalling ones. This is the "convexity"
idea behind frog-in-the-pan refinements and the momentum-of-momentum literature
(e.g. Gutierrez & Kelley 2008 on the dynamics of momentum), implemented purely
from one ticker's log-price path. All trailing-only & vectorised.

Produces:
  path_mom_diff_10_40    10d log-return minus 40d/4 normalised log-return
                         (short momentum pulling away from long) — the core
                         acceleration signal.
  path_accel_2d          2nd difference of cumulative log-return (discrete
                         acceleration of price), smoothed over 5d.
  path_curvature_20      curvature of the recent log-price path: deviation of
                         today's log-price above/below the straight line joining
                         the points 20d ago and today (convex vs concave path).
  path_mom_ratio_5_20    5d vs 20d annualised-rate momentum ratio (rate is
                         speeding up if >1).
  path_jerk_10           change in 10d momentum over the last 10d (3rd-order:
                         is the acceleration itself rising).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name":        "path_acceleration",
    "description": "Return acceleration / path convexity: short-vs-long momentum spread, 2nd-diff price acceleration, path curvature vs chord, momentum-rate ratio, and momentum jerk.",
    "requires":    ["Close"],
    "produces":    [
        "path_mom_diff_10_40",
        "path_accel_2d",
        "path_curvature_20",
        "path_mom_ratio_5_20",
        "path_jerk_10",
    ],
    "tags":        ["momentum", "trend", "tail", "experimental"],
    "version":     "1.0",
    "author":      "Tier-2 lit build (momentum-of-momentum / convexity)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].clip(lower=1e-8)
    logp = np.log(close)

    # 10d and 40d cumulative log-returns.
    r10 = logp - logp.shift(10)
    r40 = logp - logp.shift(40)
    # Short momentum minus PER-PERIOD-matched long momentum (40d/4 == per-10d rate).
    df["path_mom_diff_10_40"] = (r10 - r40 / 4.0).clip(-1.0, 1.0).values

    # 2nd difference of cumulative log-return == discrete acceleration of log-price.
    # diff() of log-price is the daily return; diff() again is the acceleration.
    accel = logp.diff().diff()
    df["path_accel_2d"] = accel.rolling(5, min_periods=3).mean().clip(-0.5, 0.5).values

    # Path curvature over 20d: today's log-price minus the midpoint-chord value.
    # Straight line from logp[t-20] to logp[t]; midpoint check at t-10.
    # curvature = logp[t-10] - 0.5*(logp[t] + logp[t-20]); >0 = path bowed UP
    # in the middle (concave), <0 = bowed down (convex toward acceleration).
    # We report the negative so that positive = accelerating/convex-up path.
    mid = logp.shift(10)
    chord = 0.5 * (logp + logp.shift(20))
    df["path_curvature_20"] = (chord - mid).clip(-0.5, 0.5).values

    # Momentum RATE ratio: 5d-rate vs 20d-rate. >1 == speeding up.
    r5 = logp - logp.shift(5)
    rate5 = r5 / 5.0
    rate20 = (logp - logp.shift(20)) / 20.0
    # Use signed magnitude ratio guarded against tiny denominators.
    denom = rate20.abs().clip(lower=1e-4)
    df["path_mom_ratio_5_20"] = (rate5 / denom).clip(-10.0, 10.0).values

    # Jerk: change in 10d momentum over the last 10d (is acceleration itself rising).
    df["path_jerk_10"] = (r10 - r10.shift(10)).clip(-1.0, 1.0).values

    return df
