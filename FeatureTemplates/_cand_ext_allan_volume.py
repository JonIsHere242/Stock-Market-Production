"""
ext_allan_volume — Allan deviation of the log-volume process.

Applies the Allan deviation (tau=5 lags, 80-bar rolling window) to the
first-differences of log(Volume).  The Allan deviation is the square root
of half the mean-squared second-difference at lag tau; it measures
frequency-stability / drift character of a time series.  Applied to
Δlog(Volume) it captures whether trading activity is white-noise-like
(low Allan dev) or is trending / clustering (high Allan dev) — a
DIFFERENT signal axis from xdom_allan_variance which operates on price
returns.

Produces:
  ext_allan_volume_adev   — rolling Allan deviation of Δlog(Volume) at tau=5
  ext_allan_volume_zscore — 20-bar z-score of adev (drift-intensity regime)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_allan_volume",
    "description": (
        "Allan deviation of the log-volume process.  Computes ADEV(tau=5) "
        "over an 80-bar rolling window on first-differences of log(Volume), "
        "measuring whether trading-activity is white-noise-like or "
        "trending/clustering.  Orthogonal to xdom_allan_variance (price "
        "returns); captures VOLUME frequency-stability signal."
    ),
    "requires": ["Volume"],
    "produces": ["ext_allan_volume_adev", "ext_allan_volume_zscore"],
    "tags": ["volume", "allan_deviation", "frequency_stability", "drift"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension of gate-validated winner xdom_allan_variance "
        "(Stock-Market project internal spec, 2026)."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Guard: need at least some valid Volume
    # ------------------------------------------------------------------ #
    vol = df["Volume"].copy().astype(float)
    vol = vol.replace(0, np.nan)

    # ------------------------------------------------------------------ #
    # 2. First-differences of log(Volume)
    #    dv[t] = log(V[t]) - log(V[t-1])  — no lookahead
    # ------------------------------------------------------------------ #
    log_vol = np.log(vol)
    dv = log_vol.diff()          # NaN at t=0; fine

    # ------------------------------------------------------------------ #
    # 3. Rolling Allan deviation at tau=5 over a 80-bar window
    #
    #    Allan deviation formula (for a scalar tau):
    #        ADEV(tau) = sqrt( 0.5 * mean( (y[t] - y[t-tau])^2 ) )
    #    where y[t] are the tau-averaged (block-mean) values of the
    #    underlying series.
    #
    #    For the Δlog-volume process x[t]:
    #      - Form non-overlapping block means of length tau:
    #            Y[k] = mean(x[k*tau : (k+1)*tau])
    #      - ADEV = sqrt(0.5 * mean( (Y[k+1] - Y[k])^2 ))
    #
    #    Vectorised rolling implementation:
    #      Within each 80-bar window we need the sequence of tau-block
    #      averages and their consecutive squared differences.
    #      We use a stride trick: rolling sum of length tau gives block
    #      sums; we offset them by tau steps to form non-overlapping pairs.
    #
    #    Practical efficient approach:
    #      Let S[t] = rolling_sum(dv, tau) / tau  (centred at right edge)
    #      Successive non-overlapping blocks:  A[t] = S[t], B[t] = S[t-tau]
    #      diff = A[t] - B[t]
    #      Over 80-bar window we have floor(80/tau)=16 non-overlapping
    #      block-mean differences.  We rolling-collect their squared values
    #      and take the mean, then sqrt(0.5 * that).
    # ------------------------------------------------------------------ #
    TAU = 5
    WIN = 80
    ZSCORE_WIN = 20

    # Rolling sum of length TAU (right-aligned, minimum periods = TAU)
    block_mean = dv.rolling(window=TAU, min_periods=TAU).mean()

    # Pair consecutive non-overlapping blocks: diff at spacing TAU
    block_diff = block_mean - block_mean.shift(TAU)
    # block_diff is defined only at multiples of TAU from the right edge;
    # intermediate samples are "half-block" pairs — but for the rolling
    # Allan dev we want ONLY non-overlapping pairs.  We zero out the
    # in-between steps so they don't contribute:
    #   positions that ARE block boundaries: index % TAU == 0 (modulo
    #   position in df, not calendar date).  We use the positional index.
    pos = np.arange(len(df), dtype=float)
    boundary_mask = (pos % TAU == 0).astype(float)   # 1 at block edges
    boundary_mask[boundary_mask == 0] = np.nan

    block_diff_sparse = block_diff * pd.Series(boundary_mask, index=df.index)
    sq = block_diff_sparse ** 2

    # Rolling mean of squared diffs over WIN bars; min_periods = 2*TAU
    # (need at least 2 complete blocks to compute one diff)
    roll_mean_sq = sq.rolling(window=WIN, min_periods=2 * TAU).mean()

    # Allan deviation
    adev = np.sqrt(0.5 * roll_mean_sq)

    # Guard inf/nan from upstream
    adev = adev.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #
    # 4. 20-bar z-score of ADEV
    # ------------------------------------------------------------------ #
    roll_mu = adev.rolling(window=ZSCORE_WIN, min_periods=ZSCORE_WIN // 2).mean()
    roll_sd = adev.rolling(window=ZSCORE_WIN, min_periods=ZSCORE_WIN // 2).std()
    roll_sd = roll_sd.replace(0, np.nan)
    zscore = (adev - roll_mu) / roll_sd
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #
    # 5. Assign produced columns — never mutate existing columns
    # ------------------------------------------------------------------ #
    df["ext_allan_volume_adev"] = adev
    df["ext_allan_volume_zscore"] = zscore

    return df
