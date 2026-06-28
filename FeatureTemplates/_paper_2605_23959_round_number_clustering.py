"""
Round-number clustering / price-level psychology features.

Inspired by the microstructure literature on round-number magnetism and
barrier effects (referenced as the honest per-ticker proxy in the
companion leakage-benchmark paper arxiv 2605.23959).

Prices cluster at psychologically salient round-dollar levels ($1, $5, $10
increments). The distance of the current close to the nearest such level
is a known microstructure signal: stocks near a round-number boundary
exhibit predictable short-term mean-reversion or breakout dynamics.

All computations are causal: each row t uses only Close[0..t] and the
trailing 20-row window of past closes. No future information leaks in.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "_paper_2605_23959_round_number_clustering",
    "description": (
        "Per-ticker price-level psychology: normalized distances to nearest $1/$10 "
        "round-number levels, a proximity flag, a trailing-window cluster fraction, "
        "and above/below-barrier distances to the next $10 boundary."
    ),
    "requires": ["Close"],
    "produces": [
        "rnd_dist_to_round_1",
        "rnd_dist_to_round_10",
        "rnd_near_round_flag",
        "rnd_cluster_frac_20",
        "rnd_dist_above_barrier",
        "rnd_dist_below_barrier",
    ],
    "tags": ["microstructure", "price_level", "experimental"],
    "version": "1.0",
    "author": "paper proxy: arxiv 2605.23959 (round-number magnetism / barrier effect)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute round-number clustering features for one ticker.

    Definitions (all guard Close <= 0 -> NaN):

    rnd_dist_to_round_1
        abs(Close - round(Close / 1) * 1) / Close
        Fractional distance to the nearest whole-dollar level.

    rnd_dist_to_round_10
        abs(Close - round(Close / 10) * 10) / Close
        Fractional distance to the nearest $10 level.

    rnd_near_round_flag
        1.0 if Close is within 0.5% of ANY level defined by $1, $5, or $10
        increments; else 0.0.  NaN when Close <= 0.

    rnd_cluster_frac_20
        Fraction of the trailing 20 closes (rows t-19 .. t) that are each
        within 0.5% of a round level ($1 / $5 / $10).  NaN for the first
        19 rows (min_periods=20).

    rnd_dist_above_barrier
        (next $10 level strictly above Close - Close) / Close
        How far, as a fraction of Close, until the stock hits the ceiling
        of the current $10 band.

    rnd_dist_below_barrier
        (Close - prev $10 level at or below Close) / Close
        How far, as a fraction of Close, below the floor of the current
        $10 band.
    """

    close = df["Close"]

    # -------------------------------------------------------------------------
    # Safety mask: any row with Close <= 0 produces NaN across all features.
    # -------------------------------------------------------------------------
    valid = close > 0

    # -------------------------------------------------------------------------
    # Helper: normalized distance from c to nearest multiple of inc.
    # Returns NaN wherever valid is False or inc <= 0.
    # -------------------------------------------------------------------------
    def _dist_to_nearest(c: pd.Series, inc: float) -> pd.Series:
        nearest = np.round(c.to_numpy() / inc) * inc
        dist = np.abs(c.to_numpy() - nearest)
        result = np.where(valid.to_numpy(), dist / c.to_numpy(), np.nan)
        return pd.Series(result, index=df.index)

    # -------------------------------------------------------------------------
    # rnd_dist_to_round_1 and rnd_dist_to_round_10
    # -------------------------------------------------------------------------
    df["rnd_dist_to_round_1"] = _dist_to_nearest(close, 1.0)
    df["rnd_dist_to_round_10"] = _dist_to_nearest(close, 10.0)

    # -------------------------------------------------------------------------
    # rnd_near_round_flag: within 0.5% of ANY level at $1, $5, or $10 inc.
    # -------------------------------------------------------------------------
    c_arr = close.to_numpy(dtype=np.float64)
    v_arr = valid.to_numpy()

    def _near_any(c_val: float) -> bool:
        """True if c_val is within 0.5% of any $1, $5, or $10 round level."""
        tol = 0.005 * c_val
        for inc in (1.0, 5.0, 10.0):
            nearest = round(c_val / inc) * inc
            if abs(c_val - nearest) <= tol:
                return True
        return False

    near_arr = np.where(
        v_arr,
        np.array([_near_any(c) if v else False for c, v in zip(c_arr, v_arr)],
                 dtype=np.float64),
        np.nan,
    )
    df["rnd_near_round_flag"] = pd.Series(near_arr, index=df.index)

    # -------------------------------------------------------------------------
    # rnd_cluster_frac_20: rolling mean of the near_flag over 20 rows.
    # near_flag is already NaN where close <= 0; rolling mean propagates NaN
    # correctly for the warmup and for any invalid rows inside the window.
    # -------------------------------------------------------------------------
    near_series = df["rnd_near_round_flag"]
    df["rnd_cluster_frac_20"] = near_series.rolling(20, min_periods=20).mean()

    # -------------------------------------------------------------------------
    # rnd_dist_above_barrier: (ceil_$10 - close) / close
    # rnd_dist_below_barrier: (close - floor_$10) / close
    #
    # floor = floor(close / 10) * 10
    # ceil  = floor + 10         (strictly above, by construction)
    #
    # Guard: close <= 0 -> NaN (already handled via v_arr).
    # -------------------------------------------------------------------------
    floor_arr = np.floor(c_arr / 10.0) * 10.0          # e.g. close=14.7 -> 10.0
    ceil_arr = floor_arr + 10.0                          # e.g. 20.0

    # dist_above = (ceil - close) / close
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_above = np.where(v_arr, (ceil_arr - c_arr) / c_arr, np.nan)
        dist_below = np.where(v_arr, (c_arr - floor_arr) / c_arr, np.nan)

    df["rnd_dist_above_barrier"] = pd.Series(dist_above, index=df.index)
    df["rnd_dist_below_barrier"] = pd.Series(dist_below, index=df.index)

    return df
