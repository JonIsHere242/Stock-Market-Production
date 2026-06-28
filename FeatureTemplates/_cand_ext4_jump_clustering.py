"""
ext4_jump_clustering — Jump clustering (self-excitation / Hawkes proxy)
From the jump-indicator series (|ret| > 4x local bipower vol), produce:
  - Rolling 90-day lag-1 autocorrelation of the jump indicator (Hawkes-like clustering)
  - Rolling mean gap between consecutive jumps (inverse intensity)
  - Rolling 90-day jump count (activity level)

Pure OHLCV, no lookahead.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ext4_jump_clustering",
    "description": (
        "Per-ticker jump-clustering proxy inspired by Hawkes self-excitation. "
        "Identifies jump days as |log-return| > 4× local bipower variation (90-day rolling), "
        "then computes: (1) lag-1 autocorrelation of the jump indicator over 90 days "
        "(positive = jumps cluster together, Hawkes-like); "
        "(2) mean gap in trading days between consecutive jumps (inverse jump intensity); "
        "(3) rolling 90-day jump count (raw activity). "
        "Cross-sectional ranking is not performed — this is a per-ticker time-series proxy. "
        "All three axes are orthogonal to smooth momentum/vol features."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext4_jump_clustering_ac1",    # lag-1 autocorr of jump indicator (90d)
        "ext4_jump_clustering_gap",    # mean inter-jump gap in trading days (90d)
        "ext4_jump_clustering_count",  # number of jump days in rolling 90d window
    ],
    "tags": ["jumps", "hawkes", "self-excitation", "volatility", "clustering"],
    "version": "1.0",
    "author": "Round-5 expansion (xdom_allan_variance); spec: ext4_jump_clustering",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Log returns
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    log_ret = np.empty(n, dtype=float)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # ------------------------------------------------------------------ #
    # 2. Bipower variation (local volatility robust to jumps)
    #    BV_t = (pi/2) * mean(|r_{i-1}| * |r_i|) over rolling 90-day window
    #    Uses absolute-return pairs (lag-0, lag-1) so no lookahead.
    # ------------------------------------------------------------------ #
    abs_ret = np.abs(log_ret)

    # abs_ret_lag1[t] = |r_{t-1}|
    abs_ret_lag1 = np.empty(n, dtype=float)
    abs_ret_lag1[0] = np.nan
    abs_ret_lag1[1:] = abs_ret[:-1]

    bv_pair = abs_ret * abs_ret_lag1  # product of consecutive |returns|

    WINDOW = 90
    BV_SCALE = np.pi / 2.0

    # rolling mean of bv_pair then scale → local BV
    bv_series = pd.Series(bv_pair).rolling(WINDOW, min_periods=20).mean() * BV_SCALE
    bv_arr = bv_series.to_numpy(dtype=float)

    # per-day bipower vol (annualised sqrt not needed; we compare |ret| to 4*sqrt(bv))
    local_bv_vol = np.sqrt(np.where(bv_arr > 0, bv_arr, np.nan))

    # ------------------------------------------------------------------ #
    # 3. Jump indicator: 1 if |ret| > 4 * local_bv_vol else 0
    # ------------------------------------------------------------------ #
    jump = np.where(
        np.isfinite(log_ret) & np.isfinite(local_bv_vol),
        (np.abs(log_ret) > 4.0 * local_bv_vol).astype(float),
        np.nan,
    )

    jump_s = pd.Series(jump, index=df.index)

    # ------------------------------------------------------------------ #
    # 4. Feature A: rolling 90-day lag-1 autocorrelation of jump indicator
    #    corr(jump[t], jump[t-1]) over past 90 obs
    # ------------------------------------------------------------------ #
    jump_lag1 = jump_s.shift(1)

    def _roll_corr(x: pd.Series, y: pd.Series, window: int, min_p: int) -> pd.Series:
        """Rolling Pearson correlation between two aligned series."""
        # Use expanding/rolling cov/var decomposition — fully vectorised
        roll = pd.concat([x, y], axis=1)
        roll.columns = ["x", "y"]

        x_mean = roll["x"].rolling(window, min_periods=min_p).mean()
        y_mean = roll["y"].rolling(window, min_periods=min_p).mean()
        x_std  = roll["x"].rolling(window, min_periods=min_p).std(ddof=0)
        y_std  = roll["y"].rolling(window, min_periods=min_p).std(ddof=0)

        cov = (
            (roll["x"] - x_mean) * (roll["y"] - y_mean)
        ).rolling(window, min_periods=min_p).mean()

        denom = x_std * y_std
        corr = np.where(denom > 0, cov / denom, np.nan)
        return pd.Series(corr, index=x.index)

    ac1 = _roll_corr(jump_s, jump_lag1, window=WINDOW, min_p=20)

    # ------------------------------------------------------------------ #
    # 5. Feature B: mean inter-jump gap (trading days) over rolling 90-day
    #    window.  "Gap" = days between consecutive jump days.
    #    Proxy: within each 90-day window count jumps; gap = 90 / max(count,1).
    #    This is O(n) rolling, no Python row loops.
    # ------------------------------------------------------------------ #
    jump_count_90 = jump_s.rolling(WINDOW, min_periods=20).sum()
    # mean gap = window / count; if count == 0 → nan
    mean_gap = np.where(
        jump_count_90 > 0,
        WINDOW / jump_count_90,
        np.nan,
    )
    mean_gap_s = pd.Series(mean_gap, index=df.index)

    # ------------------------------------------------------------------ #
    # 6. Feature C: raw jump count in rolling 90-day window (activity level)
    # ------------------------------------------------------------------ #
    jump_count_s = jump_count_90

    # ------------------------------------------------------------------ #
    # 7. Attach produced columns — never emit inf
    # ------------------------------------------------------------------ #
    df["ext4_jump_clustering_ac1"]   = np.where(np.isfinite(ac1),        ac1,          np.nan)
    df["ext4_jump_clustering_gap"]   = np.where(np.isfinite(mean_gap_s), mean_gap_s,   np.nan)
    df["ext4_jump_clustering_count"] = np.where(np.isfinite(jump_count_s), jump_count_s, np.nan)

    return df
