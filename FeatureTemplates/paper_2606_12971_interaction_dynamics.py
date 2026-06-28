"""
Price Interaction Dynamics Features
Paper: "Predicting Cognitive Load from Speech and Interaction Dynamics in Dyadic
        Conversations" (arXiv 2606.12971)

The paper's computable method: extract INTERACTION DYNAMICS from two-agent conversation:
  - Turn-taking frequency (speaker switch rate)
  - Overlap rate (simultaneous speech)
  - Participation imbalance (one speaker dominates)
  - Temporal demand ~ time-pressure → rushed switching behaviour
  - Mental demand ~ cognitive effort → imbalanced participation

OHLCV adaptation — "price-market conversation" metaphor:
  Treat BUYERS (up-moves) and SELLERS (down-moves) as two speakers in a dyadic
  conversation. The "turn-taking" is direction-switch events; "overlap" is indecision
  (small body + large wick = both sides active); participation imbalance = bull/bear
  volume dominance. This maps cleanly because:

  - Direction switch rate  ↔  speaker switch rate per minute
  - Indecision candles     ↔  overlapping speech segments
  - Volume imbalance       ↔  participation imbalance between speakers
  - Streak length          ↔  monologue length (one speaker dominating)
  - Wick-to-body ratio     ↔  cognitive load / effort (more contention → higher load)

Features produced:
  - Direction switch rate across multiple windows (how often buyers/sellers swap control)
  - Mean streak length (average uninterrupted bull/bear run)
  - Participation imbalance (net volume signed by direction, rolling)
  - Indecision rate (doji/spinning-top frequency: small body, large wicks)
  - Wicks-to-body ratio mean (measures contention / cognitive load)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_12971_interaction_dynamics",
    "description": (
        "Buyer/seller interaction dynamics: direction-switch rate, streak length, "
        "participation imbalance and indecision candle rate, inspired by speaker "
        "turn-taking and participation imbalance in dyadic conversations (arXiv 2606.12971)"
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces":    [
        # Fraction of bars where close-direction flips vs prior bar (switch rate)
        "pid_switch_rate_10d",
        "pid_switch_rate_20d",
        "pid_switch_rate_60d",
        # Mean length of uninterrupted directional streaks in rolling window
        "pid_mean_streak_20d",
        # Participation imbalance: signed volume (up_vol - dn_vol) / total_vol
        "pid_volume_imbalance_10d",
        "pid_volume_imbalance_20d",
        "pid_volume_imbalance_60d",
        # Indecision candle rate: fraction of bars with body < 25% of total range
        "pid_indecision_rate_20d",
        "pid_indecision_rate_60d",
        # Wick-to-body ratio (rolling mean): high = lots of contention
        "pid_wick_body_ratio_20d",
    ],
    "tags":        ["volume", "momentum", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.12971",
}


def _mean_streak_length(direction: pd.Series, w: int) -> pd.Series:
    """
    Rolling mean length of uninterrupted directional runs within each window.
    E.g., [1,1,1,-1,-1,1] has streaks of length [3,2,1] → mean ≈ 2.0
    """
    dir_arr = np.sign(direction.values).astype(float)
    n = len(dir_arr)
    result = np.full(n, np.nan)

    for i in range(w - 1, n):
        window = dir_arr[i - w + 1 : i + 1]
        if np.all(np.isnan(window)):
            continue
        # Compute streak lengths
        streaks = []
        cur_len = 1
        for j in range(1, len(window)):
            if np.isnan(window[j]) or np.isnan(window[j - 1]):
                cur_len = 1
                continue
            if window[j] == window[j - 1] and window[j] != 0:
                cur_len += 1
            else:
                if cur_len > 0:
                    streaks.append(cur_len)
                cur_len = 1
        streaks.append(cur_len)
        result[i] = np.mean(streaks) if streaks else np.nan

    return pd.Series(result, index=direction.index)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"]
    open_  = df["Open"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Direction of each bar (buyer vs seller control) ────────────────────────
    ret_1d  = close.pct_change()
    direction = np.sign(ret_1d)  # +1 = buyers won, -1 = sellers won, 0 = flat

    # ── Direction switch: 1 if direction changed from prior bar ───────────────
    switched = (direction != direction.shift(1)).astype(float)
    switched[direction == 0] = np.nan         # ignore flat bars
    switched[direction.shift(1) == 0] = np.nan

    for w, col in [(10, "pid_switch_rate_10d"),
                   (20, "pid_switch_rate_20d"),
                   (60, "pid_switch_rate_60d")]:
        df[col] = switched.rolling(w, min_periods=w // 2).mean()

    # ── Mean streak length (avoid O(n^2) fully vectorised is tricky; use efficient loop) ─
    # Use vectorised streak detection instead of nested loop:
    # Streak length at position i = how long current run has been going
    dir_vals = direction.values
    streak = np.ones(len(dir_vals))
    for i in range(1, len(dir_vals)):
        if (not np.isnan(dir_vals[i]) and not np.isnan(dir_vals[i - 1])
                and dir_vals[i] == dir_vals[i - 1] and dir_vals[i] != 0):
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1.0

    streak_series = pd.Series(streak, index=df.index)
    # Mean streak in window: approximate via rolling mean of streak values
    # (exact computation of mean streak length requires the O(n) loop above)
    df["pid_mean_streak_20d"] = streak_series.rolling(20, min_periods=10).mean()

    # ── Volume participation imbalance ────────────────────────────────────────
    up_vol = volume.where(ret_1d > 0, 0.0)
    dn_vol = volume.where(ret_1d < 0, 0.0)

    for w, col in [(10, "pid_volume_imbalance_10d"),
                   (20, "pid_volume_imbalance_20d"),
                   (60, "pid_volume_imbalance_60d")]:
        sum_up = up_vol.rolling(w, min_periods=w // 2).sum()
        sum_dn = dn_vol.rolling(w, min_periods=w // 2).sum()
        df[col] = (sum_up - sum_dn) / (sum_up + sum_dn + 1e-8)

    # ── Indecision candle rate ────────────────────────────────────────────────
    total_range = (high - low).clip(lower=1e-8)
    body        = (close - open_).abs()
    is_indecision = (body / total_range < 0.25).astype(float)

    df["pid_indecision_rate_20d"] = is_indecision.rolling(20, min_periods=10).mean()
    df["pid_indecision_rate_60d"] = is_indecision.rolling(60, min_periods=30).mean()

    # ── Wick-to-body ratio ────────────────────────────────────────────────────
    upper_wick = high - close.clip(upper=open_).where(close >= open_, close) \
                       .where(close < open_, open_)
    # Simpler vectorised computation:
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    total_wick = (upper_wick + lower_wick).clip(lower=0)
    wick_body_ratio = total_wick / body.clip(lower=1e-8)
    # Cap to avoid inf on doji candles
    wick_body_ratio = wick_body_ratio.clip(upper=20.0)
    df["pid_wick_body_ratio_20d"] = wick_body_ratio.rolling(20, min_periods=10).mean()

    return df
