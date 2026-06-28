"""
Volatility-conditioned Markov transition features derived from:
  "Inspectable Neural Markov Models for Non-Stationary Time Series" (arXiv 2605.30943).

The paper shows that conditioning on realized volatility (rather than returns) produces
a more internally consistent Markovian structure, reducing Chapman-Kolmogorov discrepancy
by 5.6%. Under high-volatility regimes, transition probabilities homogenize (all states
become equiprobable), which is a regime signal in itself.

Per-ticker proxy: discretize rolling realized volatility into 3 states (low/mid/high),
estimate rolling empirical transition matrices, and extract:
  - Current volatility state (0=low, 1=mid, 2=high)
  - Self-loop probability of current state (persistence)
  - Probability homogeneity index of current state's row (max - min)
  - Rolling CK discrepancy proxy: |P(2-step) - P(1-step)^2| trace
All via rolling windows, no lookahead.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2605_30943_markov_vol_regime",
    "description": (
        "Volatility-conditioned Markov state transition features: rolling 3-state vol "
        "discretization with empirical transition matrix persistence and homogeneity "
        "metrics; proxy for arXiv 2605.30943 neural Markov / CK-discrepancy signal."
    ),
    "requires":    ["Close", "High", "Low"],
    "produces": [
        "mkv_vol_state",
        "mkv_self_loop_prob",
        "mkv_row_homogeneity",
        "mkv_ck_discrepancy",
    ],
    "tags":        ["volatility", "market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2605.30943",
}


def _realized_vol(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  window: int) -> np.ndarray:
    """Rolling Garman-Klass realized vol proxy (annualized)."""
    n = len(close)
    rv = np.full(n, np.nan)
    log_hl = np.log(high / np.maximum(low, 1e-12)) ** 2
    log_cc = np.log(close / np.maximum(np.roll(close, 1), 1e-12)) ** 2
    log_cc[0] = np.nan
    # rolling mean of GK terms
    for i in range(window - 1, n):
        sl_hl = log_hl[i - window + 1: i + 1]
        sl_cc = log_cc[i - window + 1: i + 1]
        valid = ~(np.isnan(sl_hl) | np.isnan(sl_cc))
        if valid.sum() < max(4, window // 2):
            continue
        inner = 0.5 * np.nanmean(sl_hl) - (2 * np.log(2) - 1) * np.nanmean(sl_cc)
        if inner > 0:
            rv[i] = np.sqrt(252 * inner)
    return rv


def _rolling_markov_features(vol_state: np.ndarray, n_states: int,
                              window: int, min_obs: int):
    """
    Rolling estimation of Markov transition features.
    Returns arrays: self_loop, row_homogeneity, ck_discrepancy.
    """
    n = len(vol_state)
    self_loop     = np.full(n, np.nan)
    row_homog     = np.full(n, np.nan)
    ck_disc       = np.full(n, np.nan)

    for i in range(window, n):
        start = i - window
        sl = vol_state[start: i + 1]
        # filter out NaN states (encoded as -1)
        valid_mask = sl >= 0
        sl_v = sl[valid_mask].astype(int)
        if len(sl_v) < min_obs + 1:
            continue

        # Build empirical 1-step transition matrix from consecutive valid pairs
        # (only consecutive pairs where both are valid)
        s_cur = sl[:-1]
        s_nxt = sl[1:]
        both_valid = (s_cur >= 0) & (s_nxt >= 0)
        s_cur = s_cur[both_valid].astype(int)
        s_nxt = s_nxt[both_valid].astype(int)
        if len(s_cur) < min_obs:
            continue

        # Count matrix
        P = np.zeros((n_states, n_states), dtype=np.float64)
        for a, b in zip(s_cur, s_nxt):
            P[a, b] += 1

        # Row-normalize
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P = P / row_sums

        # Current state (most recent valid)
        cur_state_idx = vol_state[i]
        if cur_state_idx < 0:
            continue
        cs = int(cur_state_idx)

        # Self-loop probability P[cs, cs]
        self_loop[i] = P[cs, cs]

        # Row homogeneity: max - min of row (paper: high-vol → homogenization)
        self_loop[i] = P[cs, cs]
        row_homog[i] = P[cs].max() - P[cs].min()

        # CK discrepancy: ||P^2 - P @ P|| as sum of absolute differences
        # P^2 estimated from data, compare with P @ P
        P2 = np.zeros((n_states, n_states), dtype=np.float64)
        if len(s_cur) > 2:
            # Build 2-step transitions from same window
            for k in range(len(s_cur) - 1):
                a0 = s_cur[k]
                a2 = s_nxt[k + 1] if k + 1 < len(s_nxt) else -1
                if a2 >= 0:
                    P2[a0, a2] += 1
            p2_sums = P2.sum(axis=1, keepdims=True)
            p2_sums = np.where(p2_sums == 0, 1.0, p2_sums)
            P2 = P2 / p2_sums
            Pmat2 = P @ P
            ck_disc[i] = np.abs(P2 - Pmat2).sum() / (n_states * n_states)

    return self_loop, row_homog, ck_disc


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    high  = df["High"].values.astype(np.float64)
    low   = df["Low"].values.astype(np.float64)
    close = df["Close"].values.astype(np.float64)

    vol_window   = 20   # rolling window for realized vol estimation
    state_window = 60   # rolling window for Markov estimation
    min_obs      = 20
    n_states     = 3
    expand_window = 60  # use expanding vol stats for thresholding (no lookahead)

    # Step 1: rolling realized vol
    rv = _realized_vol(high, low, close, vol_window)

    # Step 2: discretize into 3 states using EXPANDING quantiles (no lookahead)
    # State 0 = low vol (rv < 33rd pct), 1 = mid, 2 = high (rv > 67th pct)
    vol_state = np.full(n, -1, dtype=np.int8)
    for i in range(expand_window, n):
        past_rv = rv[: i + 1]
        past_rv = past_rv[~np.isnan(past_rv)]
        if len(past_rv) < 10:
            continue
        q33 = np.percentile(past_rv, 33.33)
        q67 = np.percentile(past_rv, 66.67)
        v = rv[i]
        if np.isnan(v):
            continue
        if v < q33:
            vol_state[i] = 0
        elif v < q67:
            vol_state[i] = 1
        else:
            vol_state[i] = 2

    # Step 3: rolling Markov transition features
    self_loop, row_homog, ck_disc = _rolling_markov_features(
        vol_state, n_states, state_window, min_obs
    )

    df["mkv_vol_state"]       = vol_state.astype(np.float64)
    df["mkv_vol_state"]       = df["mkv_vol_state"].where(df["mkv_vol_state"] >= 0)
    df["mkv_self_loop_prob"]  = self_loop
    df["mkv_row_homogeneity"] = row_homog
    df["mkv_ck_discrepancy"]  = ck_disc

    return df
