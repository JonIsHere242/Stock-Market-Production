"""
Entropy rate of the return-sign process.

Over a rolling 120-day window, model the daily-return sign sequence as a
2nd-order Markov chain (4 states: {-1,+1}^2) and compute the conditional
entropy rate H(X_t | X_{t-1}, X_{t-2}).  A low entropy rate means the next
sign is highly predictable from the last two; a high rate means near-random.

Produces:
  ext3_entropy_rate_h   : conditional entropy rate (bits, 0–1), rolling 120d
  ext3_entropy_rate_chg : 20d change in entropy rate (momentum of predictability)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import List

METADATA = {
    "name": "ext3_entropy_rate",
    "description": (
        "Conditional entropy rate of the return-sign 2nd-order Markov chain over a "
        "rolling 120-day window.  Low values indicate that the next return sign is "
        "predictable from the prior two signs; high values indicate near-randomness.  "
        "Produces the entropy rate (bits) and its 20-day change.  "
        "Pure OHLCV per-ticker proxy; no cross-sectional information needed."
    ),
    "requires": ["Close"],
    "produces": ["ext3_entropy_rate_h", "ext3_entropy_rate_chg"],
    "tags": ["entropy", "information_theory", "markov", "return_sign", "predictability"],
    "version": "1.0.0",
    "author": "Spec: Round-4 expansion (xdom_allan_variance); implementation by Claude",
}

# ---------------------------------------------------------------------------
# Helper: compute conditional entropy rate from a sign array for one window
# ---------------------------------------------------------------------------

def _cond_entropy_rate(signs: np.ndarray) -> float:
    """
    Given a 1-D array of signs (+1 or -1), estimate H(X_t | X_{t-1}, X_{t-2})
    using empirical counts from a 2nd-order Markov chain.

    States are encoded as (s_{t-2}, s_{t-1}) pairs mapped to integers 0..3:
        (−1,−1)→0  (−1,+1)→1  (+1,−1)→2  (+1,+1)→3

    H = -sum_{ij} P(i,j) * log2 P(j|i)   where i = 2-step context (0..3),
                                                  j = next sign (0 or 1)
    """
    n = len(signs)
    if n < 3:
        return float("nan")

    # Encode signs as 0/1 for indexing
    s = (signs > 0).astype(np.int8)  # +1 -> 1, -1 -> 0

    # Build context index: context = 2*s[t-2] + s[t-1]  (0..3)
    ctx = 2 * s[:-2] + s[1:-1]  # length n-2
    nxt = s[2:]                  # length n-2, the next sign

    # Count transitions: joint[ctx, nxt] – shape (4, 2)
    joint = np.zeros((4, 2), dtype=np.float64)
    for c, x in zip(ctx, nxt):
        joint[c, x] += 1.0

    total = joint.sum()
    if total == 0.0:
        return float("nan")

    # P(ctx, nxt)
    p_joint = joint / total

    # P(ctx) = marginal over nxt
    p_ctx = p_joint.sum(axis=1, keepdims=True)  # (4,1)

    # Conditional P(nxt | ctx)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_cond = np.where(p_ctx > 0, p_joint / p_ctx, 0.0)

    # H = -sum_{c,x} P(c,x) * log2 P(x|c)   (skip zero terms)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_cond = np.where(p_cond > 0, np.log2(p_cond), 0.0)

    h = -np.sum(p_joint * log_cond)
    # Clamp to [0, 1] (theoretical max = 1 bit for binary outcome)
    return float(np.clip(h, 0.0, 1.0))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    WINDOW = 120
    CHANGE_LAG = 20

    n = len(df)

    # Daily return signs: +1 if Close_t >= Close_{t-1}, else -1
    close = df["Close"].to_numpy(dtype=np.float64)

    # sign of daily return; first element is NaN-equivalent (no prior bar)
    ret = np.diff(close, prepend=np.nan)
    signs = np.where(np.isnan(ret) | (ret == 0), np.nan, np.sign(ret))

    # Rolling entropy rate
    h_vals = np.full(n, np.nan, dtype=np.float64)

    for t in range(WINDOW - 1, n):
        window_signs = signs[t - WINDOW + 1 : t + 1]
        # Drop NaNs at the start (first bar has no sign)
        valid = window_signs[~np.isnan(window_signs)]
        if len(valid) >= 10:  # need at least a few transitions
            h_vals[t] = _cond_entropy_rate(valid)

    # 20-day change in entropy rate
    chg_vals = np.full(n, np.nan, dtype=np.float64)
    for t in range(CHANGE_LAG, n):
        if not math.isnan(h_vals[t]) and not math.isnan(h_vals[t - CHANGE_LAG]):
            chg_vals[t] = h_vals[t] - h_vals[t - CHANGE_LAG]

    df["ext3_entropy_rate_h"] = h_vals
    df["ext3_entropy_rate_chg"] = chg_vals

    return df
