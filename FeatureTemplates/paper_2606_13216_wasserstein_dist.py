"""
Wasserstein distribution distance features for return distribution shift detection.

Paper: "Layer-Resolved Optimal Transport for Hallucination Detection in NMT and
        Abstractive Summarization" (arXiv 2606.13216)

The paper uses OPTIMAL TRANSPORT (Wasserstein distance) to measure geometric
distance between empirical distributions without supervision. The key insight is
that distributional shift — measured via earth-mover / Wasserstein distance —
signals a qualitative change in the generating process (hallucination in NMT,
regime change in markets).

OHLCV adaptation — "return distribution Wasserstein shift":
  1. Maintain a "reference" distribution = rolling long-window returns (60d)
  2. Maintain a "current" distribution = short-window returns (10d / 20d)
  3. Wasserstein-1 distance between the two empirical CDFs = shift score
     (For 1D: W1 = mean |Q_ref(u_k) - Q_cur(u_k)| over quantile grid)
  4. Also: Wass-to-Uniform = W1 between current distribution and U(min, max)

Vectorised implementation: pre-compute quantiles at Q evenly spaced points using
rolling pandas operations, then average |q_short - q_long| column-wise.

Produces 7 features:
  wot_w1_10v60d      : W1 between 10d and 60d return distributions
  wot_w1_20v60d      : W1 between 20d and 60d return distributions
  wot_w1_5v20d       : W1 between 5d and 20d return distributions (fast shift)
  wot_w1_to_unif_20d : W1 between 20d returns and their uniform reference
  wot_w1_to_unif_10d : W1 between 10d returns and their uniform reference
  wot_shift_rank_40d : rolling percentile rank of wot_w1_10v60d
  wot_asymmetry_20d  : signed shift: median(10d) - median(60d)
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13216_wasserstein_dist",
    "description": (
        "Wasserstein-1 earth-mover distance between rolling return distributions "
        "at different horizons, inspired by layer-resolved optimal transport for "
        "distribution shift detection (arXiv 2606.13216)."
    ),
    "requires":    ["Close"],
    "produces": [
        "wot_w1_10v60d",
        "wot_w1_20v60d",
        "wot_w1_5v20d",
        "wot_w1_to_unif_20d",
        "wot_w1_to_unif_10d",
        "wot_shift_rank_40d",
        "wot_asymmetry_20d",
    ],
    "tags":        ["volatility", "market_regime", "statistical", "experimental"],
    "version":     "1.1",
    "author":      "paper:2606.13216",
}

# Quantile grid (16 points is fast and accurate enough for W1 estimation)
_QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(
        df["Close"].clip(lower=1e-8) / df["Close"].shift(1).clip(lower=1e-8)
    )

    # Pre-compute rolling quantiles for each window width at each quantile level
    # rolling(W).quantile(q) is O(n*W) per (W,q) pair but pandas-vectorised (fast C)
    def rolling_quantiles(series: pd.Series, W: int, min_p: int) -> pd.DataFrame:
        """Returns DataFrame with columns q_0.1 ... q_0.9 from rolling(W).quantile."""
        frames = {}
        for q in _QS:
            frames[q] = series.rolling(W, min_periods=min_p).quantile(q)
        return pd.DataFrame(frames, index=series.index)

    q5  = rolling_quantiles(log_ret, 5,  3)
    q10 = rolling_quantiles(log_ret, 10, 5)
    q20 = rolling_quantiles(log_ret, 20, 10)
    q60 = rolling_quantiles(log_ret, 60, 20)

    # W1 = mean |q_A(u) - q_B(u)| over quantile grid
    def w1_between(qa: pd.DataFrame, qb: pd.DataFrame) -> pd.Series:
        diff = (qa.values - qb.values)  # shape (n, len_QS)
        return pd.Series(np.abs(diff).mean(axis=1), index=qa.index)

    df["wot_w1_10v60d"]  = w1_between(q10, q60)
    df["wot_w1_20v60d"]  = w1_between(q20, q60)
    df["wot_w1_5v20d"]   = w1_between(q5,  q20)

    # Wass-to-Uniform: W1 between rolling quantiles and U(q_min, q_max)
    # q_min ≈ Q(0.05), q_max ≈ Q(0.95) — approximate uniform by its own range
    # U[a,b] quantile at u = a + u*(b-a), so Q_U(_QS) = linspace(a, b, len(_QS))
    def w1_to_uniform(qdf: pd.DataFrame) -> pd.Series:
        """W1 between rolling distribution and uniform over same range."""
        lo = qdf[_QS[0]].values
        hi = qdf[_QS[-1]].values
        span = hi - lo  # (n,)
        # Uniform quantiles at _QS grid
        q_unif = lo[:, None] + _QS[None, :] * span[:, None]   # (n, len_QS)
        diff = qdf.values - q_unif
        result = np.abs(diff).mean(axis=1)
        return pd.Series(result, index=qdf.index)

    df["wot_w1_to_unif_20d"] = w1_to_uniform(q20)
    df["wot_w1_to_unif_10d"] = w1_to_uniform(q10)

    # Signed shift: median(10d) - median(60d)  [Q(0.5) = index of 0.5 in _QS]
    # _QS[4] = 0.5
    med10 = log_ret.rolling(10, min_periods=5).median()
    med60 = log_ret.rolling(60, min_periods=20).median()
    df["wot_asymmetry_20d"] = med10 - med60

    # Rolling rank of 10v60d shift (regime position in distribution-shift space)
    s = df["wot_w1_10v60d"]
    df["wot_shift_rank_40d"] = s.rolling(40, min_periods=10).rank(pct=True)

    return df
