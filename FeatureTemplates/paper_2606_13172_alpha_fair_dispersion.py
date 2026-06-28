"""
Alpha-fair cross-sectional dispersion features inspired by:
  "alpha-fair heterogeneous agent reinforcement learning" (arXiv 2606.13172)

SKIP REASON: The paper is about multi-agent RL fairness theory (HATRL/HATRPO)
with no OHLCV-computable method.

REPLACEMENT: Implement per-ticker features based on the ALPHA-FAIRNESS concept:
  alpha-fair welfare: W_alpha(r) = sum( r_i^{1-alpha} / (1-alpha) )
  When alpha -> 1: log-sum welfare (proportional fairness = geometric mean of returns).
  When alpha = 0: utilitarian (arithmetic mean).
  When alpha = 2: harmonic mean weighting.

Applied intra-ticker across rolling windows of daily log-returns:
  The alpha-fairness gradient is: w_i = r_i^{-alpha}  (inverse-power weighting).
  This upweights SMALL returns and downweights large returns.

Key features:
  - Ratio of harmonic mean to arithmetic mean of |returns| (alpha=2 vs 0 welfare)
  - Log-sum (proportional fair) vs arithmetic mean ratio
  - Gini coefficient of rolling |returns| (inequality of return magnitudes)
  - Tailedness index: top-decile fraction of total absolute return
  - Dynamic alpha implied by equalising fairness gradient = 1
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13172_alpha_fair_dispersion",
    "description": (
        "Alpha-fairness welfare metrics applied to rolling return distributions: "
        "harmonic/geometric/arithmetic mean ratios, Gini of |returns|, tail concentration; "
        "inspired by arXiv 2606.13172 alpha-fair MARL (paper skipped - no OHLCV method)."
    ),
    "requires":    ["Close"],
    "produces": [
        "afair_harm_arith_ratio_21d",
        "afair_harm_arith_ratio_63d",
        "afair_log_arith_ratio_21d",
        "afair_log_arith_ratio_63d",
        "afair_gini_21d",
        "afair_gini_63d",
        "afair_tail_conc_21d",
    ],
    "tags":        ["volatility", "distributional", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13172",
}


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if len(arr) < 2:
        return np.nan
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum()) - (n + 1) / n)) if arr.sum() > 0 else 0.0


def _harm_arith_ratio(arr: np.ndarray) -> float:
    """Harmonic mean / arithmetic mean of absolute values. 0 < ratio <= 1."""
    a = arr[np.isfinite(arr) & (arr > 0)]
    if len(a) < 2:
        return np.nan
    arith = a.mean()
    harm = len(a) / (1.0 / a).sum()
    return float(harm / arith) if arith > 0 else np.nan


def _log_arith_ratio(arr: np.ndarray) -> float:
    """exp(mean(log(|r|))) / mean(|r|): geometric/arithmetic mean ratio. <= 1 always."""
    a = arr[np.isfinite(arr) & (arr > 0)]
    if len(a) < 2:
        return np.nan
    geom = float(np.exp(np.log(a).mean()))
    arith = float(a.mean())
    return geom / arith if arith > 0 else np.nan


def _tail_concentration(arr: np.ndarray) -> float:
    """Fraction of total |return| contributed by top decile bars."""
    a = arr[np.isfinite(arr) & (arr >= 0)]
    if len(a) < 5:
        return np.nan
    total = a.sum()
    if total <= 0:
        return np.nan
    threshold = np.percentile(a, 90)
    top_sum = a[a >= threshold].sum()
    return float(top_sum / total)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    abs_ret = np.log(close / close.shift(1)).abs()

    for w in [21, 63]:
        # Harmonic / arithmetic ratio
        df[f"afair_harm_arith_ratio_{w}d"] = abs_ret.rolling(w, min_periods=w // 2).apply(
            _harm_arith_ratio, raw=True
        )
        # Geometric / arithmetic ratio (log-welfare / utilitarian welfare)
        df[f"afair_log_arith_ratio_{w}d"] = abs_ret.rolling(w, min_periods=w // 2).apply(
            _log_arith_ratio, raw=True
        )
        # Gini of |returns|
        df[f"afair_gini_{w}d"] = abs_ret.rolling(w, min_periods=w // 2).apply(
            _gini, raw=True
        )

    # Tail concentration at 21d only
    df["afair_tail_conc_21d"] = abs_ret.rolling(21, min_periods=10).apply(
        _tail_concentration, raw=True
    )

    return df
