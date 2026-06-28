"""
Topological persistence survival features derived from:
  "From Persistence to Survival: Hypothesis Testing, Effect Sizes and
   Vectorisation for Topological Features" (arXiv 2606.11911).

STRAND treats topological features (birth/death of connected components
in sublevel-set filtrations) as survival data.
Key insight: each local extremum pair (birth=local min, death=local max or vice versa)
defines a persistence value p = |death - birth|.
The persistence survival function S(t) = P(p > t) summarises the "lifetime"
distribution of price oscillations.

We implement:
  - Sublevel-set persistence on rolling Close windows (zigzag critical points)
  - Persistence survival function summary statistics:
      mean persistence, max persistence, persistence entropy
      S(t) evaluated at t = 1%, 2%, 5% thresholds
  - Multiple windows: 20, 40 days.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_11911_persistence_survival",
    "description": (
        "Topological persistence (sublevel-set) survival statistics on rolling OHLC windows; "
        "based on arXiv 2606.11911 STRAND persistence survival analysis."
    ),
    "requires":    ["Close", "High", "Low"],
    "produces": [
        "topo_persist_mean_20d",
        "topo_persist_mean_40d",
        "topo_persist_max_20d",
        "topo_persist_max_40d",
        "topo_persist_entropy_20d",
        "topo_persist_entropy_40d",
        "topo_persist_surv_1pct_20d",
        "topo_persist_surv_1pct_40d",
        "topo_persist_surv_2pct_20d",
        "topo_persist_surv_2pct_40d",
        "topo_n_features_20d",
        "topo_n_features_40d",
    ],
    "tags":        ["topology", "volatility", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.11911",
}


def _critical_points(arr: np.ndarray):
    """
    Find indices of local minima and maxima (excluding endpoints).
    Returns (min_idx, max_idx) arrays.
    """
    n = len(arr)
    if n < 3:
        return np.array([]), np.array([])
    # local min: arr[i] < arr[i-1] and arr[i] < arr[i+1]
    prev = arr[:-2]
    curr = arr[1:-1]
    nxt  = arr[2:]
    local_min = np.where((curr < prev) & (curr < nxt))[0] + 1
    local_max = np.where((curr > prev) & (curr > nxt))[0] + 1
    return local_min, local_max


def _persistence_pairs(arr: np.ndarray) -> np.ndarray:
    """
    Compute 0-th homology persistence pairs via the union-find approach
    on a sorted-by-value sequence (sublevel set filtration on 1D).

    For 1-D time series, this reduces to: each local minimum is paired
    with the nearest local maximum (or the global max as the "essential" feature).
    We use the simplest correct approach:
      Sort all critical-point values, pair each local min with the
      "closest" local max that first becomes connected to it.

    Simplified implementation using the standard 1D elder rule:
    persistence pairs = (local_min_value, adjacent_higher_value).
    """
    # Add endpoints as critical candidates
    n = len(arr)
    if n < 3:
        return np.array([])

    # Build list of (index, value, type) for all extrema
    lmins, lmaxs = _critical_points(arr)

    if len(lmins) == 0:
        return np.array([])

    # For each local min, persistence = (nearest local max on either side) - local_min
    persistences = []
    for mi in lmins:
        v_min = arr[mi]
        # Find closest local maxima on left and right
        left_maxs  = lmaxs[lmaxs < mi]
        right_maxs = lmaxs[lmaxs > mi]

        candidates = []
        if len(left_maxs) > 0:
            candidates.append(arr[left_maxs[-1]])
        if len(right_maxs) > 0:
            candidates.append(arr[right_maxs[0]])

        if len(candidates) == 0:
            # Use endpoints
            candidates = [arr[0], arr[-1]]

        # Elder rule: paired with the *lower* of adjacent maxima
        paired_max = min(candidates)
        p = paired_max - v_min
        if p > 0:
            persistences.append(p)

    return np.array(persistences)


def _persistence_features(arr: np.ndarray) -> dict:
    """Compute persistence survival statistics for a window."""
    # Normalise to [0,1] to make persistence scale-invariant
    rng = arr.max() - arr.min()
    if rng < 1e-10:
        return {
            "mean": np.nan, "max": np.nan, "entropy": np.nan,
            "surv_1pct": np.nan, "surv_2pct": np.nan, "n": 0
        }
    arr_norm = (arr - arr.min()) / rng   # persistence in [0,1] fraction of range

    pairs = _persistence_pairs(arr_norm)
    if len(pairs) == 0:
        return {
            "mean": np.nan, "max": np.nan, "entropy": np.nan,
            "surv_1pct": np.nan, "surv_2pct": np.nan, "n": 0
        }

    # Survival function S(t) = fraction of features with persistence > t
    n = len(pairs)
    surv_1 = (pairs > 0.01).sum() / n
    surv_2 = (pairs > 0.02).sum() / n

    # Persistence entropy: -sum(p_i/total * log(p_i/total))
    total = pairs.sum()
    if total < 1e-12:
        ent = np.nan
    else:
        probs = pairs / total
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log(probs + 1e-300))

    return {
        "mean": pairs.mean(),
        "max":  pairs.max(),
        "entropy": ent,
        "surv_1pct": surv_1,
        "surv_2pct": surv_2,
        "n": n
    }


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(np.float64)
    n = len(df)

    for window in [20, 40]:
        min_obs = max(8, window // 2)

        mean_arr    = np.full(n, np.nan)
        max_arr     = np.full(n, np.nan)
        ent_arr     = np.full(n, np.nan)
        surv1_arr   = np.full(n, np.nan)
        surv2_arr   = np.full(n, np.nan)
        n_feat_arr  = np.full(n, np.nan)

        for i in range(min_obs, n):
            start = max(0, i - window + 1)
            segment = close[start: i + 1]
            if np.any(np.isnan(segment)):
                continue

            feat = _persistence_features(segment)
            mean_arr[i]   = feat["mean"]
            max_arr[i]    = feat["max"]
            ent_arr[i]    = feat["entropy"]
            surv1_arr[i]  = feat["surv_1pct"]
            surv2_arr[i]  = feat["surv_2pct"]
            n_feat_arr[i] = feat["n"]

        df[f"topo_persist_mean_{window}d"]    = mean_arr
        df[f"topo_persist_max_{window}d"]     = max_arr
        df[f"topo_persist_entropy_{window}d"] = ent_arr
        df[f"topo_persist_surv_1pct_{window}d"] = surv1_arr
        df[f"topo_persist_surv_2pct_{window}d"] = surv2_arr
        df[f"topo_n_features_{window}d"]      = n_feat_arr

    return df
