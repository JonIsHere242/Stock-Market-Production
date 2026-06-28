"""
Trajectory-based Quantization Sensitivity / Lyapunov-style perturbation features.

Paper: "Quantizing Time-Series Models As Dynamical Systems: Trajectory-Based
        Quantization Sensitivity Score (TQS)" (arXiv 2606.13300)

The paper reframes PTQ through DYNAMICAL-SYSTEMS stability: it models a network's
rollout as a discrete-time system x_{t+1} = f(x_t) + epsilon, measures how
quantization-induced perturbations PROPAGATE AND AMPLIFY over the rollout horizon,
and assigns a sensitivity score based on this amplification.

OHLCV adaptation — "local neighbourhood divergence / finite-time Lyapunov":

The core method: given a trajectory x_0, x_1, ..., x_t in state space,
find the k NEAREST historical states (neighbours), and measure how those
neighbours' SUBSEQUENT TRAJECTORIES diverged from x_t's trajectory.
The divergence rate = finite-time Lyapunov exponent (FLE).

  state_t = [log_ret_1d, log_ret_5d_cumul, realised_vol_5d, hl_range_norm]
  For each bar t:
    1. Find k=5 nearest bars in state_t space within past 252 bars
    2. For each neighbour n_i at lag d_i days ago, measure how much the
       SUBSEQUENT 5-day return differed: |ret_{n_i+1..5} - ret_{t+1..5}|
       BUT since we can't use future returns, we instead look at what HAPPENED
       AFTER neighbours in the HISTORICAL record (pure lookback).
    3. FLE_t = mean log(|outcome_divergence| / |state_distance|) across neighbours
    4. High FLE = chaotic (small state distance → big outcome divergence)
       Low FLE = stable (similar states → similar outcomes)

  Additional: sensitivity score = local variance of forward outcomes in neighbourhood

Produces 7 features:
  tqs_fle_k5_252d      : finite-time Lyapunov exponent, k=5 neighbours, 252d history
  tqs_fle_k10_252d     : same with k=10 neighbours
  tqs_nbr_outcome_var  : variance of 5-day forward returns among k=5 neighbours
  tqs_nbr_dist_mean    : mean state-space distance to k=5 nearest neighbours
  tqs_sensitivity_rank_60d: rolling percentile rank of fle_k5 (chaoticity regime)
  tqs_state_density_252d  : fraction of historical states within eps=0.5*median_dist ball
  tqs_knn_mean_outcome    : k=5 NN distance-weighted mean of neighbours' 5d fwd returns
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13300_tqs_lyapunov",
    "description": (
        "Nearest-neighbour finite-time Lyapunov exponents and local trajectory "
        "sensitivity in OHLCV state space, derived from the TQS dynamical-systems "
        "stability framework (arXiv 2606.13300)."
    ),
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "tqs_fle_k5_252d",
        "tqs_fle_k10_252d",
        "tqs_nbr_outcome_var",
        "tqs_nbr_dist_mean",
        "tqs_sensitivity_rank_60d",
        "tqs_state_density_252d",
        "tqs_knn_mean_outcome",
    ],
    "tags":        ["volatility", "market_regime", "statistical", "experimental"],
    "version":     "2.0",
    "author":      "paper:2606.13300",
}

_HIST = 252   # history window
_K5   = 5
_K10  = 10
_EPS_FRAC = 0.15  # epsilon = 15th percentile of pairwise distances


def _build_state(log_ret: np.ndarray, hl_norm: np.ndarray, vol_norm: np.ndarray) -> np.ndarray:
    """
    4D state vector at each bar:
    [cumul_ret_1d, cumul_ret_5d, hl_norm, vol_norm]
    Returned as (n, 4) array.
    """
    n = len(log_ret)
    cumul5 = np.full(n, np.nan)
    for i in range(4, n):
        cumul5[i] = log_ret[i-4:i+1].sum()

    state = np.column_stack([
        log_ret,
        cumul5,
        hl_norm,
        vol_norm,
    ])
    return state


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].clip(lower=1e-8)
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    log_ret = np.log(close / close.shift(1)).fillna(0.0).values.astype(np.float64)

    hl_range = (high - low).values / close.values
    hl_mean = pd.Series(hl_range).rolling(20, min_periods=5).mean().fillna(
        pd.Series(hl_range).expanding().mean()
    ).values.clip(min=1e-8)
    hl_norm = hl_range / hl_mean

    vol_arr  = volume.values.astype(np.float64)
    vol_mean = pd.Series(vol_arr).rolling(20, min_periods=5).mean().fillna(
        pd.Series(vol_arr).expanding().mean()
    ).values.clip(min=1.0)
    vol_norm = vol_arr / vol_mean

    # Normalise each dimension by its rolling 60d std (make state dimensions comparable)
    state_raw = _build_state(log_ret, hl_norm, vol_norm)
    n = len(df)

    # 5-bar forward cumulative return (historical, ONLY used for past neighbours' outcomes)
    # fwd5[i] = sum of log_ret[i+1..i+5] = cumul return over next 5 bars
    # This is historical info for neighbours we look up in the past, not for current bar
    fwd5 = np.full(n, np.nan)
    for i in range(n - 5):
        fwd5[i] = log_ret[i+1: i+6].sum()

    fle5  = np.full(n, np.nan)
    fle10 = np.full(n, np.nan)
    nbr_var  = np.full(n, np.nan)
    nbr_dist = np.full(n, np.nan)
    density  = np.full(n, np.nan)
    knn_outcome = np.full(n, np.nan)

    # Precompute per-bar NaN mask for state rows and outcomes (reused across windows).
    state_row_nan = np.any(np.isnan(state_raw), axis=1)
    invalid_row   = state_row_nan | np.isnan(fwd5)

    for i in range(_HIST + 10, n):
        cur = state_raw[i]
        if np.any(np.isnan(cur)):
            continue

        # Historical window: bars [i-HIST, i-6] (exclude last 5 bars to have outcomes)
        lo_idx = i - _HIST
        hi_idx = i - 5   # need fwd5[j] to be historical = j+5 <= i-1 → j <= i-6
        if hi_idx - lo_idx < _K5 + 2:
            continue

        hist = state_raw[lo_idx: hi_idx]   # shape (M, 4)
        outcomes = fwd5[lo_idx: hi_idx]     # shape (M,)

        # Filter rows with any nan
        valid = ~invalid_row[lo_idx: hi_idx]
        hist     = hist[valid]
        outcomes = outcomes[valid]
        m = len(hist)
        if m < _K5 + 2:
            continue

        # Standardise state dimensions by std of historical window
        mean = hist.mean(axis=0)
        std = hist.std(axis=0).clip(min=1e-8)
        cur_norm  = (cur - mean) / std
        hist_norm = (hist - mean) / std

        # Euclidean distances from current state to all historical states
        diffs = hist_norm - cur_norm[None, :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))

        # Single ordering reused for k=5 / k=10 / knn.
        order = np.argsort(dists)
        idx5  = order[:_K5]
        d5  = dists[idx5]
        out5 = outcomes[idx5]

        # kNN distance-weighted mean outcome (= k-NN return predictor); loop-2 guard
        # was the weaker m >= _K5 + 2, already satisfied here.
        d5c = d5.clip(min=1e-8)
        weights = 1.0 / d5c
        weights /= weights.sum()
        knn_outcome[i] = float(np.dot(weights, out5))

        # The remaining (loop-1) features require the stronger guard m >= _K10 + 2.
        if m < _K10 + 2:
            continue

        idx10 = order[:_K10]
        d10 = dists[idx10]
        out10 = outcomes[idx10]

        # Current bar's "outcome" = fwd5[i] — but that's future!
        # Instead FLE = how much do NEIGHBOUR outcomes VARY per unit state distance?
        # FLE_approx = log(std(outcomes_k) / mean(dists_k))
        # This is the local stretching rate without needing the current bar's future.
        mean_d5 = d5.mean()
        if mean_d5 > 1e-10 and len(out5) >= 3:
            fle5[i]  = np.log(out5.std() / mean_d5 + 1e-14)

        mean_d10 = d10.mean()
        if mean_d10 > 1e-10 and len(out10) >= 5:
            fle10[i] = np.log(out10.std() / mean_d10 + 1e-14)

        nbr_var[i]  = out5.var()
        nbr_dist[i] = mean_d5

        # State density: count of neighbours within eps = median distance / 2
        eps = np.median(dists) * 0.5
        density[i] = float((dists <= eps).mean())

    df["tqs_fle_k5_252d"]   = fle5
    df["tqs_fle_k10_252d"]  = fle10
    df["tqs_nbr_outcome_var"]  = nbr_var
    df["tqs_nbr_dist_mean"]    = nbr_dist
    df["tqs_state_density_252d"] = density

    fle5_s = pd.Series(fle5, index=df.index)
    df["tqs_sensitivity_rank_60d"] = fle5_s.rolling(60, min_periods=15).rank(pct=True)
    df["tqs_knn_mean_outcome"]     = knn_outcome

    return df
