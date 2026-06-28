"""
Markov chain stationary distribution and spectral mixing features — per-ticker proxy
derived from:
  "A Markov Chain Model for the Analysis, Prediction and Comparison of Stock Exchange
  Markets" (hal:5036106).

The paper discretises daily price movements into 5 states and computes transition
probabilities and limiting/stationary distributions to characterise market stability
and volatility across three exchanges. Key insight: the SPECTRAL GAP (1 - λ₂) of the
transition matrix controls the mixing time — how quickly the chain forgets its initial
state and converges to equilibrium.

Per-ticker OHLCV proxy:
  - Discretise daily log-returns into K=5 states using rolling quantile bins (causal).
  - Estimate rolling empirical 5×5 transition matrix over a 60-day window.
  - Compute the stationary distribution π (dominant left eigenvector) and the
    spectral gap (1 - second-largest eigenvalue magnitude).
  - Spectral gap close to 1 = fast-mixing (mean-reverting, efficient) market.
    Spectral gap close to 0 = slow-mixing (trending, persistent) market.
  - Stationary entropy = H(π) measures how evenly the chain spends time across states.
  - Current-state stationary weight = π[current_state], a mean-reversion signal:
    high π[s] means this state is frequently visited → weaker continuation signal.

All rolling, causal, no lookahead. Does NOT duplicate the existing tim_markov block
(which computes row entropy, persistence/mobility, matrix change, dominant drift, and
CK violation — none of which are the spectral gap or stationary distribution).
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "markov_spectral_mixing",
    "description": (
        "Rolling Markov chain spectral gap and stationary distribution features: "
        "spectral gap (mixing speed = 1 - λ₂), stationary entropy, and stationary "
        "weight of current state. Proxy from hal:5036106 market stability analysis."
    ),
    "requires": ["Close"],
    "produces": [
        "mkv_spectral_gap_60",          # spectral gap 1 - |λ₂| over 60-day window
        "mkv_stationary_entropy_60",    # Shannon entropy of stationary dist π
        "mkv_current_state_weight_60",  # π[current_state] — mean-reversion signal
        "mkv_spectral_gap_30",          # same but 30-day window (shorter-term)
        "mkv_stationary_entropy_30",
    ],
    "tags": ["experimental", "markov", "spectral", "regime"],
    "version": "1.0",
    "author": "paper-mining slate 5",
}

_K = 5          # number of return states
_WIN_LONG = 60  # rolling window (paper uses 102 days; we use 60 for more responsiveness)
_WIN_SHORT = 30
_MIN_OBS = 20   # minimum transitions before emitting values


def _build_transition_matrix(states: np.ndarray, k: int) -> np.ndarray:
    """Build unnormalized K×K count matrix from integer state sequence (vectorized)."""
    P = np.zeros((k, k), dtype=np.float64)
    src = states[:-1]
    dst = states[1:]
    valid = (src >= 0) & (src < k) & (dst >= 0) & (dst < k)
    if valid.any():
        np.add.at(P, (src[valid], dst[valid]), 1.0)
    return P


def _row_normalize(P: np.ndarray) -> np.ndarray:
    """Row-normalize to get stochastic matrix; rows with zero sum become uniform."""
    row_sums = P.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return P / safe


_POWER_ITERS = 12        # iterations for power iteration (stationary dist only)
_QTILE_LINSPACE = np.linspace(0, 1, _K + 1)   # module-level constant


def _process_window(win_ret: np.ndarray, k: int, min_obs: int):
    """
    Build transition matrix from a return window and compute spectral features.

    Uses power iteration (O(k² × iters), negligible for k=5) instead of full
    eigen-decomposition to keep the per-row cost under control:
      - Stationary distribution π: run P repeatedly from uniform start until convergence.
      - Spectral gap proxy: measure mixing speed as ||π_t - π|| decay across
        iterations. We compute log||d_t|| / log||d_{t/2}|| ≈ -log(λ₂)·(t-t/2)/log(||d||).
        Equivalently, the ratio of norms at t//2 and t gives λ₂ ≈ norm_t/norm_half.

    Returns (spectral_gap, stationary_entropy, current_state_weight) or NaNs.
    """
    nan_result = (np.nan, np.nan, np.nan)

    valid_mask = np.isfinite(win_ret)
    if valid_mask.sum() < min_obs:
        return nan_result

    valid_ret = win_ret[valid_mask]
    quantile_edges = np.nanquantile(valid_ret, _QTILE_LINSPACE)
    quantile_edges[0] -= 1e-12
    quantile_edges[-1] += 1e-12

    # Vectorized state assignment
    finite_mask = np.isfinite(win_ret)
    states = np.full(len(win_ret), -1, dtype=np.int32)
    states[finite_mask] = np.clip(
        np.digitize(win_ret[finite_mask], quantile_edges[1:], right=False),
        0, k - 1
    )

    P = _build_transition_matrix(states, k)
    if P.sum() < min_obs:
        return nan_result

    P_stoch = _row_normalize(P)

    # ---- Power iteration for stationary distribution -------------------------
    # Start from uniform; iterate π <- π @ P
    pi = np.full(k, 1.0 / k, dtype=np.float64)
    half_iters = _POWER_ITERS // 2

    for i in range(_POWER_ITERS):
        pi_new = pi @ P_stoch
        pi_new /= pi_new.sum()
        if i == half_iters - 1:
            pi_half = pi_new.copy()
        pi = pi_new

    # ---- Spectral gap estimate from convergence rate -------------------------
    # After half_iters: deviation from stationary ~ λ₂^half_iters * initial_deviation
    # After full iters: deviation ~ λ₂^full_iters
    # Ratio of norms: norm_full / norm_half ≈ λ₂^half_iters
    d_half = np.abs(pi_half - pi)      # deviation at half-way point vs final
    d_full_vs_half = np.linalg.norm(pi - pi_half)
    norm_half = np.linalg.norm(d_half)

    if norm_half > 1e-12 and d_full_vs_half > 1e-12:
        lambda2_est = (d_full_vs_half / norm_half) ** (1.0 / half_iters)
        lambda2_est = min(max(lambda2_est, 0.0), 1.0)
        sg = float(1.0 - lambda2_est)
    else:
        # Converged quickly → large spectral gap
        sg = float(1.0 - 1e-6)

    # ---- Stationary entropy --------------------------------------------------
    mask_pos = pi > 0
    h_pi = float(-np.dot(pi[mask_pos], np.log(pi[mask_pos])))

    # ---- Current state weight -----------------------------------------------
    current_state = states[-1]
    if 0 <= current_state < k:
        pi_current = float(pi[current_state])
    else:
        pi_current = np.nan

    return sg, h_pi, pi_current


def _precompute_rolling_quantile_edges(log_ret: np.ndarray, win: int, k: int) -> np.ndarray:
    """
    Precompute rolling quantile edges for all windows using pandas rolling.quantile.
    Returns array of shape (n, k+1): row t contains quantile edges computed from
    log_ret[t-win+1:t+1]. Rows where insufficient data are NaN.
    Avoids calling np.nanquantile inside the per-row loop.
    """
    s = pd.Series(log_ret)
    q_levels = np.linspace(0, 1, k + 1)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0] for k=5
    edges = np.full((len(log_ret), k + 1), np.nan)
    for i, q in enumerate(q_levels):
        if q == 0.0:
            edges[:, i] = s.rolling(win, min_periods=_MIN_OBS).min().values
        elif q == 1.0:
            edges[:, i] = s.rolling(win, min_periods=_MIN_OBS).max().values
        else:
            edges[:, i] = s.rolling(win, min_periods=_MIN_OBS).quantile(q).values
    # Slightly expand edges to ensure all values fall within bins
    edges[:, 0] -= 1e-12
    edges[:, -1] += 1e-12
    return edges


def _spectral_gap_from_states(states: np.ndarray, k: int) -> float:
    """
    Estimate spectral gap from the state sequence using the fact that for an
    ergodic Markov chain, the second eigenvalue λ₂ ≈ lag-1 autocorrelation of
    a centered indicator function of the states. This avoids an expensive eigvals
    call and is O(n) vs O(k³).

    We compute the normalized lag-1 autocorrelation of the state sequence
    (treating states as integers), then spectral gap ≈ 1 - |autocorr|.
    Large spectral gap = low autocorrelation = fast mixing.
    """
    valid = states[states >= 0].astype(np.float64)
    if len(valid) < 4:
        return np.nan
    # Use pairs (s_t, s_{t+1}) from original sequence (preserving order)
    src = states[:-1]
    dst = states[1:]
    valid_pairs = (src >= 0) & (dst >= 0)
    if valid_pairs.sum() < 4:
        return np.nan
    s0 = src[valid_pairs].astype(np.float64)
    s1 = dst[valid_pairs].astype(np.float64)
    s0_mean = s0.mean()
    s1_mean = s1.mean()
    s0_std = s0.std()
    s1_std = s1.std()
    if s0_std < 1e-12 or s1_std < 1e-12:
        return np.nan  # all same state — fully persistent, spectral gap ≈ 0
    autocorr = np.dot(s0 - s0_mean, s1 - s1_mean) / (len(s0) * s0_std * s1_std)
    return float(1.0 - abs(autocorr))


def _process_window_fast(win_ret: np.ndarray, edge: np.ndarray, k: int, min_obs: int):
    """
    Same as _process_window but takes pre-computed quantile edge vector (length k+1).
    Skips the np.nanquantile call since edges are already computed.
    Uses:
      - State-sequence lag-1 autocorrelation as spectral gap proxy (O(n), avoids eigvals)
      - Power iteration (12 steps) for stationary distribution
    """
    nan_result = (np.nan, np.nan, np.nan)

    if np.any(np.isnan(edge)):
        return nan_result

    valid_mask = np.isfinite(win_ret)
    if valid_mask.sum() < min_obs:
        return nan_result

    # Vectorized state assignment using pre-computed edges
    finite_mask = np.isfinite(win_ret)
    states = np.full(len(win_ret), -1, dtype=np.int32)
    states[finite_mask] = np.clip(
        np.digitize(win_ret[finite_mask], edge[1:], right=False),
        0, k - 1
    )

    # Spectral gap from state-sequence autocorrelation (fast, avoids eigvals)
    sg = _spectral_gap_from_states(states, k)

    P = _build_transition_matrix(states, k)
    if P.sum() < min_obs:
        return sg, np.nan, np.nan

    P_stoch = _row_normalize(P)

    # Power iteration for stationary distribution
    pi = np.full(k, 1.0 / k, dtype=np.float64)
    for _ in range(_POWER_ITERS):
        pi_new = pi @ P_stoch
        s_pi = pi_new.sum()
        if s_pi > 1e-12:
            pi_new /= s_pi
        pi = pi_new

    # Shannon entropy
    mask_pos = pi > 0
    h_pi = float(-np.dot(pi[mask_pos], np.log(pi[mask_pos])))

    # Current state weight
    current_state = states[-1]
    pi_current = float(pi[current_state]) if 0 <= current_state < k else np.nan

    return sg, h_pi, pi_current


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Markov spectral mixing features for a single ticker.
    Rolling windows of 30 and 60 days.
    Precomputes rolling quantile edges vectorized to avoid per-row nanquantile calls.
    """
    n = len(df)

    mkv_sg60  = np.full(n, np.nan)
    mkv_he60  = np.full(n, np.nan)
    mkv_csw60 = np.full(n, np.nan)
    mkv_sg30  = np.full(n, np.nan)
    mkv_he30  = np.full(n, np.nan)

    close = df["Close"].to_numpy(dtype=np.float64)

    # Log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # Precompute rolling quantile edges for both windows
    edges30 = _precompute_rolling_quantile_edges(log_ret, _WIN_SHORT, _K)
    edges60 = _precompute_rolling_quantile_edges(log_ret, _WIN_LONG, _K)

    for t in range(_WIN_SHORT, n):
        # 30-day window
        win30 = log_ret[t - _WIN_SHORT + 1: t + 1]
        sg30, he30, _ = _process_window_fast(win30, edges30[t], _K, _MIN_OBS)
        mkv_sg30[t] = sg30
        mkv_he30[t] = he30

        # 60-day window (only when enough history)
        if t >= _WIN_LONG:
            win60 = log_ret[t - _WIN_LONG + 1: t + 1]
            sg60, he60, csw60 = _process_window_fast(win60, edges60[t], _K, _MIN_OBS)
            mkv_sg60[t] = sg60
            mkv_he60[t] = he60
            mkv_csw60[t] = csw60

    idx = df.index
    new_cols = pd.DataFrame(
        {
            "mkv_spectral_gap_60":         mkv_sg60,
            "mkv_stationary_entropy_60":   mkv_he60,
            "mkv_current_state_weight_60": mkv_csw60,
            "mkv_spectral_gap_30":         mkv_sg30,
            "mkv_stationary_entropy_30":   mkv_he30,
        },
        index=idx,
    )
    return pd.concat([df, new_cols], axis=1)
