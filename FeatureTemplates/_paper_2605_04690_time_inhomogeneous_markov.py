"""
_paper_2605_04690_time_inhomogeneous_markov.py
===============================================
Per-ticker OHLCV proxy of the time-varying Markov dynamics described in:

  "Learning Time-Inhomogeneous Markov Dynamics in Financial Time Series"
  arXiv 2605.04690

The paper builds explicit time-varying transition matrices over discretized
return states, finding that:
  (a) High-volatility regimes HOMOGENIZE transition dynamics — operator
      row-entropy is negatively correlated with realized variance (r = -0.62).
  (b) Chapman-Kolmogorov (CK) consistency breaks down in specific windows,
      signalling first-order memory failure.

This block implements per-ticker causal proxies using only OHLCV data:
  1. Discretize daily log-returns into K=5 states using rolling quantile bins
     (all bins computed from past data only — no lookahead).
  2. Over a 60-day trailing window estimate the empirical K×K transition matrix.
  3. Emit: mean row entropy (vol-homogenisation signal), diagonal persistence
     mass, off-diagonal mobility, a matrix-change regime-shift score,
     a dominant-state drift signal, and a CK-consistency diagnostic.

NOTE: This is an UNPROVEN candidate block (leading underscore). The paper works
on aggregated cross-sectional matrices; this per-ticker translation is a proxy
only and has not yet been validated against the model's top-decile marginal IC.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "tim_markov",
    "description": (
        "Per-ticker OHLCV proxy of time-inhomogeneous Markov transition dynamics "
        "(arXiv 2605.04690): rolling 60-day transition matrices over 5 return states — "
        "emits row-entropy (vol-homogenisation signal), persistence, mobility, "
        "regime-shift score, dominant-state drift, and Chapman-Kolmogorov violation."
    ),
    "requires": ["Close"],
    "produces": [
        "tim_row_entropy",          # mean row entropy of K×K transition matrix (paper's key signal)
        "tim_persistence",          # diagonal mass — avg self-transition probability
        "tim_mobility",             # off-diagonal mass — 1 - persistence
        "tim_matrix_change",        # Frobenius norm ||P_t - P_{t-1}|| (regime shift)
        "tim_dominant_drift",       # expected next state – current state (signed drift)
        "tim_ck_violation",         # ||P^2 - P_2step|| Frobenius norm (CK diagnostic)
    ],
    "tags": ["market_regime", "experimental", "markov"],
    "version": "1.0",
    "author": "proxy of arXiv 2605.04690 (time-inhomogeneous Markov); per-ticker OHLCV only",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_K = 5           # number of return states
_WIN = 60        # rolling window length in trading days
_MIN_OBS = 30    # minimum transitions needed before emitting a value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_entropy(P: np.ndarray) -> float:
    """Mean Shannon entropy across rows of a K×K stochastic matrix P.

    Rows that are all-zero (no observations in that source state) are skipped.
    Entropy is in nats. Returns NaN if no valid row.
    """
    entropies = []
    for row in P:
        total = row.sum()
        if total <= 0:
            continue
        p = row / total          # normalise to probability
        # Only non-zero entries contribute
        mask = p > 0
        if mask.sum() == 0:
            continue
        entropies.append(-np.dot(p[mask], np.log(p[mask])))
    if not entropies:
        return np.nan
    return float(np.mean(entropies))


def _transition_matrix(states: np.ndarray, k: int) -> np.ndarray:
    """Build an unnormalised K×K count matrix from a sequence of integer states.

    States must be integers in [0, k-1]. Only consecutive valid pairs counted.
    """
    P = np.zeros((k, k), dtype=np.float64)
    for i in range(len(states) - 1):
        s, ns = states[i], states[i + 1]
        if 0 <= s < k and 0 <= ns < k:
            P[s, ns] += 1.0
    return P


def _stochastic(P: np.ndarray) -> np.ndarray:
    """Row-normalise P; rows with zero sum become uniform distributions."""
    row_sums = P.sum(axis=1, keepdims=True)
    # Replace zero-sum rows with 1 to avoid div-by-zero; result will be uniform
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return P / safe


def _frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.sum((A - B) ** 2)))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-inhomogeneous Markov features on a single ticker's OHLCV frame.

    Rolling window = 60 days, K = 5 return states (quintile bins from past data).
    NaN for the first ~(_WIN + K) rows while insufficient history accumulates.
    """
    n = len(df)

    # Outputs pre-filled with NaN
    tim_row_entropy   = np.full(n, np.nan)
    tim_persistence   = np.full(n, np.nan)
    tim_mobility      = np.full(n, np.nan)
    tim_matrix_change = np.full(n, np.nan)
    tim_dominant_drift = np.full(n, np.nan)
    tim_ck_violation  = np.full(n, np.nan)

    close = df["Close"].to_numpy(dtype=np.float64)

    # Log returns: r[i] = log(close[i] / close[i-1]), r[0] = NaN
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    # We need at least _WIN + 1 data points (for _WIN transitions)
    # plus a few more for the 2-step CK check.
    prev_P_stoch = None   # stochastic matrix from the previous window (for matrix_change)

    for t in range(_WIN, n):
        # Window covers indices [t - _WIN + 1 .. t] — all past + current row.
        # Returns in this window: indices [t - _WIN + 1 .. t]
        # We need the preceding return too for state labelling, so we look at
        # returns [t - _WIN + 1 .. t] = _WIN values.
        win_ret = log_ret[t - _WIN + 1: t + 1]   # shape (_WIN,)

        # Skip if too many NaNs
        valid_mask = np.isfinite(win_ret)
        if valid_mask.sum() < _MIN_OBS:
            prev_P_stoch = None
            continue

        # ---- 1. Discretise returns into K states using causal quantile bins ----
        # Use only the valid (non-NaN) returns within the window for quantile edges.
        valid_ret = win_ret[valid_mask]
        # K quintile edges from past window only (causal)
        quantile_edges = np.nanquantile(valid_ret, np.linspace(0, 1, _K + 1))
        # Clip to avoid boundary issues
        quantile_edges[0] -= 1e-12
        quantile_edges[-1] += 1e-12

        # Assign states (0 to K-1); NaN returns get state -1 (excluded)
        states = np.full(_WIN, -1, dtype=np.int32)
        for i, r in enumerate(win_ret):
            if np.isfinite(r):
                s = np.searchsorted(quantile_edges[1:], r, side="right")
                states[i] = min(s, _K - 1)

        # ---- 2. Build 1-step transition matrix over the window ----------------
        P = _transition_matrix(states, _K)
        total_transitions = P.sum()
        if total_transitions < _MIN_OBS:
            prev_P_stoch = None
            continue

        P_stoch = _stochastic(P)

        # ---- 3. Row entropy (paper's key signal) ------------------------------
        tim_row_entropy[t] = _row_entropy(P)

        # ---- 4. Persistence (diagonal mass) -----------------------------------
        diag_mass = np.diag(P_stoch).sum() / _K
        tim_persistence[t] = float(diag_mass)
        tim_mobility[t] = float(1.0 - diag_mass)

        # ---- 5. Matrix change vs previous window (regime shift) ---------------
        if prev_P_stoch is not None:
            tim_matrix_change[t] = _frobenius(P_stoch, prev_P_stoch)
        prev_P_stoch = P_stoch.copy()

        # ---- 6. Dominant-state drift ------------------------------------------
        # Current state = state at index t within the window = last element
        current_state = states[-1]
        if current_state >= 0:
            # Expected next state under P_stoch
            expected_next = float(np.dot(np.arange(_K), P_stoch[current_state]))
            tim_dominant_drift[t] = expected_next - float(current_state)

        # ---- 7. Chapman-Kolmogorov 1-step violation ---------------------------
        # CK: P^2 should equal the 2-step empirical matrix.
        # Build 2-step transition matrix from the same window.
        P2_emp = np.zeros((_K, _K), dtype=np.float64)
        for i in range(len(states) - 2):
            s, ns2 = states[i], states[i + 2]
            if 0 <= s < _K and 0 <= ns2 < _K:
                P2_emp[s, ns2] += 1.0

        P2_total = P2_emp.sum()
        if P2_total >= _MIN_OBS / 2:
            P2_stoch = _stochastic(P2_emp)
            P_sq = P_stoch @ P_stoch       # theoretical 2-step under Markov assumption
            tim_ck_violation[t] = _frobenius(P_sq, P2_stoch)

    # ---- Assemble output columns and concat to df ----------------------------
    idx = df.index
    new_cols = pd.DataFrame(
        {
            "tim_row_entropy":    tim_row_entropy,
            "tim_persistence":    tim_persistence,
            "tim_mobility":       tim_mobility,
            "tim_matrix_change":  tim_matrix_change,
            "tim_dominant_drift": tim_dominant_drift,
            "tim_ck_violation":   tim_ck_violation,
        },
        index=idx,
    )

    return pd.concat([df, new_cols], axis=1)
