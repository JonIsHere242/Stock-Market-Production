"""
Geometric Spectral Observables — per-ticker proxy derived from:
  "Geometric Observables for Financial Regime Detection" (arXiv 2605.17117).

The paper extracts four observables from a LEARNED SPECTRAL EMBEDDING of
equity-index returns:
  1. Berry Phase Rate      — curvature / rotation speed of the leading eigenvector
  2. Spectral Entropy      — Shannon entropy of the eigenvalue spectrum (how spread out)
  3. Reduced State Purity  — concentration of the spectral decomposition (inverse entropy)
  4. Hamiltonian Sensitivity — eigenvalue sensitivity to small perturbations

The paper's method is cross-sectional (index-level embedding across many stocks).
This block is a FAITHFUL PER-TICKER PROXY: we construct a rolling multivariate
feature matrix from per-ticker OHLCV primitives (daily return, high-low range,
body fraction, log-volume), compute its rolling empirical covariance, then extract
the eigenspectrum-based observables from that covariance.

All four observables are computed from the eigenvalues of the rolling d x d
covariance matrix (d=4 features), giving a compact but geometrically faithful
approximation of the paper's spectral regime signals.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "paper_2605_17117_geometric_spectral_observables",
    "description": (
        "Per-ticker proxy for the geometric spectral regime-detection observables "
        "(Berry Phase Rate, Spectral Entropy, State Purity, Hamiltonian Sensitivity) "
        "from arXiv 2605.17117; paper is cross-sectional index-level — this uses "
        "rolling eigendecomposition of a per-ticker OHLCV covariance matrix."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "geo_spectral_entropy_20d",
        "geo_spectral_entropy_60d",
        "geo_state_purity_20d",
        "geo_state_purity_60d",
        "geo_berry_phase_rate_20d",
        "geo_berry_phase_rate_60d",
        "geo_hamiltonian_sensitivity_20d",
        "geo_hamiltonian_sensitivity_60d",
    ],
    "tags":        ["market_regime", "spectral", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "paper:2605.17117 — per-ticker OHLCV proxy; paper requires cross-sectional index embedding",
}

# Number of primitive OHLCV features used to form the covariance matrix
_N_FEATURES = 4


def _build_primitives(df: pd.DataFrame) -> np.ndarray:
    """
    Build a (n, 4) matrix of OHLCV primitives:
      [0] log daily return
      [1] normalized high-low range (range / close)
      [2] candle body fraction (|close - open| / (high - low + eps))
      [3] log-volume (normalized by rolling mean to make stationary-ish)
    """
    eps = 1e-10
    close = df["Close"].values.astype(np.float64)
    open_ = df["Open"].values.astype(np.float64)
    high  = df["High"].values.astype(np.float64)
    low   = df["Low"].values.astype(np.float64)
    vol   = df["Volume"].values.astype(np.float64)

    log_ret    = np.log(close[1:] / (close[:-1] + eps))
    hl_range   = (high[1:] - low[1:]) / (close[1:] + eps)
    body_frac  = np.abs(close[1:] - open_[1:]) / (high[1:] - low[1:] + eps)
    log_vol    = np.log(vol[1:] + 1.0)

    # Stack into (n-1, 4); pad first row with NaN to align with df
    n = len(df)
    mat = np.full((n, _N_FEATURES), np.nan)
    mat[1:, 0] = log_ret
    mat[1:, 1] = hl_range
    mat[1:, 2] = body_frac
    mat[1:, 3] = log_vol

    return mat


def _spectral_entropy(eigvals: np.ndarray) -> float:
    """Shannon entropy of normalized eigenvalue spectrum (all eigenvalues >= 0)."""
    ev = eigvals.clip(min=0.0)
    total = ev.sum()
    if total < 1e-14:
        return np.nan
    p = ev / total
    # Clip to avoid log(0)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def _state_purity(eigvals: np.ndarray) -> float:
    """Purity = sum(p_i^2); 1/d means uniform (low purity), 1.0 means one dominant mode."""
    ev = eigvals.clip(min=0.0)
    total = ev.sum()
    if total < 1e-14:
        return np.nan
    p = ev / total
    return float(np.sum(p ** 2))


def _hamiltonian_sensitivity(eigvals: np.ndarray) -> float:
    """
    Condition-number proxy for Hamiltonian sensitivity:
    ratio of max to min non-zero eigenvalue.
    High value = ill-conditioned = sensitive to perturbations.
    """
    ev = eigvals.clip(min=0.0)
    ev_pos = ev[ev > 1e-12]
    if len(ev_pos) < 2:
        return np.nan
    return float(ev_pos.max() / ev_pos.min())


def _rolling_spectral_obs(mat: np.ndarray, window: int, min_obs: int):
    """
    Rolling spectral observables computed from the (n, d) primitive matrix.

    For each row i, we take the window mat[max(0,i-window+1):i+1], drop NaN rows,
    and if enough rows remain, compute the empirical covariance, then eigenvalues.

    Returns four arrays of length n: entropy, purity, berry_rate, sensitivity.
    """
    n, d = mat.shape
    entropy_arr     = np.full(n, np.nan)
    purity_arr      = np.full(n, np.nan)
    berry_arr       = np.full(n, np.nan)
    sensitivity_arr = np.full(n, np.nan)

    # Cache last eigenvector for Berry Phase Rate (angle between successive leading evecs)
    prev_eigvec = None
    prev_i      = -1

    for i in range(min_obs - 1, n):
        start = max(0, i - window + 1)
        block = mat[start: i + 1, :]
        # Drop rows with any NaN
        valid = block[~np.isnan(block).any(axis=1)]
        if valid.shape[0] < min_obs:
            prev_eigvec = None
            continue

        # Centre (subtract mean) — expanding within window
        centred = valid - valid.mean(axis=0)
        # Empirical covariance
        cov = (centred.T @ centred) / (valid.shape[0] - 1)

        # Eigendecomposition (symmetric matrix -> use eigh for stability + ascending order)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            prev_eigvec = None
            continue

        # eigh returns ascending eigenvalues; leading = last
        entropy_arr[i]     = _spectral_entropy(eigvals)
        purity_arr[i]      = _state_purity(eigvals)
        sensitivity_arr[i] = _hamiltonian_sensitivity(eigvals)

        # Berry Phase Rate: angle between current and previous leading eigenvector
        # (proxy for "curvature" / rotation rate of the dominant spectral mode)
        leading_evec = eigvecs[:, -1]  # leading eigenvector (largest eigenvalue)
        if prev_eigvec is not None and (i - prev_i) == 1:
            # Cosine similarity; clamp to [-1, 1] for acos safety
            cos_sim = float(np.clip(np.dot(leading_evec, prev_eigvec), -1.0, 1.0))
            berry_arr[i] = np.arccos(abs(cos_sim))  # absolute angle in [0, pi/2]
        prev_eigvec = leading_evec.copy()
        prev_i = i

    return entropy_arr, purity_arr, berry_arr, sensitivity_arr


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling geometric spectral observables from per-ticker OHLCV.

    For each window (20d, 60d):
      - geo_spectral_entropy_{w}d : entropy of rolling eigenspectrum (high = dispersed/uncertain)
      - geo_state_purity_{w}d     : purity (sum of squared eigenvalue fractions; high = dominant mode)
      - geo_berry_phase_rate_{w}d : angle between consecutive leading eigenvectors (rotation speed)
      - geo_hamiltonian_sensitivity_{w}d : eigenvalue condition ratio (sensitivity to perturbations)
    """
    mat = _build_primitives(df)

    for window, min_obs in [(20, 8), (60, 20)]:
        ent, pur, berry, sens = _rolling_spectral_obs(mat, window, min_obs)

        w = f"{window}d"
        df[f"geo_spectral_entropy_{w}"]       = ent
        df[f"geo_state_purity_{w}"]           = pur
        df[f"geo_berry_phase_rate_{w}"]       = berry
        df[f"geo_hamiltonian_sensitivity_{w}"] = sens

    return df
