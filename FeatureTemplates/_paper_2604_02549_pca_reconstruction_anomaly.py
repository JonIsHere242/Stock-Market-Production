"""
_paper_2604_02549_pca_reconstruction_anomaly.py
================================================
Causal PCA-based anomaly / stress-event features inspired by the PCA branch of:

    "Financial Anomaly Detection for the Canadian Market"
    arXiv 2604.02549

Per-ticker, OHLCV-only, strictly-causal proxy:
  - At each time t, a lag-embedding trajectory matrix is built from the PAST
    `window` rows (strictly <= t-1 for the PCA basis, current row for scoring).
  - PCA is computed from scratch via truncated SVD (numpy) on the past window.
  - The current observation is projected onto the top-k PC subspace and the
    reconstruction error (residual norm) is recorded.
  - Additional features: share of energy in minor components, Hotelling T² in
    PC space, and eigenvalue concentration (top-1 / sum).

NO LOOKAHEAD: the SVD basis is always fit on rows strictly before the scored
row.  The causality test (truncating history and re-running) must produce
bit-identical values for all rows that fall within both runs.

Window = 60 bars, embedding dimension (lags) = 5, top-k PCs = 2.
Warm-up = window + lag = 65 rows → first 65 rows are NaN.

Columns produced (prefix `pcr_`):
  pcr_recon_error      - L2 norm of the residual (raw obs – reconstructed)
  pcr_minor_energy     - fraction of total energy in the minor (k+1..) PCs
  pcr_hotelling_t2     - Hotelling T² = squared Mahalanobis distance in PC space
  pcr_top_eigenconc    - top eigenvalue / sum of all eigenvalues (concentration)
  pcr_recon_zscore     - rolling 30-bar z-score of pcr_recon_error (anomaly signal)
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "pca_reconstruction_anomaly",
    "description": (
        "Causal trailing-window PCA reconstruction-error anomaly detector: "
        "projects each OHLCV observation onto top PCs fit on strictly past data, "
        "emitting residual norm, minor-component energy share, Hotelling T², "
        "eigenvalue concentration, and a rolling z-score anomaly signal."
    ),
    "requires":    ["Open", "High", "Low", "Close", "Volume"],
    "produces":    [
        "pcr_recon_error",
        "pcr_minor_energy",
        "pcr_hotelling_t2",
        "pcr_top_eigenconc",
        "pcr_recon_zscore",
    ],
    "tags":        ["volatility", "market_regime", "experimental"],
    "version":     "1.0",
    "author":      "paper arXiv 2604.02549 — per-ticker causal PCA anomaly proxy",
}

# ---------------------------------------------------------------------------
# Constants (tune here; kept small for speed)
# ---------------------------------------------------------------------------
_WINDOW: int = 60   # trailing bars used to fit PCA basis (past only)
_LAG:    int = 5    # number of lagged returns in the trajectory / embedding vector
_TOP_K:  int = 2    # number of top PCs to keep for reconstruction
_ZSCORE_WIN: int = 30  # rolling window for the anomaly z-score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_features(prices: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D price array of length L into a 2-D feature matrix with shape
    (L - _LAG, _LAG + 4).

    Columns:
      [0 .._LAG-1] : log-return lags 1.._LAG (lag 1 = most recent)
      [_LAG]       : log(High/Low)  — intraday range proxy
      [_LAG+1]     : (Close-Open)/Open — body ratio
      [_LAG+2]     : log(Volume+1) change 1-bar
      [_LAG+3]     : (Close - Open_open) normalised by range (Heiken-Ashi proxy)

    All derived from the price/volume slice passed in; the caller is responsible
    for passing only past-window data.

    `prices` must be a structured array of shape (L, 5): Open, High, Low, Close, Volume.
    """
    n = len(prices)
    opens  = prices[:, 0].astype(np.float64)
    highs  = prices[:, 1].astype(np.float64)
    lows   = prices[:, 2].astype(np.float64)
    closes = prices[:, 3].astype(np.float64)
    vols   = prices[:, 4].astype(np.float64)

    # Log returns (length n-1)
    log_ret = np.log(closes[1:] / np.where(closes[:-1] > 0, closes[:-1], np.nan))

    # Intraday range log ratio (length n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl_ratio = np.log(np.where(lows > 0, highs / lows, np.nan))
        body     = np.where(opens > 0, (closes - opens) / opens, np.nan)
        dvol     = np.log(vols[1:] + 1.0) - np.log(vols[:-1] + 1.0)

    # Trajectory (lag) embedding — shape (n-1-_LAG, _LAG)
    # row i corresponds to bar index i+_LAG in log_ret
    valid_start = _LAG
    n_rows = len(log_ret) - _LAG
    if n_rows <= 0:
        return np.empty((0, _LAG + 4), dtype=np.float64)

    traj = np.empty((n_rows, _LAG), dtype=np.float64)
    for lag in range(_LAG):
        # lag=0 → most recent return (index _LAG-1 .. n-2), lag=1 → one bar older …
        traj[:, lag] = log_ret[valid_start - lag - 1: len(log_ret) - lag - 1]

    # Align auxiliary series to the same row indices
    # row i of traj corresponds to bar (i + _LAG) in the 0-based log_ret index,
    # which itself corresponds to bar (i + _LAG + 1) in the original prices array.
    base = _LAG + 1  # first original-prices index aligned with traj row 0
    hl_a  = hl_ratio[base: base + n_rows]
    body_a = body[base: base + n_rows]
    dvol_a = dvol[_LAG: _LAG + n_rows]

    # Pad if any series is shorter (edge safety)
    def _pad(a, target):
        if len(a) >= target:
            return a[:target]
        return np.concatenate([np.full(target - len(a), np.nan), a])

    hl_a   = _pad(hl_a,   n_rows)
    body_a = _pad(body_a, n_rows)
    dvol_a = _pad(dvol_a, n_rows)

    feats = np.column_stack([traj, hl_a, body_a, dvol_a,
                              np.zeros(n_rows)])  # placeholder 4th extra col
    # Replace the placeholder with a price-velocity proxy: mean of lags 0..1
    feats[:, -1] = (traj[:, 0] + traj[:, min(1, _LAG - 1)]) / 2.0
    return feats  # shape (n_rows, _LAG + 4)


def _fit_pca_basis(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA on a 2-D feature matrix X (n_samples × n_features).
    Returns (mean, Vk, eigenvalues_all) where:
      mean      : (n_features,) column mean
      Vk        : (n_features, k) top-k right singular vectors
      eigvals   : (min(n,p),) all squared singular values / (n-1) ≈ eigenvalues
    """
    n, p = X.shape
    if n < 2 or p < 1:
        return np.zeros(p), np.zeros((p, k)), np.zeros(k)

    mean = np.nanmean(X, axis=0)
    Xc = X - mean

    # Replace NaN with 0 before SVD (treat missing as zero-deviation)
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)

    # Thin SVD — economy mode; U: (n,r), S: (r,), Vt: (r,p), r=min(n,p)
    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return mean, np.zeros((p, k)), np.zeros(p)

    eigvals = S ** 2 / max(n - 1, 1)
    k_eff = min(k, Vt.shape[0], p)
    Vk = Vt[:k_eff].T  # shape (p, k_eff)
    if k_eff < k:
        # Pad with zeros so shape is always (p, k)
        Vk = np.concatenate([Vk, np.zeros((p, k - k_eff))], axis=1)

    return mean, Vk, eigvals


def _score_one(
    x: np.ndarray, mean: np.ndarray, Vk: np.ndarray, eigvals: np.ndarray, k: int
) -> Tuple[float, float, float, float]:
    """
    Score a single observation x against a fitted PCA basis.

    Returns:
      recon_error   : L2 norm of (x - x_reconstructed)
      minor_energy  : fraction of total ||x_c||² in the minor-component subspace
      hotelling_t2  : T² = sum((x projected onto PC_i)² / eigenvalue_i) for i<k
      top_eigenconc : eigenvalue[0] / sum(eigenvalues)  (concentration)
    """
    xc = x - mean
    xc = np.where(np.isfinite(xc), xc, 0.0)

    # Project onto top-k PCs
    scores = Vk.T @ xc  # (k,)

    # Reconstruction in original space
    x_recon = Vk @ scores  # (p,)
    residual = xc - x_recon

    total_energy = float(np.dot(xc, xc))
    recon_error  = float(np.sqrt(np.dot(residual, residual)))

    if total_energy > 0:
        minor_energy = float(np.dot(residual, residual)) / total_energy
    else:
        minor_energy = np.nan

    # Hotelling T² (sum of standardised squared scores)
    ev_k = eigvals[:k]
    safe_ev = np.where(ev_k > 0, ev_k, np.nan)
    hot = float(np.nansum(scores ** 2 / np.where(np.isfinite(safe_ev), safe_ev, np.nan)))
    if not np.isfinite(hot):
        hot = np.nan

    # Top eigenvalue concentration
    ev_sum = float(np.nansum(eigvals)) if len(eigvals) > 0 else 0.0
    if ev_sum > 0 and len(eigvals) > 0:
        top_eigenconc = float(eigvals[0]) / ev_sum
    else:
        top_eigenconc = np.nan

    return recon_error, minor_energy, hot, top_eigenconc


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-causal PCA reconstruction anomaly features.

    For each row t:
      1. Gather OHLCV rows [t - _WINDOW - _LAG, t-1] (past only).
      2. Build feature matrix via lag-embedding of log-returns + intraday stats.
      3. Fit PCA on that feature matrix (mean + SVD).
      4. Build the current observation from rows [t - _LAG, t].
      5. Score and record pcr_* metrics.

    Warm-up: first (_WINDOW + _LAG + 1) rows will be NaN.

    Performance: O(N * _WINDOW * p²) with p = _LAG+4 = 9.
    At N=700, _WINDOW=60, p=9 the inner SVD dominates; total ~5-15 ms.
    """
    n = len(df)
    k = _TOP_K

    # Extract raw arrays once (avoids repeated pandas overhead inside loop)
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=np.float64)

    # Output arrays, initialised to NaN
    out_recon   = np.full(n, np.nan, dtype=np.float64)
    out_minor   = np.full(n, np.nan, dtype=np.float64)
    out_hot     = np.full(n, np.nan, dtype=np.float64)
    out_topconc = np.full(n, np.nan, dtype=np.float64)

    # Number of feature columns produced by _build_features
    n_feat = _LAG + 4

    # Minimum history needed before we can score row t:
    #   _WINDOW bars for the PCA basis  +  _LAG bars to form the current vector
    #   The basis requires at least _LAG+2 rows to produce at least 1 feature row.
    min_basis_rows = _WINDOW   # we want _WINDOW feature-rows → need _WINDOW+_LAG+1 price rows
    warmup = _WINDOW + _LAG + 1  # first row we can score (0-based index)

    for t in range(warmup, n):
        # ---- 1. PCA basis: rows [t - _WINDOW - _LAG, t) (past only) --------
        basis_start = t - _WINDOW - _LAG
        if basis_start < 0:
            continue
        basis_prices = ohlcv[basis_start: t]   # shape (_WINDOW+_LAG, 5)

        feats = _build_features(basis_prices)   # shape ~ (_WINDOW, n_feat)
        if feats.shape[0] < k + 1:
            continue

        # Use the last _WINDOW rows (most recent past) for PCA fitting
        X_fit = feats[-_WINDOW:] if feats.shape[0] >= _WINDOW else feats

        mean_v, Vk, eigvals = _fit_pca_basis(X_fit, k)

        # ---- 2. Current observation: rows [t - _LAG - 1, t] ----------------
        # We need _LAG+2 rows (for _LAG log-returns + auxiliaries) ending at t.
        # Row t is the CURRENT bar (index t) — it has already occurred (close price
        # is known at end of day t); the PCA basis was fit on data UP TO t-1.
        cur_prices = ohlcv[t - _LAG - 1: t + 1]   # shape (_LAG+2, 5) or less
        cur_feats = _build_features(cur_prices)     # shape (1, n_feat) if enough data
        if cur_feats.shape[0] < 1:
            continue
        x_cur = cur_feats[-1]  # last row = current bar's feature vector

        if not np.all(np.isfinite(x_cur)):
            # Replace NaN with 0 for scoring consistency (match _fit_pca_basis)
            x_cur = np.where(np.isfinite(x_cur), x_cur, 0.0)

        # ---- 3. Score -------------------------------------------------------
        re, me, ht, tc = _score_one(x_cur, mean_v, Vk, eigvals, k)
        out_recon[t]   = re
        out_minor[t]   = me
        out_hot[t]     = ht
        out_topconc[t] = tc

    # ---- Build output series aligned to df.index ----------------------------
    idx = df.index
    s_recon   = pd.Series(out_recon,   index=idx, dtype=np.float64)
    s_minor   = pd.Series(out_minor,   index=idx, dtype=np.float64)
    s_hot     = pd.Series(out_hot,     index=idx, dtype=np.float64)
    s_topconc = pd.Series(out_topconc, index=idx, dtype=np.float64)

    # Rolling z-score of recon_error (anomaly signal) using PAST data only
    # shift(1) ensures we don't use the current bar's error in its own z-score
    _past_recon = s_recon.shift(1)
    _mean_re = _past_recon.rolling(_ZSCORE_WIN, min_periods=max(5, _ZSCORE_WIN // 2)).mean()
    _std_re  = _past_recon.rolling(_ZSCORE_WIN, min_periods=max(5, _ZSCORE_WIN // 2)).std()
    s_zscore = pd.Series(
        np.where(
            _std_re > 0,
            (s_recon - _mean_re) / _std_re,
            np.nan,
        ),
        index=idx,
        dtype=np.float64,
    )

    # Clamp extreme values to avoid inf propagation
    s_hot    = s_hot.clip(lower=0, upper=1e6)
    s_zscore = s_zscore.clip(lower=-20, upper=20)

    # ---- Assign new columns (never modify existing) -------------------------
    df["pcr_recon_error"]   = s_recon
    df["pcr_minor_energy"]  = s_minor
    df["pcr_hotelling_t2"]  = s_hot
    df["pcr_top_eigenconc"] = s_topconc
    df["pcr_recon_zscore"]  = s_zscore

    return df
