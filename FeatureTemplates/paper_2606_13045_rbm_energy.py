"""
Restricted Boltzmann Machine Energy Landscape Features.

PAPER: "A solvable model for unsupervised federated learning" (arxiv 2606.13045).
       Theoretical physics/ML paper — federated learning maps to a Restricted
       Boltzmann Machine (RBM) with a structured hidden layer. No direct OHLCV
       feature extractable from the paper's method.

SUBSTITUTE (same theme — energy landscape / Boltzmann-inspired latent states):

An RBM's FREE ENERGY for a visible state v is:
    F(v) = -b^T v - sum_j log(1 + exp(c_j + W_j^T v))

Low energy = familiar/high-probability pattern.
High energy = anomalous/low-probability pattern.

We fit a lightweight RBM on rolling windows of OHLCV primitives and extract:
  1. Free energy of current bar (anomaly = high energy).
  2. Z-score of free energy vs rolling history.
  3. Mean hidden unit activation (latent state intensity).
  4. Hidden unit entropy (latent state uncertainty).
  5. Visible reconstruction error.
  6. Pattern novelty: log-ratio vs median energy.

SPEED: RBM is refitted every 21 bars; only 5 CD-1 steps; hidden dim=6.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13045_rbm_energy",
    "description": (
        "RBM-inspired rolling energy landscape: free energy anomaly score, "
        "hidden-unit activation statistics, reconstruction error, and pattern "
        "novelty. Inspired by federated learning mapped to RBM dynamics "
        "(arxiv 2606.13045)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "rbm_free_energy",
        "rbm_free_energy_zscore",
        "rbm_hidden_activation_mean",
        "rbm_hidden_entropy",
        "rbm_pattern_novelty",
        "rbm_reconstruction_error",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13045",
}

_N_HIDDEN = 6
_WIN = 50
_CD_STEPS = 5
_REFIT_EVERY = 21


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                    np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))))


def _build_ohlcv_signal(open_, high, low, close, volume, idx):
    """Build 6-dimensional OHLCV feature vector for a single bar."""
    eps = 1e-10
    c, o, h, l, v = close[idx], open_[idx], high[idx], low[idx], volume[idx]
    c_prev = close[idx - 1] if idx > 0 else c

    ret = np.log(c / c_prev) if c_prev > 0 and c > 0 else np.nan
    hl = (h - l) / c if c > 0 else np.nan
    body = (c - o) / (h - l + eps) if (h - l) > eps else 0.0
    # Upper/lower shadow
    upper = (h - max(c, o)) / (h - l + eps) if (h - l) > eps else 0.0
    lower = (min(c, o) - l) / (h - l + eps) if (h - l) > eps else 0.0
    # Volume relative to prev
    v_prev = volume[idx - 1] if idx > 0 else v
    vol_ret = np.log(v / v_prev) if v > 0 and v_prev > 0 else np.nan

    return np.array([ret, hl, body, upper, lower, vol_ret])


def _build_signal_matrix(open_, high, low, close, volume):
    """Vectorized equivalent of _build_ohlcv_signal for every bar -> (n, 6).

    Row 0 mirrors the idx==0 branch (c_prev=c, v_prev=v). Reproduces the exact
    scalar branch logic of _build_ohlcv_signal so downstream values are identical.
    """
    eps = 1e-10
    n = len(close)
    c, o, h, l, v = close, open_, high, low, volume

    c_prev = np.empty(n, dtype=float)
    c_prev[0] = c[0]
    c_prev[1:] = c[:-1]
    v_prev = np.empty(n, dtype=float)
    v_prev[0] = v[0]
    v_prev[1:] = v[:-1]

    hl_diff = h - l
    denom = hl_diff + eps
    valid_hl = hl_diff > eps

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where((c_prev > 0) & (c > 0), np.log(c / c_prev), np.nan)
        hl = np.where(c > 0, hl_diff / c, np.nan)
        body = np.where(valid_hl, (c - o) / denom, 0.0)
        upper = np.where(valid_hl, (h - np.maximum(c, o)) / denom, 0.0)
        lower = np.where(valid_hl, (np.minimum(c, o) - l) / denom, 0.0)
        vol_ret = np.where((v > 0) & (v_prev > 0), np.log(v / v_prev), np.nan)

    return np.column_stack([ret, hl, body, upper, lower, vol_ret])


def _build_window_matrix_from(S, end_idx, win):
    """Build (win, 6) normalized visible matrix from precomputed signal matrix S
    over bars [end_idx-win, end_idx)."""
    start = end_idx - win
    V_raw = S[start:end_idx].copy()  # (win, 6)
    # Rows with global index <= 0 are forced to NaN (mirrors original j<=0 branch)
    if start <= 0:
        V_raw[: 1 - start] = np.nan

    # Filter valid rows
    valid = np.all(np.isfinite(V_raw), axis=1)
    if valid.sum() < win // 2:
        return None, None, None

    V_v = V_raw[valid]
    mu = V_v.mean(axis=0)
    std = V_v.std(axis=0)
    std = np.where(std > 1e-10, std, 1.0)

    # Normalize full window
    V_norm = (V_raw - mu) / std
    V_norm = np.clip(V_norm, -4, 4)
    # Map to [0,1] for soft RBM
    V_01 = _sigmoid(V_norm)

    # Return only valid rows for training
    V_train = V_01[valid]
    return V_train, mu, std


def _fit_rbm(V: np.ndarray, n_hidden: int, n_steps: int) -> tuple:
    """
    Fast CD-k RBM fit. V shape (m, n_vis). Returns W, b, c.
    Uses fixed seed (42) for determinism.
    """
    rng = np.random.default_rng(42)
    n_vis = V.shape[1]
    W = rng.normal(0, 0.01, (n_vis, n_hidden)).astype(np.float32)
    b = np.zeros(n_vis, dtype=np.float32)
    c = np.zeros(n_hidden, dtype=np.float32)
    V = V.astype(np.float32)

    lr = 0.05
    for _ in range(n_steps):
        h_pos = _sigmoid(V @ W + c)                        # (m, n_h)
        h_samp = (rng.random(h_pos.shape) < h_pos).astype(np.float32)
        v_neg = _sigmoid(h_samp @ W.T + b)                 # (m, n_v)
        h_neg = _sigmoid(v_neg @ W + c)
        dW = (V.T @ h_pos - v_neg.T @ h_neg) / len(V)
        db = (V - v_neg).mean(axis=0)
        dc = (h_pos - h_neg).mean(axis=0)
        W += lr * dW
        b += lr * db
        c += lr * dc

    return W, b, c


def _free_energy(v: np.ndarray, W, b, c) -> float:
    """F(v) = -b^T v - sum_j softplus(c_j + W_j^T v)"""
    h_in = c + v @ W  # (n_h,)
    sp = np.where(h_in >= 0,
                  h_in + np.log1p(np.exp(-np.clip(h_in, 0, 50))),
                  np.log1p(np.exp(np.clip(h_in, -50, 0))))
    return float(-np.dot(b, v) - sp.sum())


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # Precompute the OHLCV signal vector for every bar once (vectorized).
    S = _build_signal_matrix(open_, high, low, close, volume)

    fe_arr = np.full(n, np.nan)
    fe_z = np.full(n, np.nan)
    h_act = np.full(n, np.nan)
    h_ent = np.full(n, np.nan)
    novelty = np.full(n, np.nan)
    recon_err = np.full(n, np.nan)

    W_cache = b_cache = c_cache = None
    mu_cache = std_cache = None
    last_fit = -999
    fe_history = []  # rolling list of free energies

    for i in range(_WIN + 2, n):
        # Refit every _REFIT_EVERY bars
        if i - last_fit >= _REFIT_EVERY or W_cache is None:
            result = _build_window_matrix_from(S, i, _WIN)
            if result[0] is not None:
                V_train, mu_cache, std_cache = result
                try:
                    W_cache, b_cache, c_cache = _fit_rbm(V_train, _N_HIDDEN, _CD_STEPS)
                    last_fit = i
                except Exception:
                    pass

        if W_cache is None or mu_cache is None:
            fe_history.append(np.nan)
            continue

        # Build current bar's visible vector
        v_raw = S[i]
        if not np.all(np.isfinite(v_raw)):
            fe_history.append(np.nan)
            continue

        # Normalize with window stats and apply sigmoid
        v_norm = np.clip((v_raw - mu_cache) / (std_cache + 1e-10), -4, 4)
        v_01 = _sigmoid(v_norm)

        # Free energy
        try:
            f_e = _free_energy(v_01, W_cache, b_cache, c_cache)
        except Exception:
            fe_history.append(np.nan)
            continue

        fe_arr[i] = f_e
        fe_history.append(f_e)

        # Hidden activations
        h_prob = _sigmoid(c_cache + v_01 @ W_cache)
        h_act[i] = float(h_prob.mean())
        # Entropy of binary hidden units
        h_clip = np.clip(h_prob, 1e-8, 1.0 - 1e-8)
        h_ent[i] = float(-(h_clip * np.log(h_clip) + (1 - h_clip) * np.log(1 - h_clip)).mean())

        # Reconstruction error
        v_recon = _sigmoid(h_prob @ W_cache.T + b_cache)
        recon_err[i] = float(np.mean((v_01 - v_recon) ** 2))

        # Z-score and novelty from rolling history (last 63 non-nan)
        _hist_slice = np.asarray(fe_history[-63:], dtype=float)
        hist_arr = _hist_slice[np.isfinite(_hist_slice)]
        if len(hist_arr) >= 10:
            mu_fe = float(hist_arr.mean())
            std_fe = float(hist_arr.std())
            fe_z[i] = (f_e - mu_fe) / (std_fe + 1e-12)
            med_fe = float(np.median(hist_arr))
            # novelty: normalized deviation from median
            novelty[i] = abs(f_e - med_fe) / (abs(med_fe) + 1e-8)

    df["rbm_free_energy"] = fe_arr
    df["rbm_free_energy_zscore"] = fe_z
    df["rbm_hidden_activation_mean"] = h_act
    df["rbm_hidden_entropy"] = h_ent
    df["rbm_pattern_novelty"] = novelty
    df["rbm_reconstruction_error"] = recon_err

    return df
