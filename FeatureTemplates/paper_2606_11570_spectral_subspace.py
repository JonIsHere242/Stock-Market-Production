"""
Spectral Subspace Projection Features.

PAPER: "Enhancing Spectral Embedding through Robust and Flexible Knowledge Transfer
        in Electronic Health Records" (arxiv 2606.11570).

The paper is about EHR representation learning — no direct OHLCV method.
SUBSTITUTE (same theme — spectral embedding / subspace decomposition):

The paper's core COMPUTABLE method is a two-step spectral procedure:
  Step 1: Identify and remove irrelevant eigenvectors from a covariance matrix.
  Step 2: Project data separately onto "shared" (low-rank) vs "heterogeneous"
          (residual) subspace components.

OHLCV adaptation:
  We build a rolling multivariate matrix from [ret, hl_range, body_frac, vol_ret]
  compute a rolling eigendecomposition of its correlation matrix, and extract:
    - Projection onto the dominant (shared) eigenvector — "universal" regime.
    - Residual idiosyncratic component (unexplained by leading eigenvectors).
    - Eigenvalue gap: separation between 1st and 2nd eigenvalue.
    - Eigenvector stability: how much the leading eigenvector rotates bar-to-bar.
    - Signed projection: direction * magnitude of leading component.

Key improvement over naive PCA-momentum: we look at the GEOMETRY of the
covariance structure, not just the first PC score.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11570_spectral_subspace",
    "description": (
        "Rolling spectral subspace decomposition of multivariate OHLCV primitives: "
        "dominant-eigenvector projection, idiosyncratic residual, eigenvalue gap, "
        "eigenvector stability. Inspired by spectral embedding with knowledge "
        "transfer (arxiv 2606.11570)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ssp_shared_proj",        # signed projection onto leading eigenvector
        "ssp_idio_norm",          # norm of idiosyncratic residual
        "ssp_eigval_gap",         # lambda_1 - lambda_2
        "ssp_eigvec_stability",   # signed dot product with prev eigvec (rotation signal)
        "ssp_shared_frac",        # fraction of variance in top-2 eigenvectors
        "ssp_proj_zscore",        # z-score of shared_proj vs 63d rolling history
        "ssp_eigval_ratio",       # lambda_1 / trace (dominance ratio)
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11570",
}

_WINDOW = 40   # rolling window for covariance estimation


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    eps = 1e-12

    # Build signal channels
    ret = np.full(n, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    hl = np.where(close > 0, (high - low) / close, np.nan)
    body = np.where((high - low) > eps, (close - open_) / (high - low), np.nan)

    log_vol = np.log(np.where(volume > 0, volume, np.nan))
    vol_ret = np.full(n, np.nan)
    vol_ret[1:] = np.diff(log_vol)

    # Upper shadow fraction
    upper = np.where((high - low) > eps,
                     (high - np.maximum(close, open_)) / (high - low), np.nan)

    # Stack into signal matrix: (n, 5) = [ret, hl, body, vol_ret, upper]
    X = np.column_stack([ret, hl, body, vol_ret, upper])
    k = X.shape[1]

    shared_proj = np.full(n, np.nan)
    idio_norm = np.full(n, np.nan)
    eigval_gap = np.full(n, np.nan)
    eigvec_stab = np.full(n, np.nan)
    shared_frac = np.full(n, np.nan)
    ret_loading = np.full(n, np.nan)
    eigval_ratio = np.full(n, np.nan)

    prev_v1 = None

    for i in range(_WINDOW - 1, n):
        seg = X[i - _WINDOW + 1: i + 1]  # (_WINDOW, k)
        valid = np.all(np.isfinite(seg), axis=1)
        if valid.sum() < _WINDOW // 2:
            prev_v1 = None
            continue

        seg_v = seg[valid]
        # Standardize each column (z-score)
        mu = seg_v.mean(axis=0)
        std = seg_v.std(axis=0)
        std = np.where(std > eps, std, 1.0)
        seg_std = (seg_v - mu) / std

        # Correlation matrix (== cov of standardized)
        cor = seg_std.T @ seg_std / (len(seg_v) - 1)

        try:
            eigvals, eigvecs = np.linalg.eigh(cor)
        except np.linalg.LinAlgError:
            prev_v1 = None
            continue

        # eigh returns ascending; flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        v1 = eigvecs[:, 0]

        # Standardize current bar
        x_cur = X[i]
        if not np.all(np.isfinite(x_cur)):
            prev_v1 = None
            continue
        x_std = (x_cur - mu) / std

        # Signed projection onto leading eigenvector
        proj_coef = np.dot(x_std, v1)
        shared_proj[i] = proj_coef

        # Idiosyncratic: norm of residual after removing top-2 subspace
        if k >= 2:
            v2 = eigvecs[:, 1]
            proj2 = np.dot(x_std, v2)
            residual = x_std - proj_coef * v1 - proj2 * v2
        else:
            residual = x_std - proj_coef * v1
        idio_norm[i] = np.linalg.norm(residual)

        # Eigenvalue gap
        if len(eigvals) >= 2:
            eigval_gap[i] = eigvals[0] - eigvals[1]

        # Eigenvector stability: dot product with prev (signed, not abs, for direction)
        if prev_v1 is not None:
            eigvec_stab[i] = np.dot(v1, prev_v1)  # signed: rotation = negative
        prev_v1 = v1.copy()

        # Shared fraction: top-2 eigenvalues / trace
        total_ev = eigvals.sum()
        if total_ev > eps:
            shared_frac[i] = eigvals[:2].sum() / total_ev
            eigval_ratio[i] = eigvals[0] / total_ev

    df["ssp_shared_proj"] = shared_proj
    df["ssp_idio_norm"] = idio_norm
    df["ssp_eigval_gap"] = eigval_gap
    df["ssp_eigvec_stability"] = eigvec_stab
    df["ssp_shared_frac"] = shared_frac
    df["ssp_eigval_ratio"] = eigval_ratio

    # Derived: z-score of shared projection over rolling 63-bar history
    sp_s = pd.Series(shared_proj)
    sp_mu = sp_s.rolling(63, min_periods=20).mean()
    sp_std = sp_s.rolling(63, min_periods=20).std()
    df["ssp_proj_zscore"] = ((sp_s - sp_mu) / (sp_std + 1e-12)).values

    return df
