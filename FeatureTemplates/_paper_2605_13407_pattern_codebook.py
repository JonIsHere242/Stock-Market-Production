"""
_paper_2605_13407_pattern_codebook.py  --  Pattern-codebook features inspired by
arxiv 2605.13407 "Vector-Quantized Discrete Latent Factors Meet Financial Priors:
Dynamic Cross-Sectional Stock Ranking" (PRISM-VQ).

PRISM-VQ uses vector quantisation (VQ) as an information bottleneck: learned
discrete codes suppress noise while preserving robust market structure. This block
implements a DETERMINISTIC (no-learning) per-ticker proxy:

  - Slide a 10-day window over daily log-returns.
  - Z-normalise the window (subtract mean, divide by std) to obtain a shape vector.
  - Compare that shape vector against a HARD-CODED codebook of 5 archetype shapes
    via Pearson correlation (= cosine similarity of z-vectors, since both are already
    zero-meaned after z-normalisation of the window; the archetype is also manually
    z-normalised so both sides are unit-norm).
  - Emit the best-match correlation (quantisation quality), the winning archetype
    index, the quantisation error (1 - best_match_corr), and three named
    archetype correlations (uptrend, V-bottom, mean-reversion oscillation).

All arithmetic uses trailing windows only (no look-ahead). Zero-variance windows
produce NaN. The codebook is a compile-time constant — no fitting, no randomness.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CODEBOOK  — 5 archetype shapes, each of length 10.
# Each vector is hand-crafted to represent a canonical price-shape pattern and
# is pre-normalised (zero-mean, unit-norm) so correlation == cosine similarity.
# ---------------------------------------------------------------------------

def _make_codebook() -> np.ndarray:
    """Build and normalise the 5-archetype codebook. Returns shape (5, 10)."""
    raw = np.array([
        # 0: steady uptrend ramp  (monotone increasing)
        np.arange(1, 11, dtype=float),

        # 1: steady downtrend ramp  (monotone decreasing)
        np.arange(10, 0, -1, dtype=float),

        # 2: V-bottom  (falls first half, rises second half)
        np.array([-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=float),

        # 3: inverted-V  (rises first half, falls second half)
        np.array([4.5, 3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5, -4.5], dtype=float),

        # 4: mean-reverting oscillation  (alternating ±, decaying amplitude)
        np.array([3.0, -2.5, 2.0, -1.5, 1.0, -0.5, 0.25, -0.1, 0.05, -0.025], dtype=float),
    ], dtype=float)

    # Zero-mean then unit-norm each archetype
    raw -= raw.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    # Guard against degenerate archetypes (should not happen with these shapes)
    norms = np.where(norms == 0, 1.0, norms)
    return raw / norms


# Compile-time constant — shape (5, 10), each row is unit-norm, zero-mean
_CODEBOOK: np.ndarray = _make_codebook()

# Convenience index constants for the three named archetypes
_IDX_UPTREND  = 0   # pcb_corr_uptrend_10
_IDX_VBOTTOM  = 2   # pcb_corr_vbottom_10
_IDX_MEANREV  = 4   # pcb_corr_meanrevert_10

_WINDOW = 10

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name":        "paper_2605_13407_pattern_codebook",
    "description": (
        "Deterministic VQ-inspired pattern codebook: z-normalises each trailing "
        "10-day log-return window, correlates it against 5 hard-coded archetype "
        "shapes (uptrend, downtrend, V-bottom, inverted-V, oscillation), and emits "
        "best-match correlation, archetype id, quantisation error, and three named "
        "archetype correlations."
    ),
    "requires":    ["Close"],
    "produces":    [
        "pcb_best_match_corr_10",
        "pcb_best_match_id_10",
        "pcb_quant_error_10",
        "pcb_corr_uptrend_10",
        "pcb_corr_vbottom_10",
        "pcb_corr_meanrevert_10",
    ],
    "tags":        ["momentum", "pattern", "mean_reversion", "experimental"],
    "version":     "1.0",
    "author":      "paper arxiv:2605.13407 proxy (deterministic codebook)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _znorm(arr: np.ndarray) -> np.ndarray:
    """Z-normalise a 1-D array. Returns all-NaN if std == 0."""
    mu = arr.mean()
    sd = arr.std()
    if sd == 0.0:
        return np.full_like(arr, np.nan, dtype=float)
    return (arr - mu) / sd


def _codebook_correlations(window: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between `window` (length W) and each of the K
    rows of `codebook` (shape K x W).

    Both sides are unit-norm after z-normalisation, so correlation equals the
    dot product of the z-normalised window with each unit-norm archetype.

    Returns a length-K array. Returns all-NaN if the window has zero variance.
    """
    z = _znorm(window)
    if np.any(np.isnan(z)):
        return np.full(len(codebook), np.nan)
    # z has zero mean; dot with unit-norm zero-mean archetype = Pearson corr
    return codebook @ z / (len(z) - 1 + 1e-10) * (len(z) - 1)


# ---------------------------------------------------------------------------
# Vectorised rolling implementation
# ---------------------------------------------------------------------------

def _rolling_codebook(log_ret: np.ndarray, codebook: np.ndarray, window: int) -> np.ndarray:
    """
    Slide a rolling window of length `window` across `log_ret` (1-D float64).
    For each position t (0-indexed) where t >= window-1, compute the codebook
    correlations for log_ret[t-window+1 : t+1].

    Returns an array of shape (n, K) with NaN where t < window-1 or zero variance.
    """
    n = len(log_ret)
    k = len(codebook)
    out = np.full((n, k), np.nan)

    # Pre-compute the unit-norm denominator factor for the dot product.
    # Because the codebook rows are already unit-norm and zero-mean, and after
    # z-normalisation the window vector is also zero-mean, the Pearson correlation
    # between the z-normed window and archetype a_i is:
    #   corr = sum(z * a_i) / (W - 1)  ... but we want corr ∈ [-1, 1].
    # Actually: since a_i is unit-norm (||a_i||=1) and z-normed x has ||x||=sqrt(W-1)
    # when we normalise by std (not std/sqrt(W-1)), the dot product = corr * sqrt(W-1).
    # To get true Pearson corr we divide by sqrt(W-1). We bake this into a single
    # matrix multiply below.

    # Pre-normalise codebook to match our z-norm convention (std = sample std):
    #   z = (x - mean) / std  => ||z||^2 = W - 1  (sample variance denominator W-1)
    # So corr = (z . a_i) / ||z|| / ||a_i||
    #         = (z . a_i) / sqrt(W-1)   (since a_i is unit-norm built the same way)
    denom = np.sqrt(window - 1) if window > 1 else 1.0

    # Stride-trick: build a view of rolling windows to avoid a Python loop
    if n < window:
        return out

    # Use a strided view for efficiency
    shape   = (n - window + 1, window)
    strides = (log_ret.strides[0], log_ret.strides[0])
    windows = np.lib.stride_tricks.as_strided(log_ret, shape=shape, strides=strides)
    # windows[i] = log_ret[i : i+window], shape (n-window+1, window)

    # Vectorised z-normalise all windows at once
    mu = windows.mean(axis=1, keepdims=True)     # (n-window+1, 1)
    sd = windows.std(axis=1, keepdims=True)       # sample std (ddof=0)

    # Mark zero-variance rows; we will leave those as NaN
    valid = (sd > 0).ravel()                      # (n-window+1,)
    sd_safe = np.where(sd > 0, sd, 1.0)

    z_windows = (windows - mu) / sd_safe          # (n-window+1, window)

    # Correlation: z_windows @ codebook.T / denom  shape (n-window+1, K)
    corrs = (z_windows @ codebook.T) / denom      # (n-window+1, K)

    # Clip to [-1, 1] to handle minor floating-point overflow
    corrs = np.clip(corrs, -1.0, 1.0)

    # Mask zero-variance rows
    corrs[~valid] = np.nan

    # Write into output aligned to the LAST element of each window
    out[window - 1:] = corrs

    return out


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PRISM-VQ-inspired pattern codebook features on per-ticker OHLCV data.

    For each row t, takes the trailing 10-day window of daily log-returns ending at t,
    z-normalises it, and correlates it against the 5 hard-coded archetype shape vectors.
    Produces 6 new columns (all prefixed pcb_).  Leading 10 rows are NaN (warm-up).
    """
    close = df["Close"].to_numpy(dtype=np.float64)

    # Log-returns: ret[t] = log(Close[t] / Close[t-1]).  ret[0] = NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.empty(len(close), dtype=np.float64)
        log_ret[0] = np.nan
        valid_mask = (close[:-1] > 0) & (close[1:] > 0)
        log_ret[1:] = np.where(
            valid_mask,
            np.log(close[1:] / np.where(valid_mask, close[:-1], 1.0)),
            np.nan,
        )

    # Rolling codebook correlations: shape (n, 5)
    corrs = _rolling_codebook(log_ret, _CODEBOOK, _WINDOW)

    # Derived columns
    n_rows = len(close)
    all_nan_rows = np.all(np.isnan(corrs), axis=1)   # shape (n,)

    # best_match_corr: max correlation — compute row-wise to avoid nanmax warning
    best_corr = np.full(n_rows, np.nan)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        with np.errstate(all="ignore"):
            best_corr[valid_rows] = np.nanmax(corrs[valid_rows], axis=1)

    # best_match_id: argmax only on rows that have at least one finite correlation
    best_id = np.full(n_rows, np.nan)
    if np.any(valid_rows):
        with np.errstate(all="ignore"):
            best_id[valid_rows] = np.nanargmax(corrs[valid_rows], axis=1).astype(float)

    # quant_error = 1 - best_match_corr  (in [0, 2]; 0 = perfect match)
    quant_error = np.where(np.isnan(best_corr), np.nan, 1.0 - best_corr)

    df["pcb_best_match_corr_10"] = pd.array(best_corr, dtype="Float64")
    df["pcb_best_match_id_10"]   = pd.array(best_id,   dtype="Float64")
    df["pcb_quant_error_10"]     = pd.array(quant_error, dtype="Float64")
    df["pcb_corr_uptrend_10"]    = pd.array(corrs[:, _IDX_UPTREND],  dtype="Float64")
    df["pcb_corr_vbottom_10"]    = pd.array(corrs[:, _IDX_VBOTTOM],  dtype="Float64")
    df["pcb_corr_meanrevert_10"] = pd.array(corrs[:, _IDX_MEANREV],  dtype="Float64")

    return df
