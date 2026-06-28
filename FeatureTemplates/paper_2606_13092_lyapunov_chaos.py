"""
Lyapunov Exponent / Chaos Predictability Features.

PAPER: "Scale Buys Interpolation, Structure Buys a Horizon: Certified
        Predictability for Equivariant World Models" (arxiv 2606.13092).

The paper's core COMPUTABLE method: the Lyapunov spectrum — how quickly nearby
trajectories diverge.  Predictable horizon T(ε) ~ log(1/ε) / λ.

Applied to OHLCV:
  - FINITE-TIME LYAPUNOV EXPONENTS (FTLE) from delay-embedding of return series.
  - Estimated via mean log-divergence of nearest-neighbor trajectory pairs.

VECTORIZED IMPLEMENTATION for speed:
  - Build entire embedding matrix once per rolling window.
  - Vectorized distance computation (no Python loop over pairs).
  - Sample 20 reference points per window only.

Features:
  1. ftle_ret: FTLE of log-return embedding.
  2. ftle_hl: FTLE of HL-range normalized signal.
  3. ftle_vol_adj: volatility-adjusted FTLE (chaos above baseline vol).
  4. ftle_pred_horizon: predictable horizon in bars.
  5. ftle_lyap_regime: z-score of ftle_ret (relative chaos level).
  6. ftle_chaos_accel: derivative of smoothed FTLE.
  7. ftle_return_predictability: -ftle_ret smoothed (high = more predictable).
  8. ftle_hl_corr: rolling correlation between ftle_ret and ftle_hl.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13092_lyapunov_chaos",
    "description": (
        "Finite-time Lyapunov exponents from return/HL trajectory divergence: "
        "chaos level, predictable horizon, volatility-adjusted chaos, regime "
        "z-score. Inspired by Lyapunov spectrum / certified predictability in "
        "world models (arxiv 2606.13092)."
    ),
    "requires": ["High", "Low", "Close"],
    "produces": [
        "ftle_ret",
        "ftle_hl",
        "ftle_vol_adj",
        "ftle_pred_horizon",
        "ftle_lyap_regime",
        "ftle_chaos_accel",
        "ftle_return_predictability",
        "ftle_hl_corr",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.13092",
}

# Parameters
_EMBED_DIM = 3
_EVOLVE = 3
_WIN = 60         # rolling window (number of embedded points)
_N_REF = 20       # reference points per window (subsampled for speed)
_STEP = _WIN // 4  # recompute every STEP bars


def _ftle_vectorized(series: np.ndarray, embed_dim: int, evolve: int,
                     win: int, n_ref: int, recompute_step: int) -> np.ndarray:
    """
    Vectorized rolling FTLE. Recomputes every recompute_step bars.
    Uses matrix broadcasting for pairwise distances — no inner Python loops.
    """
    n = len(series)
    out = np.full(n, np.nan)
    last_val = np.nan

    for i in range(win + embed_dim - 1, n):
        # Only recompute every recompute_step bars
        if (i % recompute_step) != 0 and not np.isnan(last_val):
            out[i] = last_val
            continue

        # Extract window of series
        seg = series[i - win - embed_dim + 1: i + 1]
        if np.any(~np.isfinite(seg)):
            continue

        # Build delay-embedding matrix: shape (win, embed_dim)
        m = win - evolve
        if m < n_ref + 5:
            continue
        cols = [seg[j: j + m] for j in range(embed_dim)]
        E = np.column_stack(cols)  # (m, embed_dim)

        # Sample n_ref reference indices (evenly spaced, avoid edges)
        ref_idx = np.linspace(0, m - evolve - 1, n_ref, dtype=int)
        ref_idx = np.unique(ref_idx)
        E_ref = E[ref_idx]  # (n_ref, embed_dim)

        # Vectorized pairwise distances: E_ref vs all of E
        # (n_ref, m) distance matrix
        diff = E_ref[:, np.newaxis, :] - E[np.newaxis, :, :]  # (n_ref, m, d)
        dists2 = np.sum(diff ** 2, axis=2)  # (n_ref, m)

        # Exclude self by setting diagonal to inf
        for k, ri in enumerate(ref_idx):
            dists2[k, ri] = np.inf

        # Nearest neighbor index for each reference point
        nn_idx = np.argmin(dists2, axis=1)  # (n_ref,)
        d0_sq = dists2[np.arange(n_ref), nn_idx]  # (n_ref,)

        # Evolved indices
        ref_ev = ref_idx + evolve
        nn_ev = nn_idx + evolve

        # Valid pairs (all within bounds)
        valid = (ref_ev < m) & (nn_ev < m) & (d0_sq > 1e-24)
        if valid.sum() < 3:
            continue

        # Evolved distances
        d_ev_sq = np.sum((E[ref_ev[valid]] - E[nn_ev[valid]]) ** 2, axis=1)
        d_ev_sq = np.where(d_ev_sq > 1e-24, d_ev_sq, 1e-24)

        log_divs = 0.5 * (np.log(d_ev_sq) - np.log(d0_sq[valid]))
        ftle_val = float(np.mean(log_divs)) / evolve

        out[i] = ftle_val
        last_val = ftle_val

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(close)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret = np.full(n, np.nan)
        ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan))

    # HL normalized range
    hl = np.where(close > 0, (high - low) / close, np.nan)

    # --- FTLE on returns ---
    ftle_ret = _ftle_vectorized(ret, _EMBED_DIM, _EVOLVE, _WIN, _N_REF, _STEP)

    # --- FTLE on HL range ---
    ftle_hl_arr = _ftle_vectorized(hl, _EMBED_DIM, _EVOLVE, _WIN, _N_REF, _STEP)

    df["ftle_ret"] = ftle_ret
    df["ftle_hl"] = ftle_hl_arr

    ret_s = pd.Series(ret)
    ftle_s = pd.Series(ftle_ret)
    ftle_hl_s = pd.Series(ftle_hl_arr)

    # --- Volatility-adjusted FTLE ---
    roll_vol = ret_s.rolling(21, min_periods=5).std()
    df["ftle_vol_adj"] = (ftle_s / (roll_vol + 1e-12)).values

    # --- Predictable horizon ---
    eps = 0.01
    lam_abs = np.where(np.abs(ftle_ret) > 1e-6, np.abs(ftle_ret), 1e-6)
    ph = np.where(np.isfinite(ftle_ret), np.clip(np.log(1.0 / eps) / lam_abs, 0, 300), np.nan)
    df["ftle_pred_horizon"] = ph

    # --- Lyapunov regime z-score ---
    mu = ftle_s.rolling(126, min_periods=30).mean()
    std = ftle_s.rolling(126, min_periods=30).std()
    df["ftle_lyap_regime"] = ((ftle_s - mu) / (std + 1e-12)).values

    # --- Chaos acceleration ---
    ftle_smooth = ftle_s.rolling(5, min_periods=2).mean()
    df["ftle_chaos_accel"] = ftle_smooth.diff(3).values

    # --- Return predictability (negative chaos = more predictable) ---
    df["ftle_return_predictability"] = (-ftle_s).rolling(5, min_periods=2).mean().values

    # --- Directed chaos: vol-adj chaos weighted by return sign ---
    # Sign of recent return determines if chaos is "upward" or "downward"
    ret_sign = np.sign(ret_s.rolling(3, min_periods=1).mean())
    ftle_va_s = pd.Series(df["ftle_vol_adj"].values)
    df["ftle_hl_corr"] = (ftle_va_s * ret_sign).values

    return df
