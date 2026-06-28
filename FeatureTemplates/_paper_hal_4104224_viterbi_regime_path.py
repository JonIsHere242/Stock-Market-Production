"""
_paper_hal_4104224_viterbi_regime_path.py
------------------------------------------
Per-ticker causal 2-state Gaussian HMM decoded by Viterbi, tracking
low-vol vs high-vol market regimes.  Inspired by:

  "A New Viterbi-Based Decoding Strategy for Market Risk Tracking"
  HAL: hal-04104224

CAUSALITY GUARANTEE
-------------------
All outputs at time t are computed using ONLY data up to and including t.

Two-tier scheme:
  REFIT (every REFIT_STEP bars):
    Re-estimate HMM params via Baum-Welch EM on a trailing WIN-bar window of
    absolute log-returns ending at bar t-1.  Run Viterbi to pin the anchor
    state; reset the forward-filter to that state.

  BETWEEN REFITS (every bar):
    Step the forward filter one observation forward using cached HMM params.
    O(K^2) per bar -- no window scan.  MAP state = argmax of forward posterior.

Bit-identical re-run guarantee: every bar's state is a pure function of
abs_ret[0..t-1] (strictly causal).  Truncating to K rows leaves all rows
0..K-1 bit-identical because no row's computation touches rows > K-1.

IMPLEMENTATION NOTE
-------------------
hmmlearn and sklearn are NOT used.  Baum-Welch EM (2-state diagonal Gaussian)
is implemented in scaled probability space (avoids log per step, uses
vectorised xi_sum over (T-1, K, K)).  Forward filter and Viterbi also run in
scaled space for speed.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "_paper_hal_4104224_viterbi_regime_path",
    "description": (
        "Per-ticker causal 2-state Gaussian HMM + Viterbi path decoding: "
        "each day's regime label is decoded using only a trailing window of "
        "past returns, yielding a look-ahead-free low-vol/high-vol state path "
        "with dwell time, posterior probability, and transition estimates."
    ),
    "requires": ["Close"],
    "produces": [
        "vit_regime",          # int 0/1: 0=low-vol, 1=high-vol
        "vit_prob_highvol",    # float [0,1]: causal forward posterior P(state=1)
        "vit_dwell",           # int: consecutive days in current regime
        "vit_vol_gap",         # float: high-state sigma minus low-state sigma
        "vit_persist",         # float: estimated self-transition P(s->s)
    ],
    "tags": ["market_regime", "volatility", "experimental"],
    "version": "1.3",
    "author": "paper proxy: hal-04104224, hand-coded HMM+Viterbi (no hmmlearn)",
}

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
WIN         = 60    # trailing window length for HMM fit (abs log-return obs)
EM_ITER     = 2     # Baum-Welch iterations per refit (keep fast; 2 steps converge well for sticky regimes)
REFIT_STEP  = 25    # re-estimate HMM params every N bars (causal; ~26 refits per 700-row series)
_EPS        = 1e-15
_MIN_STD    = 1e-6


# ---------------------------------------------------------------------------
# HMM primitives — scaled probability space, K=2 specialised
# ---------------------------------------------------------------------------

def _gauss_pdf_mat(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Gaussian pdf for all (t, k).
    obs: (T,), mu/sigma: (K,) -> returns (T, K), all positive.
    """
    sg = np.maximum(sigma, _MIN_STD)
    z = (obs[:, None] - mu[None, :]) / sg[None, :]
    return np.exp(-0.5 * z * z) / (sg * np.sqrt(2.0 * np.pi))


def _gauss_pdf_one(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Gaussian pdf for a single scalar x.  Returns (K,)."""
    sg = np.maximum(sigma, _MIN_STD)
    z = (x - mu) / sg
    return np.exp(-0.5 * z * z) / (sg * np.sqrt(2.0 * np.pi))


def _forward_scaled(
    lb: np.ndarray,   # (T, K) emission probs (positive)
    pi: np.ndarray,   # (K,)
    A: np.ndarray,    # (K, K)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled forward algorithm in probability space.
    Returns (alpha_scaled (T,K), log_scales (T,)).
    alpha_scaled[t] sums to 1 (normalised).
    log_lik = sum(log_scales).
    """
    T, K = lb.shape
    alpha = np.empty((T, K))
    log_scales = np.empty(T)

    a = pi * lb[0]
    c = a.sum() + _EPS
    alpha[0] = a / c
    log_scales[0] = np.log(c)

    for t in range(1, T):
        # (K,) = alpha[t-1] @ A  (row vector times transition matrix)
        a = (alpha[t - 1] @ A) * lb[t]
        c = a.sum() + _EPS
        alpha[t] = a / c
        log_scales[t] = np.log(c)

    return alpha, log_scales


def _backward_scaled(
    lb: np.ndarray,      # (T, K)
    A: np.ndarray,       # (K, K)
    log_scales: np.ndarray,
) -> np.ndarray:
    """Scaled backward algorithm.  Returns beta_scaled (T, K)."""
    T, K = lb.shape
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        # beta[t, i] = sum_j A[i,j] * lb[t+1,j] * beta[t+1,j]
        # normalised by scale at t+1
        b = (A * lb[t + 1][None, :] * beta[t + 1][None, :]).sum(axis=1)
        scale_t1 = np.exp(log_scales[t + 1]) + _EPS
        beta[t] = b / scale_t1
    return beta


def _em_step(
    obs: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One Baum-Welch EM step.  Returns (pi, A, mu, sigma, log_lik)."""
    lb = _gauss_pdf_mat(obs, mu, sigma)
    alpha, log_scales = _forward_scaled(lb, pi, A)
    beta = _backward_scaled(lb, A, log_scales)

    log_lik = float(log_scales.sum())

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + _EPS

    # Xi sum: vectorised over all (T-1) timesteps
    # xi[t,i,j] ~ alpha[t,i] * A[i,j] * lb[t+1,j] * beta[t+1,j]
    xi_unnorm = (
        alpha[:-1, :, None]
        * A[None, :, :]
        * lb[1:, None, :]
        * beta[1:, None, :]
    )                                                  # (T-1, K, K)
    row_sum = xi_unnorm.sum(axis=(1, 2), keepdims=True) + _EPS
    xi_unnorm /= row_sum
    xi_sum = xi_unnorm.sum(axis=0)                    # (K, K)

    # M-step
    new_pi = np.clip(gamma[0], _EPS, None)
    new_pi /= new_pi.sum()

    new_A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + _EPS)
    new_A = np.clip(new_A, _EPS, None)
    new_A /= new_A.sum(axis=1, keepdims=True)

    g_sum = gamma.sum(axis=0) + _EPS
    new_mu = (gamma * obs[:, None]).sum(axis=0) / g_sum
    diff = obs[:, None] - new_mu[None, :]
    new_sigma = np.sqrt((gamma * diff ** 2).sum(axis=0) / g_sum)
    new_sigma = np.clip(new_sigma, _MIN_STD, None)

    return new_pi, new_A, new_mu, new_sigma, log_lik


def _fit_hmm(
    obs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit 2-state Gaussian HMM via Baum-Welch EM, deterministic init.
    State 0 = low-vol, state 1 = high-vol (by emission mean ordering).
    Returns (pi, A, mu, sigma).
    """
    med = np.median(obs)
    lo, hi = obs <= med, obs > med

    mu = np.array([
        obs[lo].mean() if lo.any() else obs.mean() * 0.7,
        obs[hi].mean() if hi.any() else obs.mean() * 1.3,
    ])
    sigma = np.array([
        obs[lo].std() if lo.sum() > 1 else abs(mu[0]) * 0.5 + _MIN_STD,
        obs[hi].std() if hi.sum() > 1 else abs(mu[1]) * 0.5 + _MIN_STD,
    ])
    sigma = np.clip(sigma, _MIN_STD, None)

    if mu[0] > mu[1]:
        mu, sigma = mu[::-1].copy(), sigma[::-1].copy()

    pi = np.array([0.5, 0.5])
    A  = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)

    prev_ll = -np.inf
    for _ in range(EM_ITER):
        try:
            pi, A, mu, sigma, ll = _em_step(obs, pi, A, mu, sigma)
        except Exception:       # noqa: BLE001
            break
        if abs(ll - prev_ll) < 1e-6 * (1.0 + abs(ll)):
            break
        prev_ll = ll

    # Re-enforce state ordering
    if mu[0] > mu[1]:
        mu    = mu[::-1].copy()
        sigma = sigma[::-1].copy()
        pi    = pi[::-1].copy()
        A     = A[::-1, :][:, ::-1].copy()

    return pi, A, mu, sigma


def _viterbi_last(
    lb: np.ndarray,    # (T, K) emission probs (positive)
    pi: np.ndarray,
    A: np.ndarray,
) -> int:
    """Viterbi in scaled probability space; returns only the last-step MAP state."""
    T, K = lb.shape
    delta = np.empty((T, K))
    psi   = np.zeros((T, K), dtype=np.int32)

    d = pi * lb[0]
    c = d.max() + _EPS
    delta[0] = d / c

    for t in range(1, T):
        scores = delta[t - 1][:, None] * A   # (K, K)
        psi[t]   = scores.argmax(axis=0)
        d = scores.max(axis=0) * lb[t]
        c = d.max() + _EPS
        delta[t] = d / c

    return int(np.argmax(delta[-1]))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Causal 2-state Gaussian HMM + Viterbi regime path on per-ticker close returns.

    HMM params are re-estimated every REFIT_STEP bars on a trailing WIN-bar
    window (absolute log-returns, strictly causal: data[t-WIN..t-1]).  Between
    refits the forward filter is stepped one observation at a time -- O(K^2)
    per bar, no window scan.  MAP state = argmax of the forward posterior.

    All columns in df are preserved unchanged.  Exactly the produces columns
    are appended.  Leading WIN bars are NaN.  No inf in output.
    """
    n = len(df)

    vit_regime   = np.full(n, np.nan)
    vit_prob_hv  = np.full(n, np.nan)
    vit_dwell    = np.full(n, np.nan)
    vit_vol_gap  = np.full(n, np.nan)
    vit_persist  = np.full(n, np.nan)

    close = df["Close"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret = np.log(close[1:] / close[:-1])
    abs_ret = np.abs(log_ret)   # length n-1; abs_ret[i] = |ret arriving at bar i+1|

    # HMM parameter cache
    c_pi: Optional[np.ndarray] = None
    c_A:  Optional[np.ndarray] = None
    c_mu: Optional[np.ndarray] = None
    c_sg: Optional[np.ndarray] = None

    # Forward filter state in probability space (posterior at current bar), shape (K,)
    fwd: Optional[np.ndarray] = None

    current_regime: Optional[int] = None
    dwell_count: int = 0
    next_refit: int = WIN    # bar index at which next refit is due

    for t in range(WIN, n):
        # ── REFIT: re-estimate HMM on trailing window ─────────────────────
        if t >= next_refit or c_pi is None:
            # window = abs_ret[t-WIN .. t-1]  (WIN obs; causal at bar t)
            window = abs_ret[t - WIN: t]

            if np.isfinite(window).all() and window.std() >= _MIN_STD:
                try:
                    pi, A, mu, sg = _fit_hmm(window)
                    c_pi, c_A, c_mu, c_sg = pi, A, mu, sg

                    # Viterbi on the fit window to anchor the forward filter state
                    lb_win = _gauss_pdf_mat(window, c_mu, c_sg)
                    state_anchor = _viterbi_last(lb_win, c_pi, c_A)

                    # Seed forward filter as a near-delta at the anchor state
                    if state_anchor == 0:
                        fwd = np.array([0.99, 0.01])
                    else:
                        fwd = np.array([0.01, 0.99])

                except Exception:   # noqa: BLE001
                    pass

            next_refit = t + REFIT_STEP

        if c_pi is None or fwd is None:
            continue

        # ── INCREMENTAL FORWARD STEP ──────────────────────────────────────
        # obs at bar t = abs_ret[t-1] (return that completed at bar t)
        x_t = abs_ret[t - 1]
        if not np.isfinite(x_t):
            continue

        lb_t = _gauss_pdf_one(x_t, c_mu, c_sg)    # (K,), positive

        # fwd_new[j] = sum_i fwd[i] * A[i,j] * lb_t[j]
        fwd_new = (fwd @ c_A) * lb_t               # (K,)
        norm = fwd_new.sum() + _EPS
        fwd = fwd_new / norm                        # re-normalise

        # MAP state and posterior P(high-vol)
        state_t = int(np.argmax(fwd))
        prob_hv = float(fwd[1])

        # Dwell counter
        if state_t == current_regime:
            dwell_count += 1
        else:
            current_regime = state_t
            dwell_count    = 1

        vit_regime[t]  = float(state_t)
        vit_prob_hv[t] = float(np.clip(prob_hv, 0.0, 1.0))
        vit_dwell[t]   = float(dwell_count)
        vit_vol_gap[t] = float(np.clip(c_sg[1] - c_sg[0], 0.0, None))
        vit_persist[t] = float(np.clip(c_A[state_t, state_t], 0.0, 1.0))

    # Safety: flush any lingering inf -> nan
    def _clean(arr: np.ndarray) -> np.ndarray:
        arr[~np.isfinite(arr)] = np.nan
        return arr

    new_cols = pd.DataFrame(
        {
            "vit_regime":       _clean(vit_regime),
            "vit_prob_highvol": _clean(vit_prob_hv),
            "vit_dwell":        _clean(vit_dwell),
            "vit_vol_gap":      _clean(vit_vol_gap),
            "vit_persist":      _clean(vit_persist),
        },
        index=df.index,
    )

    return pd.concat([df, new_cols], axis=1)
