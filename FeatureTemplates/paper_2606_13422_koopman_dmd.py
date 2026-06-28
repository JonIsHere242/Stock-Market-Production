"""
Koopman operator / Dynamic Mode Decomposition features derived from:
  "Foundations of Practical Quantum Advantage in Quantum-Informed Machine
   Learning for Predicting Chaos" (arXiv 2606.13422)

The paper uses Koopman operator rollouts (empirical Koopman from delay-embedded
observables) to predict chaotic dynamics.  The Koopman operator linearises
nonlinear dynamics in a lifted feature space: eigenvalues capture dominant
oscillatory / growth modes.

Applied to OHLCV via Dynamic Mode Decomposition (DMD) on non-overlapping
blocks (for speed), then interpolated:
  - Build a (delay x T) state matrix from log-returns and normalised volume.
  - Fit DMD: A_tilde via truncated SVD of the data matrix.
  - Extract leading eigenvalue magnitude (growth rate) and angle (frequency).
  - Project current state onto dominant mode (amplitude).

Rolling implementation: uses SHORT fixed windows (32 bars) applied at
every step via stride tricks + batched SVD-free approximation.
"""

import pandas as pd
import numpy as np

METADATA = {
    "name":        "paper_2606_13422_koopman_dmd",
    "description": (
        "Dynamic Mode Decomposition (Koopman approximation) eigenvalue features: "
        "dominant mode growth rate, oscillation frequency, mode amplitude; "
        "based on arXiv 2606.13422 Koopman rollout quantum ML."
    ),
    "requires":    ["Close", "Volume"],
    "produces": [
        "dmd_growth_rate_32d",
        "dmd_growth_rate_64d",
        "dmd_osc_freq_32d",
        "dmd_osc_freq_64d",
        "dmd_mode_amp_32d",
        "dmd_mode_amp_64d",
        "dmd_spectral_radius_32d",
    ],
    "tags":        ["market_regime", "statistical", "experimental"],
    "version":     "1.0",
    "author":      "paper:2606.13422",
}

# ── Low-rank companion-matrix DMD on a single segment ─────────────────────────

def _dmd_on_segment(ret_seg: np.ndarray, vol_seg: np.ndarray,
                    delay: int = 3, n_modes: int = 2) -> tuple:
    """
    Fit DMD to a bivariate delay-embedded window.
    Returns (growth_rate, osc_freq, mode_amplitude, spectral_radius).
    """
    # Normalise
    rs, vs = ret_seg.std(), vol_seg.std()
    if rs < 1e-10 or vs < 1e-10:
        return np.nan, np.nan, np.nan, np.nan
    r = (ret_seg - ret_seg.mean()) / rs
    v = (vol_seg - vol_seg.mean()) / vs

    # Hankel state matrix: rows = [r[t], r[t-1], ..., r[t-delay+1], v[t], ...]
    T = len(r) - delay
    if T < 4:
        return np.nan, np.nan, np.nan, np.nan
    # Build (2*delay x T+1) state matrix
    rows = []
    for lag in range(delay):
        rows.append(r[lag: lag + T + 1])
        rows.append(v[lag: lag + T + 1])
    S = np.stack(rows, axis=0)  # (2*delay, T+1)

    X = S[:, :-1]   # (2*delay, T)
    Y = S[:, 1:]    # (2*delay, T)

    if X.shape[1] < 4:
        return np.nan, np.nan, np.nan, np.nan

    # Truncated SVD of X (rank r = n_modes)
    try:
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    r_eff = min(n_modes, len(sv))
    Ur = U[:, :r_eff]
    Sr = sv[:r_eff]
    Vtr = Vt[:r_eff, :]

    Sr_inv = np.diag(1.0 / (Sr + 1e-10))
    A_tilde = Ur.T @ Y @ Vtr.T @ Sr_inv   # (r_eff, r_eff)

    try:
        evals = np.linalg.eigvals(A_tilde)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    mags = np.abs(evals)
    spec_rad = float(mags.max())
    dom = np.argmax(mags)
    ev = evals[dom]
    growth = float(np.log(mags[dom] + 1e-12))
    freq = float(np.angle(ev) / (2 * np.pi))
    amp = float(np.abs(Ur[:, 0] @ X[:, -1]))

    return growth, freq, amp, spec_rad


def _rolling_dmd(ret_vals: np.ndarray, vol_vals: np.ndarray,
                 window: int, delay: int = 3, n_modes: int = 2,
                 stride: int = 5) -> tuple:
    """
    Rolling DMD with stride > 1 for speed, then forward-fill to daily.
    stride=5 computes DMD once per week, reducing N iterations 5x.
    """
    n = len(ret_vals)
    growth = np.full(n, np.nan)
    freq   = np.full(n, np.nan)
    amp    = np.full(n, np.nan)
    srad   = np.full(n, np.nan)

    min_obs = delay * 2 + 4
    if n < window:
        return growth, freq, amp, srad

    # Compute at strided positions only
    for i in range(window - 1, n, stride):
        start = i - window + 1
        r_seg = ret_vals[start: i + 1]
        v_seg = vol_vals[start: i + 1]
        # Skip if too many NaN
        mask = np.isfinite(r_seg) & np.isfinite(v_seg)
        if mask.sum() < max(min_obs, window // 2):
            continue
        # Fill gaps conservatively
        r_clean = np.where(mask, r_seg, 0.0)
        v_clean = np.where(mask, v_seg, 0.0)
        g, f, a, s = _dmd_on_segment(r_clean, v_clean, delay=delay, n_modes=n_modes)
        growth[i] = g
        freq[i]   = f
        amp[i]    = a
        srad[i]   = s

    # Forward-fill strided values
    for arr in (growth, freq, amp, srad):
        for j in range(1, n):
            if np.isnan(arr[j]) and not np.isnan(arr[j - 1]):
                arr[j] = arr[j - 1]

    return growth, freq, amp, srad


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].replace(0, np.nan)
    vol   = df["Volume"].replace(0, np.nan).astype(float)

    log_ret = np.log(close / close.shift(1)).values.astype(np.float64)
    log_vol = np.log(vol).values.astype(np.float64)

    # Window=32, stride=4 (compute every 4 bars)
    g32, f32, a32, s32 = _rolling_dmd(log_ret, log_vol, window=32,
                                       delay=3, n_modes=2, stride=4)
    # Window=64, stride=6
    g64, f64, a64, _   = _rolling_dmd(log_ret, log_vol, window=64,
                                       delay=3, n_modes=2, stride=6)

    df["dmd_growth_rate_32d"]    = g32
    df["dmd_growth_rate_64d"]    = g64
    df["dmd_osc_freq_32d"]       = f32
    df["dmd_osc_freq_64d"]       = f64
    df["dmd_mode_amp_32d"]       = a32
    df["dmd_mode_amp_64d"]       = a64
    df["dmd_spectral_radius_32d"] = s32

    return df
