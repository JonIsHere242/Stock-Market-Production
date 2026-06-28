"""
Time-lagged cross-channel coupling strength, Marchenko-Pastur filtered, per-ticker proxy.
Derived from: "A Stationarity-and-Coupling Criterion for Training-Free
Time-Lagged Spectral Embeddings of Multivariate Time Series" (arxiv:2606.13823).

The paper builds descriptor D(τ) from a TIME-LAGGED correlation matrix between
channels of a multivariate time series, truncated at the Marchenko-Pastur (MP)
noise edge so that only signal-bearing eigenvalues survive.  The core insight:
the lag-τ cross-correlation matrix captures temporal COUPLING BETWEEN channels,
distinct from instantaneous co-movement (zero-lag PCA / ssp_ blocks).

OHLCV adaptation:
  - Treat k=5 OHLCV-derived channels {log_ret, hl_range, body_frac, log_vol_chg,
    upper_shadow} as the multivariate input.
  - In a rolling window W, build the k×k time-lagged cross-correlation matrix
    C(τ)_{ij} = corr(ch_i[t], ch_j[t-τ]) — standardised so entries are Pearson r.
  - Apply a noise-floor threshold: under H0 (iid channels), each squared singular
    value of C(τ) has expectation k/n_eff ≈ 0.083 for k=5, n=59.
    We call a singular value "signal-bearing" if sv² > k/n_eff  (MP-inspired).
  - Extract: sum of signal sv² (coupling energy), count, top sv², ratio, and a
    normalised coupling measure (Frobenius² / null expectation).
  - Lag-2 panel gives decay rate: fast vs slow coupling structure.
  - Rolling z-score of coupling energy = regime signal.

Unlike ssp_ blocks (zero-lag PCA), this captures how much one OHLCV dimension at
bar t predicts another at bar t+1 — a genuine causal predictability signal.
All rolling, causal, no lookahead.
"""
import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_13823_timelag_mp_coupling",
    "description": (
        "Rolling time-lagged cross-channel coupling (k=5 OHLCV channels) with "
        "Marchenko-Pastur noise-floor thresholding: coupling energy, signal rank, "
        "top squared singular value, normalized Frobenius coupling ratio, lag-2 "
        "energy, lag decay, and rolling z-score. Based on arXiv 2606.13823."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "tlmp_signal_energy",     # sum of signal-bearing sv² at lag-1
        "tlmp_signal_rank",       # count of signal-bearing sv² at lag-1
        "tlmp_top_sv2",           # largest squared singular value of C(1)
        "tlmp_frob_ratio",        # Frobenius² / null expectation (coupling vs noise)
        "tlmp_lag2_energy",       # signal_energy at lag-2
        "tlmp_lag_decay",         # ratio lag1 / lag2 signal energy (decay speed)
        "tlmp_energy_zscore",     # rolling z-score of lag-1 signal_energy
    ],
    "tags": ["experimental", "spectral", "coupling"],
    "version": "1.0",
    "author": "paper-mining slate 4",
}

_WINDOW = 60    # rolling estimation window
_K = 5          # number of OHLCV-derived channels
_EPS = 1e-10


def _build_channels(close, open_, high, low, volume, n):
    """Return (n, k) float64 array of OHLCV-derived signal channels."""
    log_ret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(
            np.where(close[:-1] > 0, close[1:] / close[:-1], np.nan)
        )
    hl_range = np.where(close > 0, (high - low) / close, np.nan)
    spread = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        body_frac = np.where(spread > _EPS, (close - open_) / spread, np.nan)
    log_vol = np.log(np.where(volume > 0, volume, np.nan))
    log_vol_chg = np.full(n, np.nan)
    log_vol_chg[1:] = np.diff(log_vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = np.where(
            spread > _EPS,
            (high - np.maximum(close, open_)) / spread,
            np.nan,
        )
    return np.column_stack([log_ret, hl_range, body_frac, log_vol_chg, upper])


def _std_cols(M):
    """Column-wise standardization (safe std)."""
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd = np.where(sd > _EPS, sd, 1.0)
    return (M - mu) / sd


def _lag_coupling_stats(seg, lag):
    """
    Compute time-lagged coupling statistics for a window segment.

    Returns (signal_energy, signal_rank, top_sv2, frob_ratio)
    or (nan, nan, nan, nan) if degenerate.
    """
    W = seg.shape[0]
    k = seg.shape[1]
    if W <= lag + k + 1:
        return np.nan, np.nan, np.nan, np.nan

    X_lead = seg[lag:]    # "future" side: rows τ..W-1
    X_lag  = seg[:-lag]   # "past" side:   rows 0..W-τ-1

    Xs = _std_cols(X_lead)
    Ys = _std_cols(X_lag)

    # Keep only rows where both sides are fully finite
    valid = (
        np.all(np.isfinite(Xs), axis=1) &
        np.all(np.isfinite(Ys), axis=1)
    )
    n_v = int(valid.sum())
    if n_v < max(k + 2, 12):
        return np.nan, np.nan, np.nan, np.nan

    Xs = Xs[valid]
    Ys = Ys[valid]

    # k×k lag cross-correlation matrix
    C = (Xs.T @ Ys) / (n_v - 1)   # shape (k, k)

    try:
        sv = np.linalg.svd(C, compute_uv=False)   # descending
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan

    sv2 = sv ** 2

    # Noise-floor threshold: under H0, E[sv²] ≈ k / n_v
    # (each entry of C ~ N(0, 1/n_v), Frobenius² ~ k²/n_v, spread over k sv's)
    noise_floor = k / n_v

    signal_mask   = sv2 > noise_floor
    signal_energy = float(sv2[signal_mask].sum()) if signal_mask.any() else 0.0
    signal_rank   = int(signal_mask.sum())
    top_sv2       = float(sv2[0])

    # Frobenius² vs null (k² / n_v)
    frob2        = float(np.sum(sv2))           # == ||C||_F²
    null_frob2   = (k ** 2) / n_v
    frob_ratio   = frob2 / null_frob2 if null_frob2 > _EPS else np.nan

    return signal_energy, signal_rank, top_sv2, frob_ratio


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    X = _build_channels(
        df["Close"].values.astype(float),
        df["Open"].values.astype(float),
        df["High"].values.astype(float),
        df["Low"].values.astype(float),
        df["Volume"].values.astype(float),
        n,
    )

    sig_energy   = np.full(n, np.nan)
    sig_rank     = np.full(n, np.nan)
    top_sv2      = np.full(n, np.nan)
    frob_ratio   = np.full(n, np.nan)
    lag2_energy  = np.full(n, np.nan)

    for i in range(_WINDOW - 1, n):
        seg = X[i - _WINDOW + 1: i + 1]   # (_WINDOW, k)

        e1, r1, tv1, fr1 = _lag_coupling_stats(seg, lag=1)
        sig_energy[i]  = e1
        sig_rank[i]    = r1
        top_sv2[i]     = tv1
        frob_ratio[i]  = fr1

        e2, _, _, _ = _lag_coupling_stats(seg, lag=2)
        lag2_energy[i] = e2

    df["tlmp_signal_energy"] = sig_energy
    df["tlmp_signal_rank"]   = sig_rank
    df["tlmp_top_sv2"]       = top_sv2
    df["tlmp_frob_ratio"]    = frob_ratio
    df["tlmp_lag2_energy"]   = lag2_energy

    # Lag decay: ratio of lag-1 to lag-2 coupling energy (higher = faster decay)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["tlmp_lag_decay"] = np.where(
            (lag2_energy > _EPS) & np.isfinite(lag2_energy),
            sig_energy / lag2_energy,
            np.nan,
        )

    # Rolling z-score of signal energy (regime signal)
    se_s = pd.Series(sig_energy)
    se_mu  = se_s.rolling(63, min_periods=20).mean()
    se_std = se_s.rolling(63, min_periods=20).std()
    df["tlmp_energy_zscore"] = ((se_s - se_mu) / (se_std + _EPS)).values

    return df
