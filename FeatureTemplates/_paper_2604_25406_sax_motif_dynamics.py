"""
SAX Motif Dynamics  —  inspired by arXiv:2604.25406
"A Motif-Based Framework for Decomposing Risk Spillovers"

The paper shows that recurring temporal MOTIFS (short repeated shape patterns)
carry structured information about volatility regime and risk transmission.

This block implements a PER-TICKER, OHLCV-only symbolic-dynamics feature pack
using SAX (Symbolic Aggregate approXimation):

  1. Compute log returns from Close.
  2. Over a trailing window W, z-normalise the return series (window-local,
     fully causal — breakpoints derived only from the past W observations).
  3. Discretise each normalised value into one of A symbols (alphabet size A)
     using Gaussian quantile breakpoints computed from the SAME trailing window.
  4. Build short "words" (motifs) of length L from consecutive symbols.
  5. Compute information-theoretic and frequency features over the motif
     distribution in that trailing window.

Columns produced (prefix: sax_):
  sax_entropy_w40        — Shannon entropy of motif frequency distribution
                           (high = diverse, unpredictable; low = repetitive)
  sax_top_motif_freq_w40 — frequency of the most common motif (concentration)
  sax_distinct_motifs_w40 — count of distinct motifs seen in the window
  sax_novelty_w40        — 1 if the most-recent motif is rarer than the
                           25th-percentile frequency in the window, else 0
  sax_transition_entropy_w40 — Shannon entropy of the 1-step symbol transition
                           matrix (high = unpredictable next symbol)
  sax_symbol_z_w40       — z-score position of the current bar's log-return
                           within the trailing window (how extreme is today?)
  sax_repeat_run_w40     — length of the current run of the same most-recent
                           symbol (trend-persistence proxy)

CAUSALITY GUARANTEE:
  At every row t we look back at rows [t-W .. t-1] to build the SAX breakpoints
  AND the motif distribution.  Row t's own return is categorised using those
  breakpoints but contributes to motif counts only after the motif of length L
  ending at t is fully known (i.e. the last L symbols including t).  No future
  data ever enters the computation.

NOTE: Leading NaNs are emitted for the first W+L rows.  No inf values.
Distinct from paper_2606_12260_pattern_originality.py (which measures L2/cosine
distance between multi-dimensional OHLCV vectors; this block is strictly 1-D
symbolic dynamics on return series).
"""

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2604_25406_sax_motif_dynamics",
    "description": (
        "Per-ticker SAX symbolic-motif dynamics: discretises trailing-window "
        "log-returns into a small alphabet, builds length-3 motifs, and emits "
        "entropy, concentration, novelty, transition-predictability, and "
        "persistence features (arXiv:2604.25406 motif framework)."
    ),
    "requires": ["Close"],
    "produces": [
        "sax_entropy_w40",
        "sax_top_motif_freq_w40",
        "sax_distinct_motifs_w40",
        "sax_novelty_w40",
        "sax_transition_entropy_w40",
        "sax_symbol_z_w40",
        "sax_repeat_run_w40",
    ],
    "tags": ["experimental", "market_regime", "momentum"],
    "version": "1.0",
    "author": "paper:2604.25406 — SAX motif dynamics",
}

# ---------------------------------------------------------------------------
# Module-level constants (do NOT change — altering these changes all outputs)
# ---------------------------------------------------------------------------
_WINDOW   = 40   # trailing look-back window for z-norm and breakpoints
_ALPHA    = 4    # alphabet size (4 letters → Gaussian quartile breakpoints)
_MOTIF_L  = 3    # motif word length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_breakpoints(alpha: int) -> np.ndarray:
    """
    Breakpoints that divide the standard-normal distribution into `alpha`
    equal-probability regions.  These are FIXED (Gaussian assumption) but
    applied AFTER local z-normalisation, so no future information leaks.
    Returns (alpha - 1) threshold values.

    Uses scipy.stats.norm.ppf (percent-point function = inverse CDF) which is
    in the allowed import list.
    """
    from scipy.stats import norm as _norm
    probs = [i / alpha for i in range(1, alpha)]
    return np.array([_norm.ppf(p) for p in probs], dtype=np.float64)


_BPS = _gaussian_breakpoints(_ALPHA)   # shape (alpha-1,), computed once


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def _build_all_symbols(log_ret: np.ndarray, window: int, alpha: int) -> tuple:
    """
    Vectorized symbolisation pass.

    Returns (symbols, symbol_z) both length n.
    symbols[t] = SAX symbol for bar t, based solely on log_ret[t-W .. t-1].
    symbol_z[t] = z-score of log_ret[t] within the same window.
    symbols is int8 with -1 meaning undefined (insufficient history or NaN).
    """
    n = len(log_ret)
    symbols  = np.full(n, -1, dtype=np.int8)
    symbol_z = np.full(n, np.nan, dtype=np.float64)

    # Use pandas rolling with .shift(1) so window = [t-W .. t-1] (excludes t)
    s = pd.Series(log_ret)
    # min_periods = W/2 to tolerate a few NaN returns
    min_p = max(2, window // 2)
    roll_mean = s.shift(1).rolling(window, min_periods=min_p).mean().to_numpy()
    roll_std  = s.shift(1).rolling(window, min_periods=min_p).std().to_numpy()

    valid = (
        np.isfinite(log_ret) &
        np.isfinite(roll_mean) &
        np.isfinite(roll_std) &
        (roll_std > 1e-12)
    )

    z_vals = np.where(valid, (log_ret - roll_mean) / roll_std, np.nan)
    symbol_z[:] = np.clip(z_vals, -10.0, 10.0)

    # Digitise valid z-values using Gaussian breakpoints
    valid_idx = np.where(valid)[0]
    if len(valid_idx):
        symbols[valid_idx] = np.searchsorted(_BPS, z_vals[valid_idx]).astype(np.int8)

    # Flat-return case: std too small but mean finite — assign centre symbol
    flat_valid = (
        np.isfinite(log_ret) &
        np.isfinite(roll_mean) &
        np.isfinite(roll_std) &
        (roll_std <= 1e-12) &
        np.isfinite(roll_std)
    )
    symbols[flat_valid] = np.int8(alpha // 2)
    symbol_z[flat_valid] = 0.0

    return symbols, symbol_z


def _motif_features_vectorized(
    symbols: np.ndarray,
    window: int,
    motif_l: int,
    alpha: int,
) -> tuple:
    """
    Compute motif distribution features over a sliding window of symbols.

    For each valid row t (symbols[t-W+1 .. t] all >= 0), encodes each motif
    of length motif_l as a base-alpha integer so we can use bincount for O(W)
    per row rather than Counter hashing.

    Returns 5 arrays: entropy, top_freq, distinct, novelty, trans_ent, repeat_run.
    """
    n = len(symbols)
    out_entropy   = np.full(n, np.nan)
    out_top_freq  = np.full(n, np.nan)
    out_distinct  = np.full(n, np.nan)
    out_novelty   = np.full(n, np.nan)
    out_trans_ent = np.full(n, np.nan)
    out_repeat_run = np.full(n, np.nan)

    # Precompute motif encodings for the ENTIRE symbol array using stride tricks.
    # motif_codes[i] = base-alpha encoding of symbols[i .. i+motif_l-1]
    # We'll build a (n - motif_l + 1, motif_l) view, then dot with powers.
    # However we need these aligned to the window so we compute them lazily below.

    # Powers for base-alpha encoding
    powers = np.array([alpha ** i for i in range(motif_l - 1, -1, -1)], dtype=np.int32)
    n_motifs = window - motif_l + 1   # motifs per window

    # Transition matrix shape: alpha x alpha
    A = alpha

    min_t = window + motif_l - 2   # 0-indexed first valid row

    for t in range(min_t, n):
        sym_win = symbols[t - window + 1: t + 1]   # length window
        if np.any(sym_win < 0):
            continue

        # ---- Motif distribution via integer encoding + bincount ---------------
        # Build motif integer codes for this window (length n_motifs)
        # Using a (n_motifs, motif_l) slice via stride tricks is not safe in a loop,
        # but we can do it with indexing since window is small (40).
        sym_int = sym_win.astype(np.int32)
        # Stacked motif matrix (n_motifs x motif_l)
        motif_rows = np.lib.stride_tricks.sliding_window_view(sym_int, motif_l)  # (n_motifs, L)
        codes = motif_rows @ powers   # (n_motifs,)

        max_code = alpha ** motif_l
        counts = np.bincount(codes, minlength=max_code)   # shape (alpha^L,)
        nonzero_counts = counts[counts > 0]
        total = n_motifs

        # Entropy
        probs = nonzero_counts / total
        h = float(-np.sum(probs * np.log(probs)))
        out_entropy[t] = h

        # Top frequency
        out_top_freq[t] = float(nonzero_counts.max()) / total

        # Distinct motifs
        out_distinct[t] = float(len(nonzero_counts))

        # Novelty: recent motif code vs 25th-percentile frequency
        recent_code = int(codes[-1])
        recent_freq = counts[recent_code] / total
        all_freqs = nonzero_counts / total
        pct25 = float(np.percentile(all_freqs, 25))
        out_novelty[t] = 1.0 if recent_freq <= pct25 else 0.0

        # ---- Transition entropy (vectorized over the window) -----------------
        # Transition matrix T[s, s'] = count of s->s' pairs
        from_sym = sym_int[:-1]
        to_sym   = sym_int[1:]
        trans_idx = from_sym * A + to_sym
        T_flat = np.bincount(trans_idx, minlength=A * A).reshape(A, A)
        row_sums = T_flat.sum(axis=1, keepdims=True)
        valid_rows = (row_sums[:, 0] > 0)
        T_prob = np.where(row_sums > 0, T_flat / row_sums, 0.0)
        # Per-row entropy weighted by source frequency
        with np.errstate(divide="ignore", invalid="ignore"):
            log_T = np.where(T_prob > 0, np.log(T_prob), 0.0)
        row_h = -np.sum(T_prob * log_T, axis=1)
        row_weight = row_sums[:, 0] / (window - 1)
        out_trans_ent[t] = float(np.sum(row_weight[valid_rows] * row_h[valid_rows]))

        # ---- Repeat run: backward run of last symbol -------------------------
        cur = int(sym_int[-1])
        # Find first position from the right that differs
        diff_pos = np.where(sym_int[:-1] != cur)[0]
        if len(diff_pos) == 0:
            run_len = window
        else:
            run_len = window - 1 - int(diff_pos[-1])
        out_repeat_run[t] = float(run_len)

    return out_entropy, out_top_freq, out_distinct, out_novelty, out_trans_ent, out_repeat_run


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SAX motif dynamics features for a single stock.

    For each row t, uses the PAST _WINDOW log-returns [t-W .. t-1] for
    z-normalisation breakpoints (fully causal).  Leading NaNs for the first
    _WINDOW + _MOTIF_L - 1 rows.
    """
    n = len(df)

    # Log returns (row i uses Close[i-1] and Close[i], so ret[0] = NaN)
    close = df["Close"].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_ret = np.concatenate(([np.nan], np.log(close[1:] / close[:-1])))
    # Guard against inf from zero prices
    log_ret = np.where(np.isfinite(log_ret), log_ret, np.nan)

    # Step 1: vectorized symbolisation
    symbols, symbol_z = _build_all_symbols(log_ret, _WINDOW, _ALPHA)

    # Step 2: motif distribution features
    (out_entropy, out_top_freq, out_distinct,
     out_novelty, out_trans_ent, out_repeat_run) = _motif_features_vectorized(
        symbols, _WINDOW, _MOTIF_L, _ALPHA
    )

    # Assign to df — ONLY ADD, never modify existing columns
    df["sax_entropy_w40"]            = pd.array(out_entropy,    dtype="Float64")
    df["sax_top_motif_freq_w40"]     = pd.array(out_top_freq,   dtype="Float64")
    df["sax_distinct_motifs_w40"]    = pd.array(out_distinct,   dtype="Float64")
    df["sax_novelty_w40"]            = pd.array(out_novelty,    dtype="Float64")
    df["sax_transition_entropy_w40"] = pd.array(out_trans_ent,  dtype="Float64")
    df["sax_symbol_z_w40"]           = pd.array(symbol_z,       dtype="Float64")
    df["sax_repeat_run_w40"]         = pd.array(out_repeat_run, dtype="Float64")

    return df
