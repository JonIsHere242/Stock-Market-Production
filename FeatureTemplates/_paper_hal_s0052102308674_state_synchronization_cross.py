"""
Single-Series Recurrence Quantification Features (Proxy)  —  doi:10.1007/s00521-023-08674-y
"Predicting the state of synchronization of financial time series using cross recurrence plots"

PROXY RATIONALE:
The paper uses CROSS recurrence plots to measure the synchronization state between PAIRS
of stocks, then predicts that state with a DNN. This fundamentally requires TWO time series.
We cannot replicate:
  - Cross-recurrence analysis (requires a second, exogenous series)
  - Pair-wise synchronization prediction
  - DNN framework for state prediction

PROXY IMPLEMENTED (single-series RQA):
We implement auto-recurrence quantification analysis (RQA) on a single series, which
captures the structural self-similarity the paper measures between pairs. The paper's key
RQA features extracted from CRPs (sub-sampled recurrence matrices) are:
  1. Recurrence Rate (RR): fraction of phase-space points that recur (density of the RP)
  2. Determinism (DET): fraction of recurrence points forming diagonal structures (≥ min_len)
     — high DET = deterministic / trending behavior
  3. Laminarity (LAM): fraction forming vertical structures — captures laminar (low-volatility)
     states vs chaotic jumps
  4. Trapping time (TT): average vertical line length — measures mean persistence duration
  5. Entropy of diagonal lines (ENT_DIAG): Shannon entropy of diagonal line lengths

We use a delay-embedding of the log-return series (embedding dim=2, tau=1) to build a
compact recurrence matrix efficiently. The paper's "sub-sampled" CRP approach is mirrored
by computing RQA on a rolling window of the embedded series.

COMPUTATIONAL NOTE:
Full O(n^2) recurrence matrix is too slow for 700+ rows over many tickers. We use
a rolling window (~30 bars) with the embedded 2D phase space and a fixed threshold.
"""

import math
import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_hal_s0052102308674_state_synchronization_cross",
    "description": (
        "Single-series recurrence quantification analysis (RQA) proxy for "
        "doi:10.1007/s00521-023-08674-y cross-recurrence synchronization predictor; "
        "implements recurrence rate, determinism, laminarity, trapping time, and diagonal "
        "entropy on a rolling delay-embedded log-return series (cross-series pair dropped "
        "— OHLCV provides only one series per ticker)."
    ),
    "requires": ["Close"],
    "produces": [
        "rqa_rr_20",
        "rqa_det_20",
        "rqa_lam_20",
        "rqa_tt_20",
        "rqa_ent_diag_20",
        "rqa_rr_40",
        "rqa_det_40",
        "rqa_lam_40",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": (
        "proxy: doi:10.1007/s00521-023-08674-y (cross-recurrence synchronization). "
        "Cross-series pair and DNN prediction dropped (single OHLCV series available). "
        "Implemented as single-series auto-RQA on rolling delay-embedded log-returns."
    ),
}


def _runs_of_ones(seq: np.ndarray) -> np.ndarray:
    """Lengths of every maximal run of consecutive 1s in a 1-D 0/1 array (vectorised RLE)."""
    if seq.size == 0:
        return np.empty(0, dtype=np.int64)
    padded = np.empty(seq.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = seq
    d = np.diff(padded.astype(np.int16))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return (ends - starts).astype(np.int64)


def _runs_down_columns(mat: np.ndarray, min_line: int) -> np.ndarray:
    """Run lengths (>= min_line) of 1s taken DOWN each column of `mat`, columns separated
    by a zero so runs never bridge across columns — a vectorised replacement for the
    per-column / per-diagonal Python run-length loops."""
    if mat.size == 0:
        return np.empty(0, dtype=np.int64)
    T = mat.T                                   # each row = one column of mat
    sep = np.zeros((T.shape[0], 1), dtype=mat.dtype)
    flat = np.concatenate([T, sep], axis=1).ravel()
    runs = _runs_of_ones(flat)
    return runs[runs >= min_line]


def _diag_skew(R: np.ndarray) -> np.ndarray:
    """Lay every diagonal of (m,m) `R` into its own column (zero-padded) so that
    _runs_down_columns recovers the diagonal run-lengths without a per-diagonal loop.
    Column c holds diagonal k = c-(m-1); its elements stay in ascending-row order."""
    m = R.shape[0]
    skew = np.zeros((m, 2 * m - 1), dtype=R.dtype)
    ii = np.arange(m)[:, None]
    jj = np.arange(m)[None, :]
    col = (jj - ii) + (m - 1)                   # constant per diagonal
    skew[np.broadcast_to(ii, (m, m)), col] = R
    return skew


def _rqa_from_R(R: np.ndarray, min_line: int = 2) -> dict:
    """RQA metrics (rr, det, lam, tt, ent_diag) from a prebuilt recurrence matrix R
    (diagonal already zeroed). Vectorised equivalent of the old _rqa_window run-length
    loops; det_points/lam_points/run-length histograms are identical, so the metrics
    match the previous per-element implementation bit-for-bit."""
    m = R.shape[0]
    rec_total = int(R.sum())
    total_pairs = m * (m - 1)
    rr = float(rec_total) / total_pairs if total_pairs > 0 else np.nan

    # --- Determinism + diagonal-length entropy (diagonal runs >= min_line) ---
    diag_runs = _runs_down_columns(_diag_skew(R), min_line)
    det_points = int(diag_runs.sum())
    det = float(det_points) / rec_total if rec_total > 0 else 0.0
    if diag_runs.size:
        lens = diag_runs.astype(float)
        total_l = lens.sum()
        probs = np.array([np.sum(lens == v) * v / total_l for v in np.unique(lens)])
        probs = probs[probs > 0]
        ent_diag = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        ent_diag = 0.0

    # --- Laminarity + trapping time (vertical runs >= min_line) ---
    vert_runs = _runs_down_columns(R, min_line)
    lam_points = int(vert_runs.sum())
    lam = float(lam_points) / rec_total if rec_total > 0 else 0.0
    tt = float(np.mean(vert_runs)) if vert_runs.size else 0.0

    return {"rr": rr, "det": det, "lam": lam, "tt": tt, "ent_diag": ent_diag}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling single-series RQA features.

    Each bar's RQA metrics are computed from the log-return series in a trailing
    window. The recurrence threshold eps is set adaptively as a quantile of the
    pairwise distance distribution within the window.
    """
    c = df["Close"].values.astype(np.float64)
    n = len(c)

    lret = np.full(n, np.nan)
    lret[1:] = np.log(np.where(c[1:] > 0, c[1:], np.nan) /
                      np.where(c[:-1] > 0, c[:-1], np.nan))

    # Initialize output arrays
    rr20   = np.full(n, np.nan)
    det20  = np.full(n, np.nan)
    lam20  = np.full(n, np.nan)
    tt20   = np.full(n, np.nan)
    ent20  = np.full(n, np.nan)
    rr40   = np.full(n, np.nan)
    det40  = np.full(n, np.nan)
    lam40  = np.full(n, np.nan)

    # Adaptive epsilon: 20th percentile of pairwise distances within window
    # (standard RQA convention for fixed recurrence rate threshold calibration)
    eps_quantile = 0.20

    for window, arrays in [(20, (rr20, det20, lam20, tt20, ent20)),
                            (40, (rr40, det40, lam40, None, None))]:
        min_obs = window // 2
        for i in range(min_obs + 2, n):
            start = max(0, i - window + 1)
            seg = lret[start: i + 1]
            fin = seg[np.isfinite(seg)]
            if len(fin) < 5:
                continue
            pts = np.column_stack([fin[1:], fin[:-1]])
            if len(pts) < 3:
                continue
            # Pairwise distance matrix — computed ONCE and reused for both the eps
            # estimate and the recurrence matrix (the old code built it twice).
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            upper = dist[np.triu_indices_from(dist, k=1)]
            if len(upper) == 0:
                continue
            eps = float(np.quantile(upper, eps_quantile))
            if eps < 1e-10:
                eps = float(np.quantile(upper, 0.30))
            if eps < 1e-10:
                continue

            R = (dist < eps).astype(np.int8)
            np.fill_diagonal(R, 0)              # exclude self-recurrence
            metrics = _rqa_from_R(R)
            if window == 20:
                rr20[i]  = metrics["rr"]
                det20[i] = metrics["det"]
                lam20[i] = metrics["lam"]
                tt20[i]  = metrics["tt"]
                ent20[i] = metrics["ent_diag"]
            else:
                rr40[i]  = metrics["rr"]
                det40[i] = metrics["det"]
                lam40[i] = metrics["lam"]

    df["rqa_rr_20"]       = rr20
    df["rqa_det_20"]      = det20
    df["rqa_lam_20"]      = lam20
    df["rqa_tt_20"]       = tt20
    df["rqa_ent_diag_20"] = ent20
    df["rqa_rr_40"]       = rr40
    df["rqa_det_40"]      = det40
    df["rqa_lam_40"]      = lam40

    return df
