import numpy as np
import pandas as pd
from ts2vg import NaturalVG

METADATA = {
    "name":        "vaq_vg",
    "description": "Quantile-ranked NVG spectral features at three edge-threshold levels",
    "requires":    ["Close"],
    "produces":    [
        "vaq_ret_qrank",
        "vaq_q25_spectral",  "vaq_q25_nat_conn",  "vaq_q25_alg_conn",
        "vaq_q25_clustering","vaq_q25_assort",
        "vaq_q50_spectral",  "vaq_q50_nat_conn",  "vaq_q50_alg_conn",
        "vaq_q50_clustering","vaq_q50_assort",
        "vaq_q75_spectral",  "vaq_q75_nat_conn",  "vaq_q75_alg_conn",
        "vaq_q75_clustering","vaq_q75_assort",
    ],
    "tags":        ["volatility", "market_regime", "experimental"],
    "version":     "3.0",
    "author":      "VAQ-VG paper AIA 2024 / Butman 2024",
}

_NORM_WINDOW = 250
_VG_WINDOW   = 50
_THRESHOLDS  = [0.25, 0.50, 0.75]
_MIN_EDGES   = 3
_FEAT_NAMES  = ("spectral", "nat_conn", "alg_conn", "clustering", "assort")


def _spectral_features(A, n_total):
    nan5 = (np.nan,) * 5
    k = A.shape[0]
    n_edges = int(A.sum() // 2)
    if n_edges < _MIN_EDGES:
        return nan5

    eig = np.linalg.eigvalsh(A)
    spectral = float(np.max(np.abs(eig)))

    exp_sum = float(np.sum(np.exp(np.clip(eig, -50, 50)))) + (n_total - k)
    nat_conn = float(np.log(exp_sum / n_total))

    deg = A.sum(axis=1)
    if k < n_total:
        alg_conn = 0.0
    else:
        L = np.diag(deg) - A
        eigL = np.sort(np.linalg.eigvalsh(L))
        alg_conn = float(eigL[1]) if len(eigL) > 1 else 0.0

    A2  = A @ A
    tri = np.einsum("ij,ij->i", A2, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ci = np.where(deg >= 2, tri / (deg * (deg - 1)), 0.0)
    clustering = float(np.sum(Ci) / n_total)

    m  = n_edges
    S1 = 0.5 * float(deg @ (A @ deg))
    s2 = float(np.sum(deg ** 2))
    s3 = float(np.sum(deg ** 3))
    mu  = s2 / (2 * m)
    cov = S1 / m - mu * mu
    var = s3 / (2 * m) - mu * mu
    assort = float(cov / var) if var > 0 else np.nan

    return spectral, nat_conn, alg_conn, clustering, assort


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(close[1:] / close[:-1])

    qrank   = np.full(n, np.nan)
    results = {
        f"vaq_q{int(q*100)}_{feat}": np.full(n, np.nan)
        for q in _THRESHOLDS
        for feat in _FEAT_NAMES
    }

    th_arrays = [
        (q, tuple(results[f"vaq_q{int(q*100)}_{f}"] for f in _FEAT_NAMES))
        for q in _THRESHOLDS
    ]

    start = max(_NORM_WINDOW, _VG_WINDOW)
    for i in range(start, n):
        norm_slice = log_ret[i - _NORM_WINDOW : i]
        valid      = norm_slice[np.isfinite(norm_slice)]
        if len(valid) < _VG_WINDOW:
            continue

        ret_i = log_ret[i]
        norm_sorted = np.sort(valid)
        if np.isfinite(ret_i):
            qrank[i] = float(np.searchsorted(norm_sorted, ret_i) / len(valid))

        vg_slice = log_ret[i - _VG_WINDOW : i]
        if not np.all(np.isfinite(vg_slice)):
            continue

        ranks = np.searchsorted(norm_sorted, vg_slice) / len(norm_sorted)

        try:
            vg = NaturalVG()
            vg.build(ranks)
            A_full = np.asarray(vg.adjacency_matrix(), dtype=float)
        except Exception:
            continue

        for q, arrs in th_arrays:
            idx = np.flatnonzero(ranks > q)
            if len(idx) < 2:
                continue
            A_sub = A_full[np.ix_(idx, idx)]
            feats = _spectral_features(A_sub, _VG_WINDOW)
            for arr, fval in zip(arrs, feats):
                arr[i] = fval

    df["vaq_ret_qrank"] = qrank
    for col, arr in results.items():
        df[col] = arr
    return df