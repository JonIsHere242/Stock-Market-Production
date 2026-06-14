"""
Rising Visibility Graph + Weisfeiler-Lehman Kernel (RVGWL) features.
Maximum performance, output-identical to original.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

METADATA = {
    "name":        "rvg_wl",
    "description": "Rising VG + WL-3 kernel cosine similarity against 9 trend templates",
    "requires":    ["Close"],
    "produces":    [
        "rvg_wl_mono_up", "rvg_wl_mono_down",
        "rvg_wl_peak",    "rvg_wl_trough",
        "rvg_wl_asc_plateau", "rvg_wl_desc_plateau",
        "rvg_wl_v_shape", "rvg_wl_inv_v",
        "rvg_wl_flat",
    ],
    "tags":        ["momentum", "trend", "experimental"],
    "version":     "3.0",
    "author":      "RVGWL paper (ultra‑optimized)",
}

_WINDOW = 20
_WL_H   = 3

# ---------------------------------------------------------------------------
# Template shapes (same as original)
# ---------------------------------------------------------------------------
def _make_templates(w: int):
    half = w // 2
    v_arr = np.ones(w)
    v_arr[half - 1] = 0.30
    v_arr[half] = 0.00
    v_arr[half + 1] = 0.30
    iv_arr = np.zeros(w)
    iv_arr[half - 1] = 0.70
    iv_arr[half] = 1.00
    iv_arr[half + 1] = 0.70
    return {
        "mono_up":      np.linspace(0.0, 1.0, w),
        "mono_down":    np.linspace(1.0, 0.0, w),
        "peak":         np.sin(np.linspace(0, np.pi, w)),
        "trough":       1.0 - np.sin(np.linspace(0, np.pi, w)),
        "asc_plateau":  np.concatenate([np.linspace(0, 1, half), np.ones(w - half)]),
        "desc_plateau": np.concatenate([np.ones(half), np.linspace(1, 0, w - half)]),
        "v_shape":      v_arr,
        "inv_v":        iv_arr,
        "flat":         np.full(w, 0.5),
    }

# ---------------------------------------------------------------------------
# Rising Visibility Graph – adjacency list (list of lists, no sets)
# ---------------------------------------------------------------------------
def _build_rvg_adj(prices: np.ndarray):
    n = len(prices)
    adj = [[] for _ in range(n)]
    for i in range(n - 1):
        hi = prices[i]
        # scan forward; break at first value > hi
        for j in range(i + 1, n):
            pj = prices[j]
            if pj > hi:
                break
            if pj >= hi:  # pj == hi, because pj < hi is skipped
                adj[i].append(j)
                adj[j].append(i)
    return adj

# ---------------------------------------------------------------------------
# WL-h colour histogram – using tuples for speed, but string IDs for identical output
# ---------------------------------------------------------------------------
def _wl_histogram(adj, n_nodes: int, h: int):
    # initial colour = degree (int)
    labels = [len(adj[i]) for i in range(n_nodes)]
    # histogram: (round, colour_id_str) -> count
    hist = defaultdict(int)
    for lbl in labels:
        hist[(0, str(lbl))] += 1

    for rnd in range(1, h + 1):
        # Build tuple signature: (current_label, sorted(neighbour_labels))
        signatures = []
        for v in range(n_nodes):
            nbr_lbls = sorted(labels[u] for u in adj[v])
            signatures.append((labels[v], tuple(nbr_lbls)))
        # Map unique signatures to string IDs (0,1,2,...) in order of first appearance
        mapping = {}
        next_id = 0
        new_labels = []
        for sig in signatures:
            if sig not in mapping:
                mapping[sig] = str(next_id)
                next_id += 1
            new_labels.append(mapping[sig])
        labels = new_labels
        for lbl in labels:
            hist[(rnd, lbl)] += 1
    return hist

# ---------------------------------------------------------------------------
# Cosine similarity (exactly as original)
# ---------------------------------------------------------------------------
def _cosine(h1: dict, h2: dict) -> float:
    keys = set(h1) | set(h2)
    v1 = np.array([h1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([h2.get(k, 0) for k in keys], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ---------------------------------------------------------------------------
# Pre‑compute template adjacency lists and histograms at import
# ---------------------------------------------------------------------------
_SHAPES = _make_templates(_WINDOW)
_TMPL_DATA = {}
for name, shape in _SHAPES.items():
    adj = _build_rvg_adj(shape)
    hist = _wl_histogram(adj, _WINDOW, _WL_H)
    _TMPL_DATA[name] = hist   # only store histogram, adj not needed later
_TMPL_ORDER = list(_SHAPES.keys())

# ---------------------------------------------------------------------------
# compute() – main entry point
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    n = len(close)

    # Pre-allocate output columns
    out = {f"rvg_wl_{name}": np.full(n, np.nan) for name in _TMPL_ORDER}

    # Local aliases for speed
    _build = _build_rvg_adj
    _hist = _wl_histogram
    _cos = _cosine
    tmpl_hist = _TMPL_DATA
    order = _TMPL_ORDER
    w = _WINDOW
    h = _WL_H

    for i in range(w, n):
        window = close[i - w : i]
        if not np.isfinite(window).all():
            continue

        lo, hi = window.min(), window.max()
        if hi == lo:
            normed = np.full(w, 0.5)
        else:
            normed = (window - lo) / (hi - lo)

        adj = _build(normed)
        hist = _hist(adj, w, h)

        for name in order:
            out[f"rvg_wl_{name}"][i] = _cos(hist, tmpl_hist[name])

    for col, arr in out.items():
        df[col] = arr

    return df