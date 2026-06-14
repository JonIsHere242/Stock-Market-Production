"""
Vector Visibility Graph (VVG) features.

Source: "Vector Visibility Graph forecasting" (Physica A, 2025).
Phase-space reconstruct the price series, build a graph where edges connect
points with unobstructed line-of-sight, then extract topology from the
communicability-weighted maximum spanning tree.

Approximation used here: vector 'height' = Euclidean norm of the embedded
vector, reducing the multi-dimensional visibility rule to a tractable 1-D
NVG on norms. This captures the phase-space dynamics without the combinatorial
cost of true multi-dimensional visibility.
"""

import numpy as np
import networkx as nx
import pandas as pd
from scipy.linalg import expm

METADATA = {
    "name":        "vvg",
    "description": "Vector VG: communicability MST features from phase-space-embedded price",
    "requires":    ["Close"],
    "produces":    [
        "vvg_mean_degree", "vvg_max_degree",
        "vvg_max_betweenness", "vvg_mean_leaf_dist",
        "vvg_comm_spectral",
    ],
    "tags":        ["momentum", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "VVG paper Physica A 2025 — norm-height approximation",
}

_WINDOW = 22   # bars fed into embedding (must be > (M-1)*TAU)
_M      = 3    # embedding dimension
_TAU    = 1    # time delay


def _build_nvg_on_norms(norms: np.ndarray) -> nx.Graph:
    """Natural Visibility Graph on vector norms (vectorised inner loop)."""
    n = len(norms)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = norms[i], norms[j]
            if j == i + 1:
                G.add_edge(i, j)
                continue
            k      = np.arange(i + 1, j)
            t      = (k - i) / (j - i)
            interp = hi + t * (hj - hi)
            if np.all(norms[k] < interp):
                G.add_edge(i, j)
    return G


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    log_c = np.log(np.clip(close, 1e-10, None))
    n     = len(close)

    mean_deg  = np.full(n, np.nan)
    max_deg   = np.full(n, np.nan)
    max_btwn  = np.full(n, np.nan)
    leaf_dist = np.full(n, np.nan)
    comm_spec = np.full(n, np.nan)

    n_pts     = _WINDOW - (_M - 1) * _TAU   # embedded vectors per window
    min_start = _WINDOW + (_M - 1) * _TAU

    for i in range(min_start, n):
        window = log_c[i - _WINDOW : i]
        if not np.all(np.isfinite(window)):
            continue

        # phase-space reconstruction: each row is a delay-embedded vector
        vectors = np.column_stack([
            window[d * _TAU : d * _TAU + n_pts] for d in range(_M)
        ])
        norms = np.linalg.norm(vectors, axis=1)

        try:
            G = _build_nvg_on_norms(norms)

            # communicability matrix C = expm(A)
            A = nx.to_numpy_array(G)
            C = expm(A)

            # max spanning tree weighted by communicability
            k    = len(norms)
            full = nx.Graph()
            for a in range(k):
                for b in range(a + 1, k):
                    full.add_edge(a, b, weight=float(C[a, b]))
            mst = nx.maximum_spanning_tree(full)

            deg_vals = [d for _, d in mst.degree()]
            mean_deg[i] = float(np.mean(deg_vals))
            max_deg[i]  = float(np.max(deg_vals))

            btwn     = nx.betweenness_centrality(mst)
            max_btwn[i] = float(max(btwn.values()))

            leaves = [v for v, d in mst.degree() if d == 1]
            if leaves:
                dists = []
                for leaf in leaves:
                    dists.extend(
                        nx.single_source_shortest_path_length(mst, leaf).values()
                    )
                leaf_dist[i] = float(np.mean(dists))

            eigvals   = np.linalg.eigvalsh(C)
            comm_spec[i] = float(np.max(np.abs(eigvals)))

        except Exception:
            pass

    df["vvg_mean_degree"]     = mean_deg
    df["vvg_max_degree"]      = max_deg
    df["vvg_max_betweenness"] = max_btwn
    df["vvg_mean_leaf_dist"]  = leaf_dist
    df["vvg_comm_spectral"]   = comm_spec

    return df
