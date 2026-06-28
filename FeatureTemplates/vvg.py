import numpy as np
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
    "version":     "2.0",
    "author":      "VVG paper Physica A 2025 - norm-height approximation",
}

_WINDOW = 22
_M      = 3
_TAU    = 1


def _nvg_adj(norms, k):
    hl = norms.tolist()
    ei = []
    ej = []
    for i in range(k - 1):
        hi = hl[i]
        mx = float("-inf")
        for j in range(i + 1, k):
            s = (hl[j] - hi) / (j - i)
            if s > mx:
                ei.append(i); ej.append(j); mx = s
    A = np.zeros((k, k))
    A[ei, ej] = 1.0
    A[ej, ei] = 1.0
    return A


def _mst_adj(C, k, aa0, bb0):
    w = C[aa0, bb0]
    order = np.argsort(-w, kind="stable")
    aa = aa0[order].tolist()
    bb = bb0[order].tolist()
    parent = list(range(k))
    adj = [[] for _ in range(k)]
    cnt = 0
    for a, b in zip(aa, bb):
        ra = a
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]; ra = parent[ra]
        rb = b
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]; rb = parent[rb]
        if ra != rb:
            parent[ra] = rb
            adj[a].append(b); adj[b].append(a)
            cnt += 1
            if cnt == k - 1:
                break
    return adj


def _tree_features(adj, k):
    parent = [-1] * k
    order  = []
    seen   = bytearray(k)
    seen[0] = 1
    st = [0]
    while st:
        u = st.pop()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = 1
                parent[v] = u
                st.append(v)

    size = [1] * k
    for u in reversed(order):
        p = parent[u]
        if p >= 0:
            size[p] += size[u]

    csq  = [0] * k
    down = [0] * k
    for u in reversed(order):
        p = parent[u]
        if p >= 0:
            su = size[u]
            csq[p]  += su * su
            down[p] += down[u] + su

    K1 = (k - 1) * (k - 1)
    scale = 2.0 / ((k - 1) * (k - 2)) if k > 2 else 1.0
    max_pairs = 0
    for v in range(k):
        up = k - size[v]
        pairs = (K1 - (csq[v] + up * up))
        if pairs > max_pairs:
            max_pairs = pairs
    max_bt = (max_pairs / 2.0) * scale

    sumdist = [0] * k
    sumdist[0] = down[0]
    for u in order:
        if u != 0:
            p = parent[u]
            sumdist[u] = sumdist[p] + k - 2 * size[u]

    leaf_tot = 0
    n_leaf   = 0
    for v in range(k):
        if len(adj[v]) == 1:
            leaf_tot += sumdist[v]
            n_leaf += 1
    leaf_mean = leaf_tot / (n_leaf * k) if n_leaf else np.nan
    return max_bt, leaf_mean


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].to_numpy(dtype=float)
    log_c = np.log(np.clip(close, 1e-10, None))
    n     = len(close)

    mean_deg  = np.full(n, np.nan)
    max_deg   = np.full(n, np.nan)
    max_btwn  = np.full(n, np.nan)
    leaf_dist = np.full(n, np.nan)
    comm_spec = np.full(n, np.nan)

    n_pts     = _WINDOW - (_M - 1) * _TAU
    min_start = _WINDOW + (_M - 1) * _TAU
    k         = n_pts
    aa0, bb0  = np.triu_indices(k, 1)
    mean_deg_const = 2.0 * (k - 1) / k

    C_stack = np.empty((n, k, k))
    valid_i = []

    for i in range(min_start, n):
        window = log_c[i - _WINDOW : i]
        if not np.all(np.isfinite(window)):
            continue
        try:
            vectors = np.column_stack([
                window[d * _TAU : d * _TAU + n_pts] for d in range(_M)
            ])
            norms = np.linalg.norm(vectors, axis=1)

            A = _nvg_adj(norms, k)
            C = expm(A)

            adj = _mst_adj(C, k, aa0, bb0)

            mean_deg[i] = mean_deg_const
            max_deg[i]  = float(max(len(a) for a in adj))

            mb, lm = _tree_features(adj, k)
            max_btwn[i]  = mb
            leaf_dist[i] = lm

            C_stack[i] = C
            valid_i.append(i)
        except Exception:
            pass

    if valid_i:
        vi = np.array(valid_i)
        eig = np.linalg.eigvalsh(C_stack[vi])
        comm_spec[vi] = np.max(np.abs(eig), axis=1)

    df["vvg_mean_degree"]     = mean_deg
    df["vvg_max_degree"]      = max_deg
    df["vvg_max_betweenness"] = max_btwn
    df["vvg_mean_leaf_dist"]  = leaf_dist
    df["vvg_comm_spectral"]   = comm_spec
    return df