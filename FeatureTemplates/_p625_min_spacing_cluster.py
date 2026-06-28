import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name":        "_p625_min_spacing_cluster",
    "description": "Left-tail (MIN) clustering geometry: temporal spacing & burst-concentration "
                   "of a ticker's worst days within the month "
                   "(Bali-Cakici-Whitelaw 2011 JFE MAX/MIN; Bollerslev-Todorov jump clustering).",
    "requires":    [],
    "produces":    ["mspc_gap_21", "mspc_burst_21", "mspc_min_recency_21",
                    "mspc_dnrun_21", "mspc_burst_42"],
    "tags":        ["tail", "behavioral", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "paper:Bali,Cakici&Whitelaw(2011)JFE; Kahneman&Tversky prospect theory; Bollerslev-Todorov jump-clustering",
}

_K = 5  # number of worst (most-negative) days considered for spacing geometry


def _windows(arr: np.ndarray, w: int) -> np.ndarray:
    """Return sliding windows of length w over a 1-D array (shape (n-w+1, w))."""
    if len(arr) < w:
        return np.empty((0, w), dtype=float)
    return sliding_window_view(arr, w)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    n = len(close)

    # simple returns; first element is NaN (no prior bar) -> treat as 0 contribution
    r = close.pct_change().to_numpy(dtype="float64")
    r0 = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r0sq = r0 * r0

    out_gap_21      = np.full(n, np.nan)
    out_burst_21    = np.full(n, np.nan)
    out_recency_21  = np.full(n, np.nan)
    out_dnrun_21    = np.full(n, np.nan)
    out_burst_42    = np.full(n, np.nan)

    # ----- gap / recency / down-run: window W = 21 -----------------------------
    W = 21
    if n >= W:
        win = _windows(r0, W)            # (m, W), oldest..newest; row i ends at df index i+W-1
        m = win.shape[0]
        # indices into df of each window's last row
        end_idx = np.arange(W - 1, n)

        # ---- single worst day position (recency) ----
        worst_pos = np.argmin(win, axis=1)                 # 0..W-1, larger = more recent
        out_recency_21[end_idx] = (W - 1 - worst_pos) / (W - 1)

        # ---- mean consecutive spacing among K worst days ----
        # k smallest (most negative) positions per window via argpartition
        k = min(_K, W)
        part = np.argpartition(win, k - 1, axis=1)[:, :k]   # positions of k smallest, unordered
        part = np.sort(part, axis=1)                        # ascending by position within window
        if k >= 2:
            diffs = np.diff(part, axis=1)                   # consecutive position-diffs
            gap_mean = diffs.mean(axis=1).astype("float64")
        else:
            gap_mean = np.zeros(m, dtype="float64")
        out_gap_21[end_idx] = gap_mean / W

        # ---- longest run of consecutive negative days ----
        neg = (win < 0.0)                                   # (m, W) boolean
        # vectorized longest-True-run per row: reset cumulative count at False
        # running length: where neg, prev+1 else 0; longest = max over row
        run_len = np.zeros_like(neg, dtype="int64")
        run_len[:, 0] = neg[:, 0].astype("int64")
        for j in range(1, W):
            run_len[:, j] = np.where(neg[:, j], run_len[:, j - 1] + 1, 0)
        longest = run_len.max(axis=1).astype("float64")
        out_dnrun_21[end_idx] = longest / W

    # ----- burst concentration: helper over a given window --------------------
    def _burst(Wb: int) -> np.ndarray:
        res = np.full(n, np.nan)
        if n < Wb or Wb < 3:
            return res
        winr  = _windows(r0,   Wb)       # (m, Wb) raw returns
        winsq = _windows(r0sq, Wb)       # (m, Wb) squared returns
        end_idx = np.arange(Wb - 1, n)

        denom = winsq.sum(axis=1)                            # sum r^2 over the window
        # all consecutive 3-day blocks within each window
        tri_sq  = sliding_window_view(winsq, 3, axis=1)      # (m, Wb-2, 3)
        tri_sum = tri_sq.sum(axis=2)                          # (m, Wb-2) numerator candidates
        tri_r   = sliding_window_view(winr, 3, axis=1)        # (m, Wb-2, 3)
        net3    = tri_r.sum(axis=2)                            # net 3-day return per block

        # restrict to net-negative 3-day runs; others ineligible
        eligible = net3 < 0.0
        masked = np.where(eligible, tri_sum, -np.inf)
        best = masked.max(axis=1)                             # max eligible 3-day energy
        no_elig = ~np.isfinite(best)                          # no net-negative 3-day run

        denom_safe = np.where(denom > 0.0, denom, np.nan)     # guard div by zero
        ratio = best / denom_safe
        ratio = np.where(no_elig, 0.0, ratio)                 # no negative run -> 0 concentration
        res[end_idx] = ratio
        return res

    out_burst_21 = _burst(21)
    out_burst_42 = _burst(42)

    df["mspc_gap_21"]         = out_gap_21
    df["mspc_burst_21"]       = out_burst_21
    df["mspc_min_recency_21"] = out_recency_21
    df["mspc_dnrun_21"]       = out_dnrun_21
    df["mspc_burst_42"]       = out_burst_42
    return df
