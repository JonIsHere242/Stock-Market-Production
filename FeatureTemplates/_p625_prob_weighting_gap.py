import numpy as np
import pandas as pd

METADATA = {
    "name":        "_p625_prob_weighting_gap",
    "description": ("CPT inverse-S probability-weighting distortion gap on raw returns "
                    "(perceived minus true expectation); Tversky&Kahneman 1992 JRU; "
                    "Barberis 2013 JEP; Boyer&Vorkink 2014 RFS."),
    "requires":    [],
    "produces":    ["pwg_gap_60", "pwg_uptail_60", "pwg_gap_120"],
    "tags":        ["behavioral", "prospect_theory", "tail", "experimental"],
    "version":     "1.0",
    "author":      "paper:Tversky&Kahneman 1992 JRU; Barberis 2013 JEP; Boyer&Vorkink 2014 RFS",
}

# inverse-S Tversky-Kahneman (1992) probability-weighting parameters
_GAMMA_GAIN = 0.61
_GAMMA_LOSS = 0.69


def _w(p: np.ndarray, g: float) -> np.ndarray:
    """Inverse-S weighting function w(p) = p^g / (p^g + (1-p)^g)^(1/g).

    p is an array of cumulative probabilities in [0, 1]. Guard the 0/0 endpoints
    (p=0 -> 0, p=1 -> 1) so no inf/nan leaks from the power/division.
    """
    p = np.clip(p, 0.0, 1.0)
    pg = np.power(p, g)
    qg = np.power(1.0 - p, g)
    denom = np.power(pg + qg, 1.0 / g)
    out = np.where(denom > 0.0, pg / denom, 0.0)
    # enforce exact endpoints (avoids tiny float dust at p=0/p=1)
    out = np.where(p <= 0.0, 0.0, out)
    out = np.where(p >= 1.0, 1.0, out)
    return out


def _cpt_window(r: np.ndarray) -> tuple[float, float]:
    """Given one window of raw returns r (length W), return (gap, uptail).

    gap    = PW - EW  (probability-weighted mean minus equal-weighted mean)
    uptail = sum over gain-side of (pi_i - 1/W) * r_(i)
    Sign-dependent rank-dependent weights (CPT): gains cumulated from the best
    outcome downward with gamma+, losses cumulated from the worst outcome upward
    with gamma-. Decision weights pi_i are first differences of the transformed
    cumulative tail probabilities. Strictly within-window (uses only these rows).
    """
    W = r.size
    if W == 0:
        return np.nan, np.nan

    rs = np.sort(r)                       # ascending: rs[0] worst ... rs[-1] best
    ew = float(rs.mean())

    is_gain = rs >= 0.0
    n_gain = int(is_gain.sum())
    n_loss = W - n_gain

    pi = np.zeros(W, dtype="float64")

    # ---- gain side: outcomes rs[n_loss:], cumulate from the BEST downward ----
    if n_gain > 0:
        # number of gain outcomes >= this one, scanning from best (k=1) to worst gain (k=n_gain)
        # cumulative tail prob P(X >= outcome) uses gain-rank fraction k / W
        # pi for the j-th best gain = w(k/W) - w((k-1)/W)
        k = np.arange(1, n_gain + 1, dtype="float64")          # 1..n_gain, best->worst
        wk = _w(k / W, _GAMMA_GAIN)
        wkm1 = _w((k - 1.0) / W, _GAMMA_GAIN)
        pi_gain_best_first = wk - wkm1                          # decision weight, best gain first
        # rs gains in ascending order are rs[n_loss:]; best is last -> reverse to align best-first
        pi[n_loss:] = pi_gain_best_first[::-1]

    # ---- loss side: outcomes rs[:n_loss], cumulate from the WORST upward ----
    if n_loss > 0:
        k = np.arange(1, n_loss + 1, dtype="float64")          # 1..n_loss, worst->best loss
        wk = _w(k / W, _GAMMA_LOSS)
        wkm1 = _w((k - 1.0) / W, _GAMMA_LOSS)
        pi_loss_worst_first = wk - wkm1                         # decision weight, worst loss first
        # rs losses ascending rs[:n_loss]; worst is first -> already worst-first aligned
        pi[:n_loss] = pi_loss_worst_first

    pw = float(np.dot(pi, rs))
    gap = pw - ew

    # uptail: gain-side overweight relative to equal weight (1/W), dotted with gain returns
    if n_gain > 0:
        rs_gain = rs[n_loss:]
        pi_gain = pi[n_loss:]
        uptail = float(np.dot(pi_gain - (1.0 / W), rs_gain))
    else:
        uptail = 0.0

    return gap, uptail


def _rolling_cpt(r: np.ndarray, W: int, want_uptail: bool):
    """Apply _cpt_window over every trailing window of length W.

    Returns (gap_full, uptail_full) arrays of length n with leading NaN warmup.
    Uses sliding_window_view and the index convention out[W-1:] = val[:n-W+1].
    """
    n = r.size
    gap = np.full(n, np.nan, dtype="float64")
    uptail = np.full(n, np.nan, dtype="float64") if want_uptail else None
    if n < W:
        return gap, uptail

    windows = np.lib.stride_tricks.sliding_window_view(r, W)   # shape (n-W+1, W)
    m = windows.shape[0]
    g_vals = np.empty(m, dtype="float64")
    u_vals = np.empty(m, dtype="float64") if want_uptail else None
    for i in range(m):
        wnd = windows[i]
        if not np.isfinite(wnd).all():
            g_vals[i] = np.nan
            if want_uptail:
                u_vals[i] = np.nan
            continue
        gv, uv = _cpt_window(wnd)
        g_vals[i] = gv
        if want_uptail:
            u_vals[i] = uv

    gap[W - 1:] = g_vals
    if want_uptail:
        uptail[W - 1:] = u_vals
    return gap, uptail


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    # raw simple returns; first value NaN -> treated as non-finite in window guard
    ret = close.pct_change().to_numpy(dtype="float64")

    gap60, uptail60 = _rolling_cpt(ret, 60, want_uptail=True)
    gap120, _ = _rolling_cpt(ret, 120, want_uptail=False)

    df["pwg_gap_60"] = np.clip(gap60, -0.5, 0.5)
    df["pwg_uptail_60"] = np.clip(uptail60, -0.5, 0.5)
    df["pwg_gap_120"] = np.clip(gap120, -0.5, 0.5)

    return df
