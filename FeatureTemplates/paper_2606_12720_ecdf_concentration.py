"""
Empirical CDF Concentration Features  —  arxiv:2606.12720
"On McDiarmid's Inequality under Dependence via Approximate Tensorization of Entropy"

The paper proves McDiarmid-type concentration inequalities for dependent random
variables via approximate tensorization of entropy (ATE), and establishes
Dvoretzky-Kiefer-Wolfowitz (DKW)-type bounds for empirical CDFs under dependence.

OHLCV translation:
  The COMPUTABLE METHOD is EMPIRICAL CDF ANALYSIS with DKW-inspired confidence bands.

  Key ideas from the paper translated to OHLCV:
    1. DKW bound: the empirical CDF F_n deviates from the true CDF F by at most
       eps_n = sqrt(log(2/delta) / (2n)) uniformly. Applied to rolling windows,
       this gives a "CDF confidence band width" that measures distributional certainty.

    2. KS-style deviation: compare rolling empirical CDF to a longer-horizon CDF.
       Large KS deviation = current distribution drifted from historical baseline.

    3. Entropy concentration: ATE relates to how "spread out" the log-partition
       function is. We proxy this with rolling variance of the log-density estimate
       (histogram-based entropy estimation over the return distribution).

    4. Condition number analog: the paper's Gaussian ATE constant is the condition
       number of the covariance. For OHLCV, we compute condition number of the
       rolling OHLCV covariance matrix as a regime-shift indicator.

  Features:
    - ecdf_ks_deviation_20v60: KS statistic comparing 20-bar vs 60-bar empirical CDFs
    - ecdf_dkw_band_20: DKW confidence band width for 20-bar window
    - ecdf_entropy_concentration_20: rolling log-density entropy on 20d returns
    - ecdf_ohlcv_condition_20: condition number (ratio of max/min eigenvalue) of
      rolling OHLCV covariance matrix (20-bar window)
    - ecdf_cdf_at_zero: fraction of last 20 bars with negative return (CDF at zero)
    - ecdf_tail_heaviness: ratio of empirical tails to Gaussian tails (kurtosis-free)

  Produces 6 columns prefixed "ecdf_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_12720_ecdf_concentration",
    "description": (
        "Empirical CDF concentration and DKW-band features on OHLCV returns; "
        "KS-style deviation between 20/60-bar windows, entropy concentration, "
        "and OHLCV covariance condition number; arxiv 2606.12720 (McDiarmid/ATE)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ecdf_ks_deviation_20v60",
        "ecdf_dkw_band_20",
        "ecdf_entropy_concentration_20",
        "ecdf_ohlcv_condition_20",
        "ecdf_cdf_at_zero",
        "ecdf_tail_heaviness",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2606.12720",
}


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sample KS statistic: sup|F_n(t) - G_m(t)|.
    Fast O(n log n) implementation.
    """
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan
    n, m = len(x), len(y)
    combined = np.concatenate([x, y])
    combined.sort()
    ecdf_x = np.searchsorted(np.sort(x), combined, side='right') / n
    ecdf_y = np.searchsorted(np.sort(y), combined, side='right') / m
    return float(np.max(np.abs(ecdf_x - ecdf_y)))


def _entropy_of_hist(x: np.ndarray, bins: int = 8) -> float:
    """Differential entropy estimate via histogram."""
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    counts, edges = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    bin_width = edges[1] - edges[0]
    if bin_width <= 0:
        return np.nan
    # Differential entropy: -sum(p * log(p/dx)) for non-zero bins
    mask = probs > 0
    return float(-np.sum(probs[mask] * np.log(probs[mask] / bin_width)))


def _condition_number_5d(ohlcv_block: np.ndarray) -> float:
    """
    Condition number of 5x5 OHLCV covariance matrix.
    ohlcv_block: shape (n, 5) — columns: Open, High, Low, Close, log(Volume)
    """
    if ohlcv_block.shape[0] < 6:
        return np.nan
    valid = ohlcv_block[np.all(np.isfinite(ohlcv_block), axis=1)]
    if valid.shape[0] < 6:
        return np.nan
    cov = np.cov(valid.T)
    # Eigenvalues of symmetric matrix
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) < 2:
        return np.nan
    return float(eigvals[-1] / eigvals[0])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    vol = df["Volume"].values.astype(float)
    n = len(close)

    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    lr_s = pd.Series(log_ret)

    # ---- KS deviation: 20-bar empirical CDF vs 60-bar empirical CDF -----------
    ks_dev = np.full(n, np.nan)
    for t in range(59, n):
        x20 = log_ret[t - 19: t + 1]
        x60 = log_ret[t - 59: t + 1]
        ks_dev[t] = _ks_statistic(x20, x60)
    df["ecdf_ks_deviation_20v60"] = ks_dev

    # ---- DKW confidence band width for 20-bar window --------------------------
    # DKW: eps_n = sqrt(log(2/delta) / (2n)), with delta=0.05
    # This is just a function of n when the window is fixed, but we weight
    # by the actual data std to make it informative:
    # Effective DKW band = eps_20 * std(returns_20) * sqrt(20)
    delta_dkw = 0.05
    eps_20 = np.sqrt(np.log(2.0 / delta_dkw) / (2 * 20))
    roll_std20 = lr_s.rolling(20, min_periods=6).std()
    # DKW band in return units: eps_20 * range of the empirical distribution
    roll_range20 = (
        lr_s.rolling(20, min_periods=6).max()
        - lr_s.rolling(20, min_periods=6).min()
    )
    df["ecdf_dkw_band_20"] = eps_20 * roll_range20

    # ---- Entropy concentration: rolling histogram entropy of returns -----------
    # Vectorized, output-identical replacement for the per-window
    # _entropy_of_hist(seg, bins=6) loop.  We replicate numpy's uniform-bin
    # histogram counting exactly (including its edge-correction passes) in a
    # single batched computation; windows with any non-finite value fall back to
    # the faithful per-window helper.
    entropy_20 = np.full(n, np.nan)
    if n > 19:
        bins_e = 6
        cs_r = np.concatenate(([0], np.cumsum(np.isfinite(log_ret).astype(np.int64))))
        win_n = cs_r[20:] - cs_r[:-20]  # finite count per 20-window, t in [19,n-1]
        ts_e = np.arange(19, n)
        clean_e = win_n == 20
        clean_te = ts_e[clean_e]
        if clean_te.size:
            idx_e = clean_te[:, None] - 19 + np.arange(20)[None, :]
            Wb = log_ret[idx_e]  # (K, 20), all finite
            first = Wb.min(axis=1)
            last = Wb.max(axis=1)
            eq = first == last
            fe = np.where(eq, first - 0.5, first)
            le = np.where(eq, last + 0.5, last)
            # numpy uses linspace(first, last, bins+1) for edges
            edges = np.linspace(fe, le, bins_e + 1, axis=1)  # (K, bins+1)
            norm = bins_e / (le - fe)
            f_idx = (Wb - fe[:, None]) * norm[:, None]
            bidx = f_idx.astype(np.intp)
            bidx[bidx == bins_e] = bins_e - 1
            take_lo = np.take_along_axis(edges, bidx, axis=1)
            bidx = bidx - (Wb < take_lo)
            take_hi = np.take_along_axis(edges, bidx + 1, axis=1)
            bidx = bidx + ((Wb >= take_hi) & (bidx != bins_e - 1))
            counts = np.zeros((clean_te.size, bins_e), dtype=np.int64)
            np.add.at(counts, (np.arange(clean_te.size)[:, None], bidx), 1)
            probs = counts / counts.sum(axis=1, keepdims=True)
            bin_width = edges[:, 1] - edges[:, 0]
            res_e = np.full(clean_te.size, np.nan)
            bw_ok = bin_width > 0
            if bw_ok.any():
                p_ok = probs[bw_ok]
                bw = bin_width[bw_ok]
                mask = p_ok > 0
                ratio = np.where(mask, p_ok / bw[:, None], 1.0)
                terms = np.where(mask, p_ok * np.log(ratio), 0.0)
                res_e[bw_ok] = -terms.sum(axis=1)
            entropy_20[clean_te] = res_e
        for t in ts_e[~clean_e]:
            entropy_20[t] = _entropy_of_hist(log_ret[t - 19: t + 1], bins=6)
    df["ecdf_entropy_concentration_20"] = entropy_20

    # ---- OHLCV covariance condition number (20-bar) ----------------------------
    # Log-transform volume; standardize each column to prevent scale dominance
    log_vol = np.log(np.where(vol > 0, vol, np.nan))
    log_open = np.log(np.where(open_ > 0, open_, np.nan))
    log_high = np.log(np.where(high > 0, high, np.nan))
    log_low = np.log(np.where(low > 0, low, np.nan))
    log_close = np.log(np.where(close > 0, close, np.nan))

    ohlcv_mat = np.column_stack([log_open, log_high, log_low, log_close, log_vol])
    cond_num = np.full(n, np.nan)
    # Fast batched path for the (overwhelmingly common) windows whose 20 rows are
    # all finite: vectorize the 5x5 covariance and run eigvalsh on the whole batch
    # at once.  Windows containing any non-finite row fall back to the faithful
    # per-window helper (which drops non-finite rows before np.cov).
    row_finite = np.all(np.isfinite(ohlcv_mat), axis=1)
    if n > 19:
        # number of finite rows in each 20-bar window ending at t
        cs = np.concatenate(([0], np.cumsum(row_finite.astype(np.int64))))
        win_finite = cs[20:] - cs[:-20]  # length n-19, aligned to t in [19, n-1]
        ts = np.arange(19, n)
        clean_mask = win_finite == 20

        clean_ts = ts[clean_mask]
        if clean_ts.size:
            # Stack clean blocks: shape (K, 20, 5)
            idx = clean_ts[:, None] - 19 + np.arange(20)[None, :]
            blocks = ohlcv_mat[idx]  # (K, 20, 5)
            # Covariance with ddof=1 (matches np.cov): center then X^T X / (m-1)
            m = 20
            mean = blocks.mean(axis=1, keepdims=True)  # (K,1,5)
            centered = blocks - mean  # (K,20,5)
            cov = np.einsum("kmi,kmj->kij", centered, centered) / (m - 1)  # (K,5,5)
            eig = np.linalg.eigvalsh(cov)  # (K,5) ascending
            # mimic helper: keep eigvals > 0, need >=2, ratio max/min(>0)
            pos = eig > 0
            npos = pos.sum(axis=1)
            res = np.full(clean_ts.size, np.nan)
            ok = npos >= 2
            if ok.any():
                eig_ok = eig[ok]
                pos_ok = pos[ok]
                # smallest positive eigenvalue per row
                masked = np.where(pos_ok, eig_ok, np.inf)
                min_pos = masked.min(axis=1)
                max_eig = eig_ok[:, -1]  # eigvalsh ascending -> largest last
                res[ok] = max_eig / min_pos
            cond_num[clean_ts] = res

        # Faithful fallback for any window with non-finite rows
        for t in ts[~clean_mask]:
            cond_num[t] = _condition_number_5d(ohlcv_mat[t - 19: t + 1])

    # Cap at 1e6 to avoid inf from near-singular matrices
    df["ecdf_ohlcv_condition_20"] = np.minimum(cond_num, 1e6)

    # ---- CDF at zero: fraction of 20 bars with negative return ----------------
    df["ecdf_cdf_at_zero"] = (lr_s < 0).astype(float).rolling(20, min_periods=6).mean()

    # ---- Tail heaviness: ratio of empirical tail mass to Gaussian expectation --
    # P(|Z| > 1.5*sigma) empirical vs P(|Z| > 1.5) Gaussian = 2*Phi(-1.5) ~ 0.134
    gauss_tail_prob = 0.1336  # P(|Z| > 1.5) for standard normal
    # Vectorized equivalent of the rolling(40, min_periods=12).apply(tail_ratio).
    # pandas calls the func on windows with >=12 non-NaN values; the func then
    # computes population mean/std over the finite values and the fraction with
    # |z| > 1.5.  We reproduce this exactly with a strided window view.
    W = 40

    def _tail_ratio(valid):
        if len(valid) < 6:
            return np.nan
        mu = np.mean(valid)
        sd = np.std(valid)
        if sd < 1e-10:
            return np.nan
        z = np.abs((valid - mu) / sd)
        return np.mean(z > 1.5) / gauss_tail_prob

    tail_out = np.full(n, np.nan)
    # Leading partial windows: pandas emits a value once >=12 non-NaN are seen,
    # using the growing window [0 : t+1] for t < W-1.
    lead_end = min(W - 1, n)
    for t in range(11, lead_end):
        seg = log_ret[: t + 1]
        v = seg[np.isfinite(seg)]
        if len(v) >= 12:
            tail_out[t] = _tail_ratio(v)
    if n >= W:
        # (n-W+1, W) sliding windows ending at t = W-1 .. n-1
        win = np.lib.stride_tricks.sliding_window_view(log_ret, W)  # (n-W+1, W)
        ends = np.arange(W - 1, n)
        fin = np.isfinite(win)
        cnt = fin.sum(axis=1)
        valid_rows = cnt >= 12  # pandas min_periods gate
        if valid_rows.any():
            wv = win[valid_rows]
            fv = fin[valid_rows]
            cv = cnt[valid_rows].astype(float)
            x0 = np.where(fv, wv, 0.0)
            mu = x0.sum(axis=1) / cv
            # population variance over finite values
            d = np.where(fv, wv - mu[:, None], 0.0)
            var = (d * d).sum(axis=1) / cv
            sd = np.sqrt(var)
            out = np.full(wv.shape[0], np.nan)
            sd_ok = sd >= 1e-10
            if sd_ok.any():
                thr = 1.5 * sd[sd_ok]
                fok = fv[sd_ok]
                dok = np.abs(wv[sd_ok] - mu[sd_ok, None])
                tail_cnt = (np.where(fok, dok, -np.inf) > thr[:, None]).sum(axis=1)
                emp_tail = tail_cnt / cv[sd_ok]
                out[sd_ok] = emp_tail / gauss_tail_prob
            tail_out[ends[valid_rows]] = out
    df["ecdf_tail_heaviness"] = tail_out

    return df
