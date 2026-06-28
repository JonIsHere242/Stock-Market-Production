import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METADATA = {
    "name":        "gp_input_features",
    "description": "Per-ticker rolling GP input features (information-theory, fractal, and entropy signals) for the cross-sectional GP pass.",
    "requires":    ["Close", "High", "Low", "Volume"],
    "produces": [
        "Return",
        "High_Low",
        "High_Close",
        "Low_Close",
        "time_reversal_asymmetry",
        "time_between_extremes",
        "volume_entropy",
        "information_rate_of_change",
        "emd_features_IMF_1_Energy",
        "emd_features_IMF_1_Std",
        "emd_features_IMF_2_Energy",
        "emd_features_IMF_2_Std",
        "entropy",
        "noise_WhiteNoise",
        "noise_PinkNoise",
        "higuchi_fractal_dimension",
        "lyapunov_exponent",
        "dfa_alpha",
        "spectral_entropy",
        "autocorrelation",
        "hurst_exponent",
        "higher_order_moments_Skewness",
        "higher_order_moments_Kurtosis",
    ],
    "tags":    ["gp", "experimental", "information_theory"],
    "version": "1.1",
    "author":  "migration from 3__AlphaSensitivity.py compute_gp_input_features",
}

# ---------------------------------------------------------------------------
# Hurst helpers (inlined from 3__AlphaSensitivity.py lines 237-247)
#
# NOTE: this is a faithful copy of the original R/S-style estimator. It is kept
# only for reference / parity testing — see _rolling_hurst_exponent below for
# why the live path no longer calls it.
# ---------------------------------------------------------------------------

def _hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def _rolling_hurst_exponent(series, window_size):
    """Rolling Hurst exponent — *always returns all-NaN* by construction.

    The original estimator iterates ``lags = range(2, 100)`` but the rolling
    window is only ``window_size`` (<= 60) bars long. For every lag >= window
    length, ``time_series[lag:]`` is empty, so ``np.std`` of the (empty)
    differences is NaN; ``tau`` therefore always contains NaNs and the final
    ``np.polyfit`` returns NaN coefficients. The result is an all-NaN column for
    *any* input — verified empirically against the original implementation.

    We reproduce that exact output (NaN per non-null position, same index) for
    free instead of burning ~0.45s/ticker computing it. The slow original is
    preserved above for reference/parity testing.
    """
    series_clean = series.dropna()
    return pd.Series(np.nan, index=series_clean.index)


# ---------------------------------------------------------------------------
# Vectorized rolling helpers for the two fractal features (DFA + Higuchi).
#
# Both originally ran as `rolling(window).apply(pyfunc, raw=True)`, i.e. one
# Python call per window (~640 windows/ticker each) — the bulk of the runtime.
# We vectorize the *full* windows with sliding_window_view and fall back to the
# exact original Python function for the handful of partial / degenerate windows
# so the output is numerically identical.
# ---------------------------------------------------------------------------

def _lin_slope(x, Y):
    """OLS slope of each row of Y (shape (N, k)) vs the shared 1-D x (len k).

    Algebraically identical to ``np.polyfit(x, y, 1)[0]`` for well-conditioned x
    (matches to ~1e-13), but computed once across all N rows.
    """
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Yc * xc).sum(axis=1) / denom


# --- exact per-window originals (used for partial windows + fallbacks) ------

def _higuchi(x, kmax=6):
    n = len(x)
    if n < kmax * 2:
        return 1.5
    lm = []
    for k in range(1, kmax + 1):
        lk = 0.0
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            lmk = np.sum(np.abs(np.diff(x[idxs]))) * (n - 1) / (k * (len(idxs) - 1))
            lk += lmk
        lm.append(lk / k)
    if len(lm) < 2:
        return 1.5
    slope, _ = np.polyfit(np.log(range(1, len(lm) + 1)), np.log(np.array(lm) + 1e-10), 1)
    return float(slope)


def _dfa(x):
    n = len(x)
    if n < 16:
        return 0.5
    y = np.cumsum(x - np.mean(x))
    max_scale = max(4, n // 4)
    scales = np.unique(np.round(np.logspace(1, np.log10(max_scale), 6)).astype(int))
    scales = scales[scales >= 4]
    flucts = []
    for s in scales:
        n_segs = n // s
        if n_segs < 1:
            continue
        f2 = 0.0
        for i in range(n_segs):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(len(seg), dtype=float)
            p = np.polyfit(t, seg, 1)
            f2 += np.mean((seg - np.polyval(p, t)) ** 2)
        flucts.append(np.sqrt(f2 / n_segs))
    if len(flucts) < 2:
        return 0.5
    valid = [(scales[i], flucts[i]) for i in range(len(flucts)) if flucts[i] > 0]
    if len(valid) < 2:
        return 0.5
    ls, lf = zip(*valid)
    alpha, _ = np.polyfit(np.log(ls), np.log(lf), 1)
    return float(np.clip(alpha, -1, 3))


# --- vectorized full-window kernels -----------------------------------------

def _higuchi_full(W, kmax=6):
    """Higuchi FD for every full window (rows of W, each length n)."""
    Nw, n = W.shape
    finite = np.isfinite(W).all(axis=1)
    out = np.empty(Nw, dtype=float)

    LM = np.empty((Nw, kmax), dtype=float)
    for k in range(1, kmax + 1):
        lk = np.zeros(Nw, dtype=float)
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            sub = W[:, idxs]
            lmk = np.abs(np.diff(sub, axis=1)).sum(axis=1) * (n - 1) / (k * (len(idxs) - 1))
            lk += lmk
        LM[:, k - 1] = lk / k

    x = np.log(np.arange(1, kmax + 1, dtype=float))
    out[finite] = _lin_slope(x, np.log(LM[finite] + 1e-10))

    # NaN-containing windows: defer to the exact original
    for j in np.nonzero(~finite)[0]:
        out[j] = _higuchi(W[j], kmax=kmax)
    return out


def _dfa_full(W):
    """DFA alpha for every full window (rows of W, each length n)."""
    Nw, n = W.shape
    finite = np.isfinite(W).all(axis=1)
    out = np.empty(Nw, dtype=float)

    Y = np.cumsum(W - W.mean(axis=1, keepdims=True), axis=1)
    max_scale = max(4, n // 4)
    scales = np.unique(np.round(np.logspace(1, np.log10(max_scale), 6)).astype(int))
    scales = scales[scales >= 4]

    flucts = np.empty((Nw, len(scales)), dtype=float)
    for si, s in enumerate(scales):
        n_segs = n // s
        seg = Y[:, :n_segs * s].reshape(Nw, n_segs, s)
        t = np.arange(s, dtype=float)
        tc = t - t.mean()
        denom = (tc ** 2).sum()
        seg_mean = seg.mean(axis=2, keepdims=True)
        slope = ((seg - seg_mean) * tc).sum(axis=2) / denom        # (Nw, n_segs)
        fitted = seg_mean + slope[..., None] * tc                   # intercept + slope*t
        resid = seg - fitted
        f2 = (resid ** 2).mean(axis=2).sum(axis=1)                  # (Nw,)
        flucts[:, si] = np.sqrt(f2 / n_segs)

    logs = np.log(scales.astype(float))
    # fast path: every scale produced a positive fluctuation and window is finite
    clean = finite & (flucts > 0).all(axis=1)
    out[clean] = np.clip(_lin_slope(logs, np.log(flucts[clean])), -1, 3)
    # everything else (zero fluct, NaN, degenerate) -> exact original
    for j in np.nonzero(~clean)[0]:
        out[j] = _dfa(W[j])
    return out


def _rolling_fast(arr, window, min_p, py_func, vec_func):
    """rolling(window, min_periods=min_p).apply(py_func, raw=True), vectorized.

    Partial windows (indices min_p-1 .. window-2) use the exact Python function;
    full windows (index window-1 onward) use the vectorized kernel.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for idx in range(min_p - 1, min(window - 1, n)):
        out[idx] = py_func(arr[:idx + 1])
    if n >= window:
        out[window - 1:] = vec_func(sliding_window_view(arr, window))
    return out


# ---------------------------------------------------------------------------
# Vectorized full-window kernels for the remaining rolling().apply() features.
#
# Each kernel takes W (shape (Nw, window), window == 60 here, so every row is a
# full window) and returns one value per row, numerically matching the original
# per-window Python function. The window length (60) is always >= every length
# guard in the originals (8, 4, 3, 16), so those guards never trip on full
# windows; the partial-window prefix still goes through the exact Python path
# via _rolling_fast, preserving identical output everywhere.
# ---------------------------------------------------------------------------

def _tbe_full(W):
    n = W.shape[1]
    return (np.argmax(W, axis=1) - np.argmin(W, axis=1)) / max(n - 1, 1)


def _lyap_full(W):
    d = np.abs(np.diff(W, axis=1))
    mask = d > 1e-10
    ld = np.where(mask, np.log(np.where(mask, d, 1.0)), 0.0)
    cnt = mask.sum(axis=1)
    out = np.where(cnt > 0, ld.sum(axis=1) / np.where(cnt > 0, cnt, 1), 0.0)
    return out


def _vol_ent_full(W):
    pos = W > 0
    cnt = pos.sum(axis=1)
    Wp = np.where(pos, W, 0.0)
    sums = Wp.sum(axis=1)
    # p = x / sum (only over positive entries); zeros contribute 0 to entropy term
    p = np.where(pos, Wp / np.where(sums[:, None] > 0, sums[:, None], 1.0), 0.0)
    term = np.where(pos, p * np.log(p + 1e-10), 0.0)
    ent = -term.sum(axis=1)
    return np.where(cnt >= 4, ent, 0.0)


def _ent_full(W):
    """Vectorized np.histogram(x, bins=10) Shannon entropy per row.

    Replicates numpy's uniform-bin histogram counting exactly (same edges and
    the same rightmost-edge inclusion), so the resulting entropy is identical.
    """
    Nw = W.shape[0]
    bins = 10
    mn = W.min(axis=1)
    mx = W.max(axis=1)
    # numpy: when min == max it expands range to (min-0.5, max+0.5)
    eq = mn == mx
    first = np.where(eq, mn - 0.5, mn)
    last = np.where(eq, mx + 0.5, mx)
    norm = bins / (last - first)
    # numpy computes idx = int((x - first) * norm), clipping the last edge in
    idx = ((W - first[:, None]) * norm[:, None]).astype(np.intp)
    # values exactly at the upper edge land in bin `bins`; numpy folds them back
    np.clip(idx, 0, bins - 1, out=idx)
    # additional numpy correction: decrement where x < edge[idx] (rounding), and
    # increment where x >= edge[idx+1]; replicate via recompute against edges.
    # Build per-row edges and apply the same two corrections numpy uses.
    edges = first[:, None] + np.arange(bins + 1)[None, :] * ((last - first)[:, None] / bins)
    # decrement: x < left edge of its bin
    left = np.take_along_axis(edges, idx, axis=1)
    idx = np.where(W < left, idx - 1, idx)
    np.clip(idx, 0, bins - 1, out=idx)
    # increment: x >= right edge of its bin (and not the last bin)
    right = np.take_along_axis(edges, idx + 1, axis=1)
    idx = np.where((W >= right) & (idx < bins - 1), idx + 1, idx)
    np.clip(idx, 0, bins - 1, out=idx)

    counts = np.zeros((Nw, bins), dtype=np.int64)
    rows = np.repeat(np.arange(Nw), W.shape[1])
    np.add.at(counts, (rows, idx.ravel()), 1)

    tot = counts.sum(axis=1) + 1e-10
    p = counts / tot[:, None]
    term = np.where(counts > 0, p * np.log(np.where(counts > 0, p, 1.0)), 0.0)
    return -term.sum(axis=1)


def _spectral_slope_full(W):
    n = W.shape[1]
    vals = np.abs(np.fft.rfft(W - W.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(n)[1:]
    vals = vals[:, 1:]
    x = np.log(freqs + 1e-10)
    return _lin_slope(x, np.log(vals + 1e-10))


def _spec_ent_full(W):
    ps = np.abs(np.fft.rfft(W - W.mean(axis=1, keepdims=True), axis=1)) ** 2
    ps = ps / (ps.sum(axis=1, keepdims=True) + 1e-10)
    return -(ps * np.log(ps + 1e-10)).sum(axis=1)


def _ac1_full(W):
    a = W[:, :-1]
    b = W[:, 1:]
    am = a.mean(axis=1, keepdims=True)
    bm = b.mean(axis=1, keepdims=True)
    ac = a - am
    bc = b - bm
    cov = (ac * bc).sum(axis=1)
    va = (ac ** 2).sum(axis=1)
    vb = (bc ** 2).sum(axis=1)
    denom = np.sqrt(va * vb)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / denom
    return np.where(np.isfinite(r), r, 0.0)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 23 per-ticker GP input features using a rolling window of 60 bars.

    Faithfully mirrors compute_gp_input_features() from 3__AlphaSensitivity.py.
    All columns are collected in a dict and joined once via pd.concat to avoid
    the PerformanceWarning that results from repeated in-place column insertion.

    The two fractal features (dfa_alpha, higuchi_fractal_dimension) are computed
    with vectorized sliding-window kernels, and hurst_exponent (provably all-NaN)
    is short-circuited — together these cut ~0.9s/ticker with identical output.

    NOTE: mixed-case column names (Return, High_Low, etc.) are intentional —
    a future cross-sectional GP pass consumes these exact names.
    """
    window = 60
    min_p  = max(window // 2, 10)

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    prev_c = close.shift(1)
    ret    = close.pct_change().fillna(0)

    new_cols = {}

    # ------------------------------------------------------------------
    # Basic price ratios
    # ------------------------------------------------------------------
    new_cols["Return"]     = ret
    new_cols["High_Low"]   = (high - low) / (close + 1e-10)
    new_cols["High_Close"] = (high - prev_c) / (prev_c + 1e-10)
    new_cols["Low_Close"]  = (low  - prev_c) / (prev_c + 1e-10)

    # ------------------------------------------------------------------
    # time_reversal_asymmetry: mean((x_t^2 * x_{t-1}) - (x_{t-1}^2 * x_t))
    # ------------------------------------------------------------------
    new_cols["time_reversal_asymmetry"] = (
        ret ** 2 * ret.shift(1) - ret.shift(1) ** 2 * ret
    ).rolling(window, min_periods=min_p).mean()

    # ------------------------------------------------------------------
    # time_between_extremes: (argmax - argmin) / window, in (-1, 1)
    # ------------------------------------------------------------------
    def _tbe(x):
        return (np.argmax(x) - np.argmin(x)) / max(len(x) - 1, 1)
    new_cols["time_between_extremes"] = pd.Series(
        _rolling_fast(close.to_numpy(dtype=float), window, min_p, _tbe, _tbe_full),
        index=close.index,
    )

    # ------------------------------------------------------------------
    # volume_entropy: normalized Shannon entropy of volume within window
    # ------------------------------------------------------------------
    def _vol_ent(x):
        x = x[x > 0]
        if len(x) < 4:
            return 0.0
        p = x / x.sum()
        return float(-np.sum(p * np.log(p + 1e-10)))
    new_cols["volume_entropy"] = pd.Series(
        _rolling_fast(volume.to_numpy(dtype=float), window, min_p, _vol_ent, _vol_ent_full),
        index=volume.index,
    )

    # ------------------------------------------------------------------
    # information_rate_of_change: mean absolute first difference of returns
    # ------------------------------------------------------------------
    new_cols["information_rate_of_change"] = ret.diff().abs().rolling(window, min_periods=min_p).mean()

    # ------------------------------------------------------------------
    # EMD approximations: IMF1 = close - fast_ma (high-freq),
    #                     IMF2 = fast_ma - slow_ma (mid-freq)
    # ------------------------------------------------------------------
    fast_ma = close.rolling(5, min_periods=2).mean()
    slow_ma = close.rolling(20, min_periods=5).mean()
    imf1 = close - fast_ma
    imf2 = fast_ma - slow_ma
    new_cols["emd_features_IMF_1_Energy"] = (imf1 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols["emd_features_IMF_1_Std"]    = imf1.rolling(window, min_periods=min_p).std()
    new_cols["emd_features_IMF_2_Energy"] = (imf2 ** 2).rolling(window, min_periods=min_p).mean()
    new_cols["emd_features_IMF_2_Std"]    = imf2.rolling(window, min_periods=min_p).std()

    # ------------------------------------------------------------------
    # entropy: Shannon entropy over 10 equal-width return buckets
    # ------------------------------------------------------------------
    def _ent(x):
        counts, _ = np.histogram(x, bins=10)
        p = counts / (counts.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    new_cols["entropy"] = pd.Series(
        _rolling_fast(ret.to_numpy(dtype=float), window, min_p, _ent, _ent_full),
        index=ret.index,
    )

    # ------------------------------------------------------------------
    # noise_WhiteNoise / noise_PinkNoise: proximity to spectral slope 0 or -1
    # ------------------------------------------------------------------
    def _spectral_slope(x):
        if len(x) < 8:
            return 0.0
        vals = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x))[1:]
        vals = vals[1:]
        if len(freqs) < 2:
            return 0.0
        slope, _ = np.polyfit(np.log(freqs + 1e-10), np.log(vals + 1e-10), 1)
        return float(slope)
    slopes = pd.Series(
        _rolling_fast(ret.to_numpy(dtype=float), window, min_p, _spectral_slope, _spectral_slope_full),
        index=ret.index,
    )
    new_cols["noise_WhiteNoise"] = np.exp(-(slopes ** 2) / 0.5)
    new_cols["noise_PinkNoise"]  = np.exp(-((slopes + 1.0) ** 2) / 0.5)

    # ------------------------------------------------------------------
    # higuchi_fractal_dimension  (vectorized full windows; exact partials)
    # ------------------------------------------------------------------
    new_cols["higuchi_fractal_dimension"] = pd.Series(
        _rolling_fast(close.to_numpy(dtype=float), window, min_p, _higuchi, _higuchi_full),
        index=close.index,
    )

    # ------------------------------------------------------------------
    # lyapunov_exponent: mean log of absolute price differences
    # ------------------------------------------------------------------
    def _lyap(x):
        d = np.abs(np.diff(x))
        d = d[d > 1e-10]
        return float(np.mean(np.log(d))) if len(d) > 0 else 0.0
    new_cols["lyapunov_exponent"] = pd.Series(
        _rolling_fast(close.to_numpy(dtype=float), window, min_p, _lyap, _lyap_full),
        index=close.index,
    )

    # ------------------------------------------------------------------
    # dfa_alpha: detrended fluctuation analysis scaling exponent
    #            (vectorized full windows; exact partials/fallbacks)
    # ------------------------------------------------------------------
    new_cols["dfa_alpha"] = pd.Series(
        _rolling_fast(ret.to_numpy(dtype=float), window, min_p, _dfa, _dfa_full),
        index=ret.index,
    )

    # ------------------------------------------------------------------
    # spectral_entropy: FFT power-spectrum entropy of returns
    # ------------------------------------------------------------------
    def _spec_ent(x):
        if len(x) < 8:
            return 0.0
        ps = np.abs(np.fft.rfft(x - x.mean())) ** 2
        ps = ps / (ps.sum() + 1e-10)
        return float(-np.sum(ps * np.log(ps + 1e-10)))
    new_cols["spectral_entropy"] = pd.Series(
        _rolling_fast(ret.to_numpy(dtype=float), window, min_p, _spec_ent, _spec_ent_full),
        index=ret.index,
    )

    # ------------------------------------------------------------------
    # autocorrelation: lag-1 autocorrelation of returns
    # ------------------------------------------------------------------
    def _ac1(x):
        if len(x) < 3:
            return 0.0
        cc = np.corrcoef(x[:-1], x[1:])
        return float(cc[0, 1]) if np.isfinite(cc[0, 1]) else 0.0
    new_cols["autocorrelation"] = pd.Series(
        _rolling_fast(ret.to_numpy(dtype=float), window, min_p, _ac1, _ac1_full),
        index=ret.index,
    )

    # ------------------------------------------------------------------
    # hurst_exponent: rolling R/S scaling exponent via inlined helper.
    # Provably all-NaN (see _rolling_hurst_exponent) — short-circuited.
    # ------------------------------------------------------------------
    new_cols["hurst_exponent"] = _rolling_hurst_exponent(close, window_size=window)

    # ------------------------------------------------------------------
    # higher_order_moments
    # ------------------------------------------------------------------
    new_cols["higher_order_moments_Skewness"] = ret.rolling(window, min_periods=min_p).skew()
    new_cols["higher_order_moments_Kurtosis"] = ret.rolling(window, min_periods=min_p).kurt()

    # Single concat — avoids PerformanceWarning from repeated in-place insertion
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# [AUDIT-CULL 2026-06-13] redundant near-duplicates removed from the model feature set.
# Reversible: DELETE this whole block to restore the columns. Original compute() above is
# untouched; this only drops the listed OUTPUT columns (each >=0.999 rank-correlated with a
# RETAINED feature -> tree-redundant). Rationale: Data/PaperFeed/cull_decision.md
_CULL_2026_06_13 = ['emd_features_IMF_1_Std']
_compute_precull = compute
def compute(df):
    return _compute_precull(df).drop(columns=_CULL_2026_06_13, errors="ignore")
