"""
Candidate feature block: xdom_transfer_entropy_vix
Approximate transfer entropy from VIX changes -> stock |return| on a rolling window.
Based on Schreiber (2000) transfer entropy, implemented as a binned conditional
mutual information estimate on per-ticker OHLCV + VIX data.
"""
from __future__ import annotations

import warnings
import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: load _indexes
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "xdom_transfer_entropy_vix",
    "description": (
        "Per-ticker rolling approximation of transfer entropy from VIX daily changes "
        "to the stock's absolute daily returns, following Schreiber (2000). "
        "Transfer entropy TE(X->Y) = sum p(y_{t+1}, y_t, x_t) log [p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t)] "
        "is estimated via 3-bin discretisation and a 120-day rolling window (binned conditional "
        "mutual information proxy). A slope variant captures trend in this coupling strength. "
        "NOTE: inherently cross-sectional TE (market->stock) is faithfully reproduced per-ticker "
        "because VIX is a common driver; the cross-sectional ranking of this value is the "
        "original economic signal (which stock responds most to vol regime information)."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_transfer_entropy_vix_te120",   # rolling 120d TE estimate
        "xdom_transfer_entropy_vix_slope",   # 20d slope of TE120 (trend in coupling)
        "xdom_transfer_entropy_vix_norm",    # TE120 normalised by stock entropy (fraction of info explained)
    ],
    "tags": ["cross-domain", "information-theory", "vix", "econophysics", "transfer-entropy"],
    "version": "1.0",
    "author": (
        "Spec: Cross-domain method transfer (signal processing / econophysics / HRV / DSP). "
        "Citation: Schreiber T. (2000) 'Measuring Information Transfer', PRL 85(2), 461-464. "
        "Implementation: per-ticker binned CMI proxy."
    ),
}

# ---------------------------------------------------------------------------
# Core TE estimator (vectorised over a 1-D window)
# ---------------------------------------------------------------------------

_N_BINS = 3  # 3-bin discretisation (coarse but robust for n~120)
_WIN = 120   # rolling window length
_SLOPE_WIN = 20  # window for slope of TE over time


def _binned_te(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """
    Estimate TE(X->Y) using binned conditional mutual information.

    TE(X->Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
             = sum p(y1, y0, x0) * log2[ p(y1|y0,x0) / p(y1|y0) ]

    x_vals: X_t  (VIX change, lagged 1 relative to y)
    y_vals: Y_t  (stock |ret|, current)

    Both arrays are aligned: x_vals[i] = X at time t, y_vals[i] = Y at time t.
    We form triplets: (x_vals[:-1], y_vals[:-1], y_vals[1:]) = (X_t, Y_t, Y_{t+1}).
    """
    n = len(x_vals)
    if n < _N_BINS * 3 + 2:
        return np.nan

    # Build quantile bin edges from the window data (avoid lookahead: computed
    # inside each window so only in-window distribution is used)
    def qbin(arr: np.ndarray, n_bins: int) -> np.ndarray:
        """Quantile-bin arr into 0..n_bins-1 integer labels."""
        edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        # ensure strict monotonicity to avoid duplicate edges
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.zeros(len(arr), dtype=np.int8)
        labels = np.searchsorted(edges[1:-1], arr, side="right").astype(np.int8)
        return labels

    x_t = x_vals[:-1]       # X_t
    y_t = y_vals[:-1]       # Y_t
    y_t1 = y_vals[1:]       # Y_{t+1}

    # Discretise
    xb = qbin(x_t, _N_BINS)
    yb = qbin(y_t, _N_BINS)
    y1b = qbin(y_t1, _N_BINS)

    N = len(y_t1)

    # Count triplets (y1, y, x) and pairs (y1, y), (y,)
    n_states_3 = _N_BINS ** 3
    n_states_2 = _N_BINS ** 2

    # Flatten indices
    idx3 = y1b * (_N_BINS * _N_BINS) + yb * _N_BINS + xb  # (y1, y, x)
    idx2_y1y = y1b * _N_BINS + yb                           # (y1, y)
    idx2_y = yb                                              # (y,)

    # Counts via np.bincount
    cnt3 = np.bincount(idx3, minlength=n_states_3).reshape(_N_BINS, _N_BINS, _N_BINS).astype(np.float64)
    cnt_y1y = np.bincount(idx2_y1y, minlength=n_states_2).reshape(_N_BINS, _N_BINS).astype(np.float64)
    cnt_y = np.bincount(idx2_y, minlength=_N_BINS).astype(np.float64)

    # Probabilities (add tiny epsilon to avoid log(0) – standard approach)
    eps = 1e-12
    p3 = cnt3 / (N + eps)
    p_y1_y = cnt_y1y / (N + eps)   # marginal p(y1, y)
    p_y = cnt_y / (N + eps)

    # p(y1 | y, x) = p(y1, y, x) / p(y, x)
    # p(y, x) = sum over y1 of cnt3 -> shape (y, x)
    p_yx = cnt3.sum(axis=0) / (N + eps)  # shape (n_bins, n_bins) = (y, x)

    # p(y1 | y) = p(y1, y) / p(y)
    # shape (y1, y)

    te = 0.0
    for y1 in range(_N_BINS):
        for y in range(_N_BINS):
            for x in range(_N_BINS):
                p_joint = p3[y1, y, x]
                if p_joint < eps:
                    continue
                p_cond_yx = p_joint / (p_yx[y, x] + eps)      # p(y1|y,x)
                p_cond_y = p_y1_y[y1, y] / (p_y[y] + eps)     # p(y1|y)
                if p_cond_y < eps:
                    continue
                te += p_joint * np.log2(p_cond_yx / (p_cond_y + eps))

    return max(te, 0.0)  # TE >= 0 by definition; clamp numerical noise


def _stock_entropy(y_vals: np.ndarray) -> float:
    """Marginal entropy of Y_{t+1} in nats (for normalisation)."""
    n = len(y_vals)
    if n < _N_BINS + 2:
        return np.nan
    y_t1 = y_vals[1:]
    N = len(y_t1)
    eps = 1e-12
    edges = np.unique(np.quantile(y_t1, np.linspace(0, 1, _N_BINS + 1)))
    if len(edges) < 2:
        return np.nan
    labels = np.searchsorted(edges[1:-1], y_t1, side="right")
    cnt = np.bincount(labels, minlength=_N_BINS).astype(np.float64)
    p = cnt / (N + eps)
    h = -np.sum(p * np.log2(p + eps))
    return h if h > eps else np.nan


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ---- Fetch VIX --------------------------------------------------------
    try:
        vix_df = _indexes.vix_daily_close()
        # vix_df has columns ["Date", "vix_close"]
        df_sorted = df.copy()
        df_sorted["_orig_idx"] = np.arange(n)

        # merge_asof requires both on-cols sorted
        df_m = pd.merge_asof(
            df_sorted.sort_values("Date"),
            vix_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        df_m = df_m.sort_values("_orig_idx").drop(columns=["_orig_idx"])
        vix_close = df_m["vix_close"].values.astype(np.float64)
    except Exception:
        vix_close = np.full(n, np.nan)

    # ---- Stock absolute return --------------------------------------------
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pct_ret = np.where(
            (close[:-1] != 0) & np.isfinite(close[:-1]) & np.isfinite(close[1:]),
            (close[1:] - close[:-1]) / close[:-1],
            np.nan,
        )
    abs_ret = np.concatenate([[np.nan], np.abs(pct_ret)])  # align to index

    # ---- VIX change -------------------------------------------------------
    vix_chg = np.concatenate([[np.nan], np.diff(vix_close)])  # VIX_{t} - VIX_{t-1}

    # ---- Rolling TE -------------------------------------------------------
    te120 = np.full(n, np.nan)
    h_stock = np.full(n, np.nan)

    for i in range(_WIN - 1, n):
        start = i - _WIN + 1
        x_win = vix_chg[start : i + 1]
        y_win = abs_ret[start : i + 1]

        # skip if too many NaNs
        valid = np.isfinite(x_win) & np.isfinite(y_win)
        if valid.sum() < _WIN // 2:
            continue

        # Use only valid positions (compact them; may break time-contiguity
        # slightly but is sufficient for distributional TE estimate)
        x_v = x_win[valid]
        y_v = y_win[valid]

        te120[i] = _binned_te(x_v, y_v)
        h_stock[i] = _stock_entropy(y_v)

    # ---- Slope of TE over time (20d rolling linear trend) -----------------
    te_series = pd.Series(te120)
    slope20 = np.full(n, np.nan)

    _x_reg = np.arange(_SLOPE_WIN, dtype=np.float64)
    _x_reg_dm = _x_reg - _x_reg.mean()
    _x_denom = (_x_reg_dm ** 2).sum()

    for i in range(_SLOPE_WIN - 1, n):
        window = te120[i - _SLOPE_WIN + 1 : i + 1]
        if not np.isfinite(window).all():
            continue
        y_dm = window - window.mean()
        slope20[i] = np.dot(_x_reg_dm, y_dm) / (_x_denom + 1e-30)

    # ---- Normalised TE (fraction of stock entropy explained by VIX) -------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te_norm = np.where(
            np.isfinite(te120) & np.isfinite(h_stock) & (h_stock > 0),
            te120 / h_stock,
            np.nan,
        )
    # clip to [0, 1]
    te_norm = np.clip(te_norm, 0.0, 1.0)

    df["xdom_transfer_entropy_vix_te120"] = te120
    df["xdom_transfer_entropy_vix_slope"] = slope20
    df["xdom_transfer_entropy_vix_norm"] = te_norm

    return df
