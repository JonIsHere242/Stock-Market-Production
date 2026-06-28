"""
_paper_2606_07450_transfer_entropy_flow.py
==========================================

Inspired by: "Information Networks of Stock Prices" (arXiv 2606.07450).

The paper models markets as directed information networks where edges carry
transfer entropy (TE) — the information-theoretic measure of directed
information flow between time series.  TE(X→Y) quantifies how much knowing
X's past reduces uncertainty about Y's future, BEYOND what Y's own past
already provides.  Unlike correlation or beta, TE is non-linear, asymmetric,
and captures non-linear dependencies (regime shifts, tail interactions).

SIGNALS COMPUTED  (prefix: tef_)
---------------------------------
tef_idx_to_stock  : TE(index → stock):  bits of information the index's
                    past state adds to predicting this stock's next state
                    (beyond the stock's own past).  High = index strongly
                    informs the stock.
tef_stock_to_idx  : TE(stock → index):  information the stock's past feeds
                    back into the index.  Usually small; spikes can flag
                    systemic events or high-cap names driving the index.
tef_net_flow      : tef_idx_to_stock − tef_stock_to_idx.  Positive = net
                    information flows FROM the index INTO the stock; the
                    stock is a passive receiver.  Negative = the stock feeds
                    information into the market (rare, notable).
tef_mutual_info   : Mutual information I(X;Y) over the same window: total
                    statistical coupling between the two return sequences,
                    direction-agnostic.
tef_directionality: tef_net_flow / (tef_idx_to_stock + tef_stock_to_idx + ε).
                    ∈ (−1, +1).  +1 = fully index-driven; −1 = stock drives
                    the index; 0 = symmetric coupling or both near zero.

IMPLEMENTATION NOTES
--------------------
- Discretisation: sign-based 3-bin alphabet {−1, 0, +1} (down / flat / up)
  with |ret| < ε_flat treated as flat.  Coarse bins keep count estimates
  reliable at short windows.
- Window: 40 trading days (about 2 months) — short enough to be reactive,
  long enough for stable 3×3 joint count matrices (expected ~4.4 obs/cell).
- TE formula (order-1 Markov):
      TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) · log2[ p(y_{t+1}|y_t, x_t)
                                                  / p(y_{t+1}|y_t) ]
  Computed from empirical counts with Laplace (+1) smoothing.
- ALL quantities use only on-or-before-t data (backward merge_asof, trailing
  window); no future data enters any row.  Causality-safe.

IMPORTANT NOTES
---------------
- UNPROVEN candidate — leading _ keeps this out of auto-discovery.  Promote
  (remove leading _) only after multi-seed backtest validation.
- Honest description: "per-ticker OHLCV + SPY index proxy; transfer entropy
  computed from discretised log-return sequences; no external data beyond
  _indexes.py SPY / fallback symbol."
- If SPY (or any index symbol) is unavailable, all produced columns are NaN.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the shared index helper by path (mirrors vix_features.py / ixr block).
# Auto-discovery skips _indexes.py because of the leading underscore.
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "paper_2606_07450_transfer_entropy_flow",
    "description": (
        "Per-ticker information-theoretic transfer entropy proxy inspired by "
        "arXiv 2606.07450: directed TE(index→stock) and TE(stock→index), net "
        "information flow, mutual information, and a directionality ratio — all "
        "computed from discretised log-return sequences over a rolling 40-day window."
    ),
    "requires": ["Date", "Close"],
    "produces": [
        "tef_idx_to_stock",
        "tef_stock_to_idx",
        "tef_net_flow",
        "tef_mutual_info",
        "tef_directionality",
    ],
    "tags": ["market_regime", "information_theory", "experimental"],
    "version": "1.0",
    "author": "paper arXiv 2606.07450 — information networks of stock prices; TE proxy",
}


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_WINDOW   = 40     # rolling window (trading days)
_MIN_OBS  = 25     # minimum valid observations inside window to emit a value
_EPS_FLAT = 5e-4   # |ret| < this → "flat" bin (sign-based discretisation)
_SMOOTH   = 1.0    # Laplace smoothing count (add-one per cell)
_EPS_DIV  = 1e-12  # denominator guard for directionality ratio


# ---------------------------------------------------------------------------
# Core TE helper  (pure numpy, no pandas inside the loop)
# ---------------------------------------------------------------------------

def _discretise(ret_arr: np.ndarray, eps_flat: float = _EPS_FLAT) -> np.ndarray:
    """
    Map a float return array to integer alphabet {0=down, 1=flat, 2=up}.
    NaN inputs map to -1 (sentinel; excluded from window counts).
    """
    out = np.full(len(ret_arr), -1, dtype=np.int8)
    valid = np.isfinite(ret_arr)
    out[valid & (ret_arr < -eps_flat)] = 0   # down
    out[valid & (ret_arr > eps_flat)]  = 2   # up
    out[valid & (np.abs(ret_arr) <= eps_flat)] = 1  # flat
    return out


def _te_from_window(
    y_sym: np.ndarray,   # discretised Y sequence (current stock or index), length W
    x_sym: np.ndarray,   # discretised X sequence (driver), length W
    alpha: int = 3,      # alphabet size
    smooth: float = _SMOOTH,
) -> float:
    """
    Compute TE(X→Y) from two aligned symbol arrays of length W using
    empirical counts + Laplace smoothing.

    Returns TE in bits (log2).  Returns NaN if insufficient valid pairs.

    TE(X→Y) = Σ_{y', y, x} p(y', y, x) * log2[ p(y'|y,x) / p(y'|y) ]

    where y' = Y[t+1], y = Y[t], x = X[t].
    """
    W = len(y_sym)
    if W < 3:
        return np.nan

    # Build triplets (y_t, x_t, y_{t+1}) from positions 0..W-2
    y_cur  = y_sym[:-1]   # Y[t]
    x_cur  = x_sym[:-1]   # X[t]
    y_next = y_sym[1:]    # Y[t+1]

    # Only use positions where ALL three are valid (>= 0)
    valid = (y_cur >= 0) & (x_cur >= 0) & (y_next >= 0)
    y_cur  = y_cur[valid].astype(int)
    x_cur  = x_cur[valid].astype(int)
    y_next = y_next[valid].astype(int)

    n_valid = valid.sum()
    if n_valid < _MIN_OBS:
        return np.nan

    # Joint count table  C[y_next, y_cur, x_cur]  shape (alpha, alpha, alpha)
    joint3 = np.zeros((alpha, alpha, alpha), dtype=np.float64) + smooth
    np.add.at(joint3, (y_next, y_cur, x_cur), 1.0)

    # Marginal over y_next:  C[y_cur, x_cur]  shape (alpha, alpha)
    joint2_yx = joint3.sum(axis=0)   # sum over y_next → shape (alpha, alpha)

    # Marginal over x: C[y_next, y_cur]  shape (alpha, alpha)
    joint2_yy = joint3.sum(axis=2)   # sum over x_cur

    # Marginal over x and y_next: C[y_cur]  shape (alpha,)
    marg_y = joint2_yx.sum(axis=1)   # sum over x_cur → shape (alpha,)

    # Normalise to probabilities
    N3 = joint3.sum()
    p_yyx = joint3 / N3                         # p(y', y, x)
    # p(y'|y,x) = joint3 / joint2_yx  (broadcast over y_next axis)
    # p(y'|y)   = joint2_yy / marg_y  (broadcast over x and y_next)

    with np.errstate(divide="ignore", invalid="ignore"):
        # conditional p(y'|y,x): shape (alpha,alpha,alpha)
        p_yprime_given_yx = joint3 / joint2_yx[np.newaxis, :, :]

        # conditional p(y'|y): shape (alpha, alpha)
        p_yprime_given_y = joint2_yy / marg_y[np.newaxis, :]  # (alpha, alpha)

        # log ratio: log2[ p(y'|y,x) / p(y'|y) ]
        # p(y'|y) needs to broadcast over x → shape (alpha, alpha, 1)
        log_ratio = np.log2(
            p_yprime_given_yx / p_yprime_given_y[:, :, np.newaxis].transpose(0, 2, 1)
        )

    # TE = sum p(y',y,x) * log_ratio
    # mask out any 0-probability or non-finite terms
    mask = np.isfinite(log_ratio) & (p_yyx > 0)
    te = np.sum(np.where(mask, p_yyx * log_ratio, 0.0))

    return float(max(te, 0.0))   # TE is non-negative; clip numerical noise


def _mutual_info_from_window(
    y_sym: np.ndarray,
    x_sym: np.ndarray,
    alpha: int = 3,
    smooth: float = _SMOOTH,
) -> float:
    """
    Compute I(X;Y) = Σ p(x,y) log2[p(x,y)/(p(x)p(y))] from a window.
    Uses contemporaneous pairs (no lag).  Returns NaN if insufficient.
    """
    valid = (y_sym >= 0) & (x_sym >= 0)
    y_v = y_sym[valid].astype(int)
    x_v = x_sym[valid].astype(int)

    if valid.sum() < _MIN_OBS:
        return np.nan

    # Joint count table shape (alpha, alpha)
    joint = np.zeros((alpha, alpha), dtype=np.float64) + smooth
    np.add.at(joint, (y_v, x_v), 1.0)

    N = joint.sum()
    p_xy = joint / N
    p_x  = p_xy.sum(axis=0, keepdims=True)   # (1, alpha)
    p_y  = p_xy.sum(axis=1, keepdims=True)   # (alpha, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(p_xy / (p_y * p_x))

    mask = np.isfinite(log_ratio) & (p_xy > 0)
    mi = np.sum(np.where(mask, p_xy * log_ratio, 0.0))
    return float(max(mi, 0.0))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transfer-entropy flow features to a single-ticker OHLCV frame.

    df is per-ticker, ascending by Date, ~700 rows.
    Only the columns in METADATA['produces'] are added; nothing else is touched.
    """
    n = len(df)
    _nan_series = pd.Series(np.nan, index=df.index)

    # Pre-allocate output arrays
    te_idx_to_stock = np.full(n, np.nan)
    te_stock_to_idx = np.full(n, np.nan)
    mi_vals         = np.full(n, np.nan)

    # -----------------------------------------------------------------------
    # 1.  Resolve index symbol and build a look-ahead-safe aligned series
    # -----------------------------------------------------------------------
    avail = _indexes.available()
    if not avail:
        # No index data — emit all-NaN and return
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    symbol = "SPY" if "SPY" in avail else avail[0]
    idx_close_raw = _indexes.index_close(symbol)   # pd.Series, DatetimeIndex

    if idx_close_raw.empty:
        for col in METADATA["produces"]:
            df[col] = np.nan
        return df

    # Build a lookup table for merge_asof
    idx_lookup = (
        idx_close_raw
        .reset_index()
        .rename(columns={"Close": "_idx_close"})
    )
    idx_lookup["Date"] = pd.to_datetime(idx_lookup["Date"])
    idx_lookup = idx_lookup.sort_values("Date").reset_index(drop=True)

    ticker_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(
        ticker_dates,
        idx_lookup[["Date", "_idx_close"]],
        on="Date",
        direction="backward",   # NEVER forward — no look-ahead
    )
    idx_close_aligned = merged["_idx_close"].to_numpy(dtype=np.float64)

    # -----------------------------------------------------------------------
    # 2.  Log returns for both series (row-aligned)
    # -----------------------------------------------------------------------
    stock_close = df["Close"].to_numpy(dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        stock_ret = np.empty(n, dtype=np.float64)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(stock_close[1:] / stock_close[:-1])

        idx_ret = np.empty(n, dtype=np.float64)
        idx_ret[0] = np.nan
        idx_ret[1:] = np.log(idx_close_aligned[1:] / idx_close_aligned[:-1])

    # Sanitise: replace inf → nan
    stock_ret = np.where(np.isfinite(stock_ret), stock_ret, np.nan)
    idx_ret   = np.where(np.isfinite(idx_ret),   idx_ret,   np.nan)

    # -----------------------------------------------------------------------
    # 3.  Discretise to 3-bin alphabet
    # -----------------------------------------------------------------------
    stock_sym = _discretise(stock_ret)
    idx_sym   = _discretise(idx_ret)

    # -----------------------------------------------------------------------
    # 4.  Rolling transfer-entropy loop
    #
    #     At each time t we use the trailing window [t-W+1 .. t] (inclusive).
    #     Everything in the window is from row t or earlier → no look-ahead.
    #     We start emitting at t = WINDOW-1 (index _WINDOW-1).
    # -----------------------------------------------------------------------
    W = _WINDOW

    for t in range(W - 1, n):
        s_win = stock_sym[t - W + 1: t + 1]   # length W
        i_win = idx_sym[t - W + 1: t + 1]     # length W

        # Transfer entropy in both directions
        te_i2s = _te_from_window(s_win, i_win)   # TE(idx → stock)
        te_s2i = _te_from_window(i_win, s_win)   # TE(stock → idx)

        te_idx_to_stock[t] = te_i2s
        te_stock_to_idx[t] = te_s2i

        # Mutual information (contemporaneous, no lag)
        mi_vals[t] = _mutual_info_from_window(s_win, i_win)

    # -----------------------------------------------------------------------
    # 5.  Derived columns from the raw TE arrays
    # -----------------------------------------------------------------------
    net_flow = np.where(
        np.isfinite(te_idx_to_stock) & np.isfinite(te_stock_to_idx),
        te_idx_to_stock - te_stock_to_idx,
        np.nan,
    )

    denom = te_idx_to_stock + te_stock_to_idx + _EPS_DIV
    directionality = np.where(
        np.isfinite(net_flow),
        net_flow / denom,
        np.nan,
    )

    # -----------------------------------------------------------------------
    # 6.  Safety: TE values must be non-negative; clip numerical noise
    # -----------------------------------------------------------------------
    te_idx_to_stock = np.where(te_idx_to_stock >= 0, te_idx_to_stock, np.nan)
    te_stock_to_idx = np.where(te_stock_to_idx >= 0, te_stock_to_idx, np.nan)
    mi_vals         = np.where(mi_vals >= 0, mi_vals, np.nan)

    # -----------------------------------------------------------------------
    # 7.  Assemble — only ADD produced columns, never touch existing ones
    # -----------------------------------------------------------------------
    new_cols = pd.DataFrame(
        {
            "tef_idx_to_stock":  te_idx_to_stock,
            "tef_stock_to_idx":  te_stock_to_idx,
            "tef_net_flow":      net_flow,
            "tef_mutual_info":   mi_vals,
            "tef_directionality": directionality,
        },
        index=df.index,
    )

    # Final guard: no inf anywhere
    new_cols = new_cols.replace([np.inf, -np.inf], np.nan)

    return pd.concat([df, new_cols], axis=1)
