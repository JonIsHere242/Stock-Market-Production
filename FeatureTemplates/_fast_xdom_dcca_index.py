"""
DCCA coefficient (rho_DCCA) between stock and SPY at scale s=20, rolling 120-day window.

Zebende (2011) Detrended Cross-Correlation coefficient: the ratio of the detrended
cross-covariance to the geometric mean of the detrended auto-covariances, computed at
a given scale s (box size). Captures scale-dependent co-movement that a plain rolling
Pearson correlation blurs — especially the low-frequency coupling between a stock and
the market that vanishes under local detrending at a different scale.

Per-ticker proxy: exact DCCA formula applied to cumulative log-return series of the
stock and SPY, with box size s=20 (roughly monthly), rolled over a 120-day look-back.
This is a faithful implementation — DCCA is inherently a per-series (two-series) method
and does not require cross-sectional data.

FAST drop-in replacement of `_cand_xdom_dcca_index.py`. The per-bar O(window) Python
loop with per-box OLS is replaced by a fully vectorized formulation:
  * cumulative-log-return profiles computed once;
  * for every box-aligned offset (window_start mod s), the per-box linear-detrend is a
    fixed idempotent projection (I - P) on a length-s segment, so the per-box detrended
    (co)variance is a quadratic form seg^T (I - P) seg / s evaluated on ALL boxes at once
    via numpy.lib.stride_tricks.sliding_window_view;
  * boxes are then summed/averaged per rolling window with a strided gather.
Mathematically identical to the original (bit-exact up to float associativity of the
box accumulation), guarded for NaNs/zeros, no lookahead.
"""
from __future__ import annotations

import warnings
import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: load _indexes  (by-path import idiom — keep EXACTLY as original)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata  (column names + METADATA["name"] unchanged for drop-in parity)
# ---------------------------------------------------------------------------
METADATA = {
    "name": "xdom_dcca_index",
    "description": (
        "Rolling 120-day DCCA coefficient (rho_DCCA) between the stock's and SPY's "
        "cumulative log-return profiles at box scale s=20 (Zebende 2011). "
        "rho_DCCA is the detrended cross-covariance divided by the geometric mean of "
        "the detrended auto-covariances across non-overlapping boxes of size s. "
        "Captures scale-dependent co-movement that plain Pearson correlation blurs. "
        "Three columns produced: level (dcca_spy_120), a short-window z-score of the "
        "level to surface regime shifts (dcca_spy_z20), and a 20-day rate-of-change "
        "of the level (dcca_spy_roc20). Per-ticker implementation — faithful DCCA; "
        "not a cross-sectional proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom_dcca_index_dcca_spy_120",
        "xdom_dcca_index_dcca_spy_z20",
        "xdom_dcca_index_dcca_spy_roc20",
    ],
    "tags": ["cross-domain", "econophysics", "correlation", "market", "detrended", "dcca"],
    "version": "1.0.0",
    "author": "Zebende (2011) Detrended cross-correlation coefficient — econophysics / cross-domain-method spec",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _cumulative_log_returns(series: np.ndarray) -> np.ndarray:
    """Convert price series to cumulative log-return profile (integration step)."""
    lr = np.diff(np.log(np.where(series > 0, series, np.nan)))
    cum = np.concatenate([[0.0], np.nancumsum(lr)])
    return cum


def _box_detrend_matrix(s: int) -> np.ndarray:
    """
    Idempotent residual-maker M = I - X (X^T X)^{-1} X^T for a length-s linear OLS
    detrend on the fixed design x = [0,1,...,s-1] with intercept. Detrended residuals
    of a full (no-NaN) box are M @ seg, and the original code's mean-squared residual
    over a box equals (M @ seg)·(M @ seg)/s  ==  seg·(M @ seg)/s (M symmetric+idempotent).
    """
    x = np.arange(s, dtype=np.float64)
    X = np.column_stack([np.ones(s), x])          # (s, 2)
    XtX_inv = np.linalg.inv(X.T @ X)              # (2, 2)
    H = X @ XtX_inv @ X.T                          # hat matrix (s, s)
    M = np.eye(s) - H                              # residual maker
    return M


def _dcca_rolling(cum_x: np.ndarray, cum_y: np.ndarray, s: int, window: int,
                  min_bars: int) -> np.ndarray:
    """
    Vectorized rolling rho_DCCA. Returns an array of length len(cum_x), NaN where the
    rolling window has < min_bars finite bars (matching the original gating).

    For each bar t the window is [max(0,t-window+1) .. t]; boxes are non-overlapping
    runs of size s laid down from the window start, so n_boxes = win_len // s. The set
    of boxes for window ending at t is exactly {[start + k*s, start + (k+1)*s) }.
    """
    n = len(cum_x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < min_bars:
        return out

    M = _box_detrend_matrix(s)

    # Per-position detrended residuals of every length-s box that STARTS at index j,
    # for all j in [0, n-s]. Shape (n-s+1, s).
    win_x = np.lib.stride_tricks.sliding_window_view(cum_x, s)      # (n-s+1, s)
    win_y = np.lib.stride_tricks.sliding_window_view(cum_y, s)      # (n-s+1, s)
    rx = win_x @ M.T                                                # residuals, (n-s+1, s)
    ry = win_y @ M.T

    # Per-box detrended (co)variances = mean over the box of residual products.
    # finite only where the whole box is finite (matches mask.sum()>=3 -> all-finite here).
    finite_box = np.isfinite(win_x).all(axis=1) & np.isfinite(win_y).all(axis=1)
    v_xx = np.where(finite_box, np.nanmean(rx * rx, axis=1), np.nan)
    v_yy = np.where(finite_box, np.nanmean(ry * ry, axis=1), np.nan)
    v_xy = np.where(finite_box, np.nanmean(rx * ry, axis=1), np.nan)

    # For each bar t, identify its window and accumulate its boxes.
    for t in range(min_bars - 1, n):
        start = t if t - window + 1 < 0 else t - window + 1
        win_len = t - start + 1
        # original finite-count gate on the raw window of the cumulative profiles
        if win_len < min_bars:
            continue
        seg_x = cum_x[start: t + 1]
        seg_y = cum_y[start: t + 1]
        if (np.isfinite(seg_x).sum() < min_bars) or (np.isfinite(seg_y).sum() < min_bars):
            continue

        n_boxes = win_len // s
        if n_boxes < 2:
            continue

        # box start positions within the global arrays: start, start+s, ..., start+(n_boxes-1)*s
        box_starts = start + np.arange(n_boxes) * s
        bx = v_xx[box_starts]
        by = v_yy[box_starts]
        bxy = v_xy[box_starts]

        valid = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bxy)
        valid_boxes = int(valid.sum())
        if valid_boxes < 2:
            continue

        f2_xx = bx[valid].sum() / valid_boxes
        f2_yy = by[valid].sum() / valid_boxes
        f2_xy = bxy[valid].sum() / valid_boxes

        denom = np.sqrt(f2_xx * f2_yy)
        if (not np.isfinite(denom)) or denom <= 0:
            continue

        out[t] = float(np.clip(f2_xy / denom, -1.0, 1.0))

    return out


# ---------------------------------------------------------------------------
# Public compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling DCCA coefficient between stock and SPY (fast, drop-in).

    Parameters
    ----------
    df : pd.DataFrame
        Single-stock OHLCV, ascending by Date.

    Returns
    -------
    pd.DataFrame
        df with three new columns added.
    """
    BOX_SIZE = 20        # scale s (roughly one month of trading days)
    WINDOW = 120         # rolling look-back in trading days
    Z_WINDOW = 20        # z-score window for the dynamic variant
    ROC_WINDOW = 20      # rate-of-change window

    n = len(df)

    # ------------------------------------------------------------------
    # Fetch SPY close series and align to df dates
    # ------------------------------------------------------------------
    try:
        spy_series = _indexes.index_close("SPY")  # DatetimeIndex → float
    except Exception:
        spy_series = None

    if spy_series is None or len(spy_series) == 0:
        # Degrade gracefully: all NaN
        df["xdom_dcca_index_dcca_spy_120"] = np.nan
        df["xdom_dcca_index_dcca_spy_z20"] = np.nan
        df["xdom_dcca_index_dcca_spy_roc20"] = np.nan
        return df

    # Align SPY to df dates via merge_asof (backward — no lookahead)
    df_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df,
        on="Date",
        direction="backward",
    )
    # Re-align to original df order
    merged = merged.set_index(df_dates.index)
    spy_close_aligned = merged["spy_close"].values.astype(np.float64)

    stock_close = df["Close"].values.astype(np.float64)

    # ------------------------------------------------------------------
    # Cumulative log-return profiles (length n)
    # ------------------------------------------------------------------
    cum_stock = _cumulative_log_returns(stock_close)
    cum_spy = _cumulative_log_returns(spy_close_aligned)

    # Minimum bars needed: at least 2 full boxes of BOX_SIZE → 2*BOX_SIZE
    min_bars = max(2 * BOX_SIZE, BOX_SIZE + 1)

    # ------------------------------------------------------------------
    # Vectorized rolling DCCA
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcca_arr = _dcca_rolling(cum_stock, cum_spy, BOX_SIZE, WINDOW, min_bars)

    # ------------------------------------------------------------------
    # Produce derived columns (identical to original)
    # ------------------------------------------------------------------
    dcca_s = pd.Series(dcca_arr, index=df.index)

    roll_mean = dcca_s.rolling(Z_WINDOW, min_periods=max(2, Z_WINDOW // 2)).mean()
    roll_std = dcca_s.rolling(Z_WINDOW, min_periods=max(2, Z_WINDOW // 2)).std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dcca_z = (dcca_s - roll_mean) / roll_std.replace(0, np.nan)

    dcca_roc = dcca_s.diff(ROC_WINDOW)

    df["xdom_dcca_index_dcca_spy_120"] = dcca_s.values
    df["xdom_dcca_index_dcca_spy_z20"] = dcca_z.values
    df["xdom_dcca_index_dcca_spy_roc20"] = dcca_roc.values

    return df
