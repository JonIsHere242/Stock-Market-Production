"""
Crash beta vs rally beta (extreme-day beta asymmetry).

Estimates beta vs SPY over a trailing 250-day window restricted to the
worst-decile market days (crash beta) and separately the best-decile market
days (rally beta). Produces crash beta, rally beta, and their difference
(tail beta asymmetry).

This is orthogonal to xdom2_downside_beta which conditions on any negative /
positive SPY return day. Here we condition only on the extreme 10% tails of
the SPY return distribution, giving a sharper asymmetry signal.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY daily close -> returns)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_tail_beta_asymmetry",
    "description": (
        "Per-ticker tail-beta asymmetry: OLS beta vs SPY estimated over only the "
        "worst-decile (crash) and best-decile (rally) SPY return days within a "
        "trailing 250-day window. Produces crash_beta, rally_beta, and their "
        "difference (asym). Stronger conditioning than the parent xdom2_downside_beta "
        "which splits on sign alone. Per-ticker proxy -- cross-sectional ranking "
        "applied at inference. Uses _indexes SPY; degrades to NaN if unavailable."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_tail_beta_asymmetry_crash",   # beta on worst-decile SPY days
        "ext_tail_beta_asymmetry_rally",   # beta on best-decile SPY days
        "ext_tail_beta_asymmetry_asym",    # crash_beta - rally_beta
    ],
    "tags": ["beta", "tail", "asymmetry", "market", "downside", "crash"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner xdom2_downside_beta; "
        "method derived from tail-conditioning literature on downside/crash risk "
        "beta asymmetry."
    ),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 250        # trailing days
_TAIL_FRAC = 0.10   # decile threshold


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute crash/rally tail beta asymmetry features."""
    n = len(df)

    # Pre-allocate output columns as NaN
    df["ext_tail_beta_asymmetry_crash"] = np.nan
    df["ext_tail_beta_asymmetry_rally"] = np.nan
    df["ext_tail_beta_asymmetry_asym"] = np.nan

    if n < 30:
        return df

    # -----------------------------------------------------------------------
    # Fetch SPY close series and align to df dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    # Build a small DataFrame for merge_asof
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    df_work = df[["Date"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = pd.merge_asof(
        df_work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original order
    df_work = df_work.set_index(df.index if df.index.name != "Date" else None)

    spy_aligned = df_work["spy_close"].values  # aligned to df rows

    # -----------------------------------------------------------------------
    # Compute daily log returns (stock and SPY)
    # -----------------------------------------------------------------------
    close_vals = df["Close"].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stock_ret = np.empty(n)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(close_vals[1:] / close_vals[:-1])

        spy_ret = np.empty(n)
        spy_ret[0] = np.nan
        spy_ret[1:] = np.log(
            np.where(
                spy_aligned[:-1] > 0,
                spy_aligned[1:] / spy_aligned[:-1],
                np.nan,
            )
        )

    # -----------------------------------------------------------------------
    # Rolling tail-conditioned OLS beta (vectorised with sliding windows)
    # -----------------------------------------------------------------------
    crash_beta = np.full(n, np.nan)
    rally_beta = np.full(n, np.nan)

    # We need at least WINDOW points; iterate from WINDOW onward
    # Use numpy sliding_window_view for efficiency
    if n < _WINDOW + 1:
        return df

    try:
        from numpy.lib.stride_tricks import sliding_window_view

        # Build windows over rows [1..n-1] (first row is NaN return)
        # We want windows of length _WINDOW ending at row t (inclusive)
        # t ranges from _WINDOW to n-1
        s_ret = stock_ret  # shape (n,)
        m_ret = spy_ret    # shape (n,)

        # sliding_window_view over the full arrays
        s_win = sliding_window_view(s_ret, _WINDOW)  # shape (n-_WINDOW+1, _WINDOW)
        m_win = sliding_window_view(m_ret, _WINDOW)  # shape (n-_WINDOW+1, _WINDOW)

        # s_win[i] corresponds to rows [i .. i+_WINDOW-1]
        # We want t = i+_WINDOW-1, so i = t - _WINDOW + 1
        # t starts at _WINDOW-1 (0-indexed) = _WINDOW in 1-indexed

        for i in range(len(s_win)):
            t = i + _WINDOW - 1  # row index in df
            sw = s_win[i]        # shape (_WINDOW,)
            mw = m_win[i]        # shape (_WINDOW,)

            valid = np.isfinite(sw) & np.isfinite(mw)
            if valid.sum() < 20:
                continue

            sw_v = sw[valid]
            mw_v = mw[valid]

            # Threshold for tail: decile of SPY returns within this window
            k = max(1, int(np.floor(len(mw_v) * _TAIL_FRAC)))
            sorted_m = np.sort(mw_v)
            crash_thresh = sorted_m[k - 1]    # <= this -> crash day
            rally_thresh = sorted_m[-(k)]     # >= this -> rally day

            # --- Crash beta ---
            crash_mask = mw_v <= crash_thresh
            if crash_mask.sum() >= 5:
                mx_c = mw_v[crash_mask]
                sx_c = sw_v[crash_mask]
                mx_c_dm = mx_c - mx_c.mean()
                denom = np.dot(mx_c_dm, mx_c_dm)
                if denom > 1e-12:
                    crash_beta[t] = np.dot(mx_c_dm, sx_c - sx_c.mean()) / denom

            # --- Rally beta ---
            rally_mask = mw_v >= rally_thresh
            if rally_mask.sum() >= 5:
                mx_r = mw_v[rally_mask]
                sx_r = sw_v[rally_mask]
                mx_r_dm = mx_r - mx_r.mean()
                denom = np.dot(mx_r_dm, mx_r_dm)
                if denom > 1e-12:
                    rally_beta[t] = np.dot(mx_r_dm, sx_r - sx_r.mean()) / denom

    except Exception:
        return df

    # -----------------------------------------------------------------------
    # Write outputs -- guard inf
    # -----------------------------------------------------------------------
    crash_beta = np.where(np.isfinite(crash_beta), crash_beta, np.nan)
    rally_beta = np.where(np.isfinite(rally_beta), rally_beta, np.nan)
    asym = crash_beta - rally_beta
    asym = np.where(np.isfinite(asym), asym, np.nan)

    df["ext_tail_beta_asymmetry_crash"] = crash_beta
    df["ext_tail_beta_asymmetry_rally"] = rally_beta
    df["ext_tail_beta_asymmetry_asym"] = asym

    return df
