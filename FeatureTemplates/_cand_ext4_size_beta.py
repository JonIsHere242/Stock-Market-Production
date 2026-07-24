"""
ext4_size_beta: Size-factor beta (IWM minus SPY)
Rolling 120-day 2-factor OLS beta of stock return on the SMB (small-minus-big) proxy
and SPY (market), plus the regression R^2.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper by file path
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_size_beta",
    "description": (
        "Rolling 120-day 2-factor OLS: stock daily return ~ alpha + beta_smb * smb_ret "
        "+ beta_mkt * spy_ret. smb_ret = IWM return - SPY return (small-minus-big proxy). "
        "Produces: size beta (beta_smb), market beta (beta_mkt), and regression R^2. "
        "Captures per-ticker size-factor exposure on a purely per-ticker basis (no cross-section)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_size_beta_smb",   # rolling 120d beta on size factor
        "ext4_size_beta_mkt",   # rolling 120d beta on SPY (market)
        "ext4_size_beta_r2",    # rolling 120d R^2
    ],
    "tags": ["factor", "size", "beta", "regression", "market"],
    "version": "1.0",
    "author": "Round-5 expansion (NEW: multi-index factor)",
}

_WINDOW = 120  # trading days


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        df["ext4_size_beta_smb"] = np.nan
        df["ext4_size_beta_mkt"] = np.nan
        df["ext4_size_beta_r2"] = np.nan
        return df

    # ------------------------------------------------------------------
    # 1. Fetch index series
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spy_close = _indexes.index_close("SPY")
        except Exception:
            spy_close = None
        try:
            iwm_close = _indexes.index_close("IWM")
        except Exception:
            iwm_close = None

    # Build a helper df for merging
    work = df[["Date"]].copy()
    work["_stock_close"] = df["Close"].values

    # ------------------------------------------------------------------
    # 2. Merge SPY and IWM via merge_asof (lookahead-safe, backward)
    # ------------------------------------------------------------------
    work = work.sort_values("Date").reset_index(drop=True)

    def _merge_index(ser, col_name):
        nonlocal work
        if ser is None or len(ser) == 0:
            work[col_name] = np.nan
            return
        idx_df = ser.reset_index()
        idx_df.columns = ["Date", col_name]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])
        work["Date"] = pd.to_datetime(work["Date"])
        work = pd.merge_asof(
            work.sort_values("Date"),
            idx_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )

    _merge_index(spy_close, "_spy_close")
    _merge_index(iwm_close, "_iwm_close")

    # ------------------------------------------------------------------
    # 3. Compute daily log returns (pct_change is fine here; no NaN fill)
    # ------------------------------------------------------------------
    stock_ret = work["_stock_close"].pct_change()
    spy_ret   = work["_spy_close"].pct_change()   if "_spy_close"  in work.columns else pd.Series(np.nan, index=work.index)
    iwm_ret   = work["_iwm_close"].pct_change()   if "_iwm_close"  in work.columns else pd.Series(np.nan, index=work.index)

    smb_ret = iwm_ret - spy_ret  # small-minus-big factor

    stock_ret_arr = stock_ret.values.astype(float)
    spy_ret_arr   = spy_ret.values.astype(float)
    smb_ret_arr   = smb_ret.values.astype(float)

    n = len(stock_ret_arr)
    beta_smb_out = np.full(n, np.nan)
    beta_mkt_out = np.full(n, np.nan)
    r2_out       = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 4. Rolling 120-day 2-factor OLS via numpy vectorised windows
    #    We use a sliding accumulation approach: for each end-point t,
    #    build the window and solve the normal equations (3×3 system).
    #    Using numpy.lib.stride_tricks for vectorised window extraction.
    # ------------------------------------------------------------------
    W = _WINDOW
    min_obs = max(10, W // 4)  # need at least min_obs valid rows in window

    # We iterate only over positions where t >= W-1  (first valid end index)
    for t in range(W - 1, n):
        y   = stock_ret_arr[t - W + 1: t + 1]
        x1  = smb_ret_arr[t - W + 1: t + 1]
        x2  = spy_ret_arr[t - W + 1: t + 1]

        # Drop rows where any series is NaN
        mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        if mask.sum() < min_obs:
            continue

        y_  = y[mask]
        x1_ = x1[mask]
        x2_ = x2[mask]

        # Design matrix: [1, x1, x2]
        ones = np.ones(len(y_))
        X = np.column_stack([ones, x1_, x2_])

        # OLS via normal equations: beta = (X'X)^{-1} X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y_
            coeffs, res, rank, sv = np.linalg.lstsq(XtX, Xty, rcond=None)
        except Exception:
            continue

        alpha_, b_smb, b_mkt = coeffs

        # R^2
        y_hat = X @ coeffs
        ss_res = np.sum((y_ - y_hat) ** 2)
        ss_tot = np.sum((y_ - y_.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        beta_smb_out[t] = b_smb
        beta_mkt_out[t] = b_mkt
        r2_out[t]       = r2

    # ------------------------------------------------------------------
    # 5. Align back to original df index order
    # ------------------------------------------------------------------
    # work was sorted by Date; df may not be — re-align by positional map
    orig_order = df.sort_values("Date").index  # ascending Date positions in df

    smb_series = pd.Series(beta_smb_out, index=orig_order)
    mkt_series = pd.Series(beta_mkt_out, index=orig_order)
    r2_series  = pd.Series(r2_out,       index=orig_order)

    df["ext4_size_beta_smb"] = smb_series.reindex(df.index)
    df["ext4_size_beta_mkt"] = mkt_series.reindex(df.index)
    df["ext4_size_beta_r2"]  = r2_series.reindex(df.index)

    return df
