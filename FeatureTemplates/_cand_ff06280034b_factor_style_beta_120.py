"""
Factor style beta (120-day rolling): market beta, growth-vs-small factor beta, R^2.
Two-factor OLS of stock return on (SPY return, QQQ-IWM return) over a 120-day window.
Per-ticker proxy for factor-style exposure; causal/no-lookahead.
"""
from __future__ import annotations
import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper by path
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034b_factor_style_beta_120",
    "description": (
        "120-day rolling 2-factor OLS: stock daily return ~ beta_mkt * SPY_ret "
        "+ beta_style * (QQQ_ret - IWM_ret) + eps. "
        "Produces market beta, style (growth-minus-small) beta, and R^2. "
        "Per-ticker proxy capturing factor-style exposure shifts over time."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_factor_style_beta_120_mkt_beta",
        "ff06280034b_factor_style_beta_120_style_beta",
        "ff06280034b_factor_style_beta_120_r2",
    ],
    "tags": ["factor", "beta", "style", "rolling", "ols"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}

_WINDOW = 120
_COL_MKT = "ff06280034b_factor_style_beta_120_mkt_beta"
_COL_STYLE = "ff06280034b_factor_style_beta_120_style_beta"
_COL_R2 = "ff06280034b_factor_style_beta_120_r2"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN (required on all code paths)
    df[_COL_MKT] = np.nan
    df[_COL_STYLE] = np.nan
    df[_COL_R2] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch index close series
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        qqq_close = _indexes.index_close("QQQ")
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df

    if spy_close is None or qqq_close is None or iwm_close is None:
        return df

    # Convert index to DatetimeIndex for merge_asof
    def _to_frame(series, col_name):
        s = series.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        frame = s.reset_index()
        frame.columns = ["Date", col_name]
        return frame.sort_values("Date")

    spy_df = _to_frame(spy_close, "_spy_c")
    qqq_df = _to_frame(qqq_close, "_qqq_c")
    iwm_df = _to_frame(iwm_close, "_iwm_c")

    # Merge index prices onto stock dates (backward / no lookahead)
    work = df[["Date"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = work.sort_values("Date").reset_index(drop=False)  # preserve original index

    work = pd.merge_asof(work, spy_df, on="Date", direction="backward")
    work = pd.merge_asof(work, qqq_df, on="Date", direction="backward")
    work = pd.merge_asof(work, iwm_df, on="Date", direction="backward")

    # Daily log returns for index series
    work["_spy_r"] = np.log(work["_spy_c"] / work["_spy_c"].shift(1))
    work["_qqq_r"] = np.log(work["_qqq_c"] / work["_qqq_c"].shift(1))
    work["_iwm_r"] = np.log(work["_iwm_c"] / work["_iwm_c"].shift(1))

    # Growth-minus-small factor return
    work["_factor_r"] = work["_qqq_r"] - work["_iwm_r"]

    # Stock daily log return (aligned to work row order, which matches df row order
    # because we only sorted but preserved index via reset_index(drop=False))
    stock_close = df["Close"].values.astype(float)
    stock_ret = np.empty(len(stock_close))
    stock_ret[0] = np.nan
    stock_ret[1:] = np.log(stock_close[1:] / stock_close[:-1])
    # Align to work (same order since we preserved original index order)
    work["_stock_r"] = stock_ret

    # ------------------------------------------------------------------
    # Rolling 2-factor OLS via vectorised numpy sliding window
    # We need: for each window end t, solve
    #   [1, X1, X2] @ [alpha, b_mkt, b_style] = y
    # using the closed-form formula for OLS with design matrix.
    # We use numpy sliding_window_view for efficiency.
    # ------------------------------------------------------------------
    n = len(work)
    spy_r = work["_spy_r"].values.astype(float)
    fac_r = work["_factor_r"].values.astype(float)
    stk_r = work["_stock_r"].values.astype(float)

    mkt_beta_arr = np.full(n, np.nan)
    style_beta_arr = np.full(n, np.nan)
    r2_arr = np.full(n, np.nan)

    W = _WINDOW
    # Slide over windows; window [i, i+W) ends at index i+W-1
    for start in range(n - W + 1):
        end = start + W  # exclusive
        y = stk_r[start:end]
        x1 = spy_r[start:end]
        x2 = fac_r[start:end]

        # Drop rows with any NaN
        mask = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2))
        if mask.sum() < 10:  # not enough observations
            continue

        ym = y[mask]
        x1m = x1[mask]
        x2m = x2[mask]
        k = mask.sum()

        # Design matrix [1, x1, x2]
        ones = np.ones(k)
        X = np.column_stack([ones, x1m, x2m])

        # OLS: (X'X)^{-1} X'y via closed form
        XtX = X.T @ X
        Xty = X.T @ ym

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                coeffs = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                continue

        alpha, b_mkt, b_style = coeffs

        # R^2
        y_hat = X @ coeffs
        ss_res = np.sum((ym - y_hat) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan

        idx = end - 1  # result assigned to last bar of window
        mkt_beta_arr[idx] = b_mkt
        style_beta_arr[idx] = b_style
        r2_arr[idx] = r2

    # Map results back to original df order via the preserved index column
    orig_idx = work["index"].values  # original df integer index positions

    mkt_beta_out = np.full(len(df), np.nan)
    style_beta_out = np.full(len(df), np.nan)
    r2_out = np.full(len(df), np.nan)

    for work_pos in range(n):
        oi = orig_idx[work_pos]
        mkt_beta_out[oi] = mkt_beta_arr[work_pos]
        style_beta_out[oi] = style_beta_arr[work_pos]
        r2_out[oi] = r2_arr[work_pos]

    df[_COL_MKT] = mkt_beta_out
    df[_COL_STYLE] = style_beta_out
    df[_COL_R2] = r2_out

    # Guard: replace inf/-inf with NaN
    for col in [_COL_MKT, _COL_STYLE, _COL_R2]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
