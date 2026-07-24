from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load _indexes helper by file path ---
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06280034c_factor_size_beta_120",
    "description": (
        "Rolling 120-day 2-factor OLS of stock returns on (SPY return, IWM-SPY size factor return). "
        "Produces factor beta (size tilt), market beta, and R^2. "
        "The size factor is IWM minus SPY daily return (small-minus-big). "
        "Per-ticker proxy: no cross-sectional ranking; betas estimated from 120-bar windows. "
        "Degrade to NaN when index data is unavailable or window is insufficient."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034c_factor_size_beta_120_size_beta",
        "ff06280034c_factor_size_beta_120_mkt_beta",
        "ff06280034c_factor_size_beta_120_r2",
    ],
    "tags": ["factor", "beta", "size", "rolling", "ols"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034c",
}

_WINDOW = 120
_MIN_OBS = 30  # minimum valid observations within window


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # initialise produced columns to NaN on every code path
    df["ff06280034c_factor_size_beta_120_size_beta"] = np.nan
    df["ff06280034c_factor_size_beta_120_mkt_beta"] = np.nan
    df["ff06280034c_factor_size_beta_120_r2"] = np.nan

    if len(df) < 2:
        return df

    # --- fetch index data ---
    try:
        spy_close = _indexes.index_close("SPY")
        iwm_close = _indexes.index_close("IWM")
    except Exception:
        return df

    if spy_close is None or iwm_close is None or spy_close.empty or iwm_close.empty:
        return df

    # build daily returns for SPY and IWM
    spy_ret = spy_close.pct_change()
    iwm_ret = iwm_close.pct_change()
    # size factor: IWM - SPY (small-minus-big)
    size_factor = iwm_ret - spy_ret

    # align to df dates via merge_asof (backward, lookahead-safe)
    dates = df["Date"].copy()
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)

    idx_df = pd.DataFrame({
        "Date": spy_ret.index,
        "_spy_ret": spy_ret.values,
        "_size_factor": size_factor.values,
    })
    idx_df = idx_df.dropna(subset=["_spy_ret", "_size_factor"])
    idx_df = idx_df.sort_values("Date")

    work = df[["Date", "Close"]].copy()
    work["_date_dt"] = dates
    work = work.sort_values("_date_dt").reset_index(drop=False)

    merged = pd.merge_asof(
        work,
        idx_df.rename(columns={"Date": "_date_dt"}),
        on="_date_dt",
        direction="backward",
    )

    # stock returns (per-ticker series, ascending by date)
    stock_ret = merged["Close"].pct_change().values
    mkt = merged["_spy_ret"].values
    smb = merged["_size_factor"].values

    n = len(merged)
    size_betas = np.full(n, np.nan)
    mkt_betas = np.full(n, np.nan)
    r2s = np.full(n, np.nan)

    # rolling OLS: for each bar t, use window [t-WINDOW+1 .. t]
    # vectorised via cumsum trick is complex for 2-factor; use a stride loop
    # over windows but keep it O(n * WINDOW) with numpy slice -- acceptable for n~700
    for t in range(_WINDOW - 1, n):
        y = stock_ret[t - _WINDOW + 1 : t + 1]
        x1 = mkt[t - _WINDOW + 1 : t + 1]
        x2 = smb[t - _WINDOW + 1 : t + 1]

        # mask valid (non-NaN) rows
        mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
        obs = mask.sum()
        if obs < _MIN_OBS:
            continue

        yv = y[mask]
        x1v = x1[mask]
        x2v = x2[mask]

        # OLS design matrix [1, x1, x2]
        X = np.column_stack([np.ones(obs), x1v, x2v])
        XtX = X.T @ X
        Xty = X.T @ yv

        # solve via pseudo-inverse to guard near-singular
        try:
            coeffs, _, _, _ = np.linalg.lstsq(XtX, Xty.reshape(-1, 1), rcond=None)
            b0, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        except Exception:
            continue

        # R^2
        y_hat = b0 + b1 * x1v + b2 * x2v
        ss_res = float(np.sum((yv - y_hat) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

        mkt_betas[t] = b1
        size_betas[t] = b2
        r2s[t] = r2

    # map results back to original df order via the saved original index
    orig_idx = merged["index"].values
    size_beta_out = np.full(len(df), np.nan)
    mkt_beta_out = np.full(len(df), np.nan)
    r2_out = np.full(len(df), np.nan)

    for pos, oi in enumerate(orig_idx):
        if 0 <= oi < len(df):
            size_beta_out[oi] = size_betas[pos]
            mkt_beta_out[oi] = mkt_betas[pos]
            r2_out[oi] = r2s[pos]

    df["ff06280034c_factor_size_beta_120_size_beta"] = size_beta_out
    df["ff06280034c_factor_size_beta_120_mkt_beta"] = mkt_beta_out
    df["ff06280034c_factor_size_beta_120_r2"] = r2_out

    return df
