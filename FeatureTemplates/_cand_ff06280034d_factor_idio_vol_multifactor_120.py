"""
ff06280034d_factor_idio_vol_multifactor_120
Rolling 120-day 3-factor OLS (SPY, QQQ, IWM) idiosyncratic volatility decomposition.
Produces: residual std (clean idio vol), systematic R², and 20d z-score of idio vol.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper by path
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034d_factor_idio_vol_multifactor_120",
    "description": (
        "Per-ticker rolling 120-day 3-factor OLS on daily returns (SPY, QQQ, IWM). "
        "Produces: idio_vol = std of OLS residuals (idiosyncratic volatility), "
        "r2 = R-squared (systematic share of variance), "
        "idio_vol_z20 = 20-day rolling z-score of idio_vol (vol-of-vol regime). "
        "Uses merge_asof backward alignment on index returns; degrades to NaN if index data missing. "
        "This is a per-ticker proxy -- cross-sectional rank comparison is done downstream."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034d_factor_idio_vol_multifactor_120_idio_vol",
        "ff06280034d_factor_idio_vol_multifactor_120_r2",
        "ff06280034d_factor_idio_vol_multifactor_120_idio_vol_z20",
    ],
    "tags": ["factor", "idiosyncratic", "volatility", "multifactor", "ols", "regime"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034d",
}

_WINDOW = 120
_Z_WINDOW = 20
_PREFIX = "ff06280034d_factor_idio_vol_multifactor_120"
_COL_IDIO = f"{_PREFIX}_idio_vol"
_COL_R2   = f"{_PREFIX}_r2"
_COL_Z20  = f"{_PREFIX}_idio_vol_z20"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN up front (required on every code path)
    df[_COL_IDIO] = np.nan
    df[_COL_R2]   = np.nan
    df[_COL_Z20]  = np.nan

    n = len(df)
    if n < _WINDOW + 1:
        return df

    # ------------------------------------------------------------------
    # Build stock daily log returns
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    # log returns; first element is NaN
    stock_ret = np.empty(n, dtype=np.float64)
    stock_ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret[1:] = np.log(close[1:] / close[:-1])
    # guard zeros / negatives
    stock_ret = np.where(np.isfinite(stock_ret), stock_ret, np.nan)

    # ------------------------------------------------------------------
    # Fetch index returns (SPY, QQQ, IWM) and align to df dates
    # ------------------------------------------------------------------
    def _get_idx_ret(symbol: str) -> np.ndarray:
        """Return log-return series aligned to df index, or all-NaN."""
        try:
            idx_close = _indexes.index_close(symbol)  # pd.Series, DatetimeIndex
            if idx_close is None or len(idx_close) == 0:
                return np.full(n, np.nan)
            idx_df = idx_close.rename("_c").reset_index()
            idx_df.columns = ["Date", "_c"]
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            stock_dates = pd.to_datetime(df["Date"].values)
            merged = pd.merge_asof(
                pd.DataFrame({"Date": stock_dates}),
                idx_df.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            c = merged["_c"].values.astype(np.float64)
            r = np.empty(n, dtype=np.float64)
            r[0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r[1:] = np.log(c[1:] / c[:-1])
            r = np.where(np.isfinite(r), r, np.nan)
            return r
        except Exception:
            return np.full(n, np.nan)

    spy_ret = _get_idx_ret("SPY")
    qqq_ret = _get_idx_ret("QQQ")
    iwm_ret = _get_idx_ret("IWM")

    # ------------------------------------------------------------------
    # Rolling 120-day OLS: y = a + b1*SPY + b2*QQQ + b3*IWM + e
    # We use a hand-rolled numpy stride approach for speed.
    # For each window ending at bar t, solve via (X'X)^-1 X'y.
    # ------------------------------------------------------------------
    idio_vol_arr = np.full(n, np.nan)
    r2_arr       = np.full(n, np.nan)

    W = _WINDOW

    for end in range(W, n):
        start = end - W
        y   = stock_ret[start:end]
        x1  = spy_ret[start:end]
        x2  = qqq_ret[start:end]
        x3  = iwm_ret[start:end]

        # Build design matrix [1, SPY, QQQ, IWM]
        # Find rows where all values are finite
        mask = (
            np.isfinite(y) &
            np.isfinite(x1) &
            np.isfinite(x2) &
            np.isfinite(x3)
        )
        n_ok = mask.sum()
        if n_ok < 10:  # need minimum observations
            continue

        yv  = y[mask]
        Xv  = np.column_stack([
            np.ones(n_ok, dtype=np.float64),
            x1[mask],
            x2[mask],
            x3[mask],
        ])

        # OLS via normal equations: beta = (X'X)^-1 X'y
        try:
            XtX = Xv.T @ Xv
            Xty = Xv.T @ yv
            # Use lstsq for numerical stability
            beta, _, rank, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            if rank < 4:
                continue
            resid = yv - Xv @ beta
            ss_res = float(np.dot(resid, resid))
            yv_mean = float(yv.mean())
            ss_tot = float(np.dot(yv - yv_mean, yv - yv_mean))

            idio_vol_arr[end] = np.sqrt(ss_res / max(n_ok - 4, 1))
            if ss_tot > 0:
                r2_arr[end] = max(0.0, 1.0 - ss_res / ss_tot)
            else:
                r2_arr[end] = np.nan
        except Exception:
            continue

    df[_COL_IDIO] = idio_vol_arr
    df[_COL_R2]   = r2_arr

    # ------------------------------------------------------------------
    # 20-day rolling z-score of idio_vol
    # ------------------------------------------------------------------
    idio_s = pd.Series(idio_vol_arr)
    roll_mean = idio_s.rolling(_Z_WINDOW, min_periods=max(5, _Z_WINDOW // 2)).mean()
    roll_std  = idio_s.rolling(_Z_WINDOW, min_periods=max(5, _Z_WINDOW // 2)).std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = (idio_s - roll_mean) / roll_std.replace(0, np.nan)
    z = z.where(np.isfinite(z), other=np.nan)
    df[_COL_Z20] = z.values

    return df
