from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load index helper by file path --
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290014i_factor_beta_to_squared_dispersion_4index_120",
    "description": (
        "Gamma (convex) exposure to 4-index dispersion stress. "
        "f = cross-sectional std of daily returns across [SPY, QQQ, IWM, DIA]. "
        "Feature = OLS slope of ticker daily return regressed on (f^2 - mean(f^2)) "
        "over a 120-day rolling window. Captures a ticker's nonlinear sensitivity to "
        "inter-index disagreement (dispersion-of-dispersion stress). "
        "Complementary variant: 20-day short-window slope for recency dynamics. "
        "Per-ticker proxy; causal; no cross-sectional data needed at compute time."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta120",
        "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta20",
        "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta_chg",
    ],
    "tags": ["factor", "dispersion", "nonlinear", "beta", "index", "convexity"],
    "version": "1.0.0",
    "author": "feature-factory ff06290014i",
}

_SYMS = ["SPY", "QQQ", "IWM", "DIA"]
_WIN_LONG = 120
_WIN_SHORT = 20


def _ols_slope_rolling(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OLS slope of y on x over `window` bars.
    Returns an array of the same length as x, with leading NaNs.
    Vectorised via cumulative sums (O(n)).
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    # Compute rolling sums using cumsum
    # For window w ending at i: sum_{t=i-w+1}^{i}
    cx = np.cumsum(x)
    cy = np.cumsum(y)
    cxx = np.cumsum(x * x)
    cxy = np.cumsum(x * y)

    # Vectorised for all valid windows
    for i in range(window - 1, n):
        if i >= window:
            sx = cx[i] - cx[i - window]
            sy = cy[i] - cy[i - window]
            sxx = cxx[i] - cxx[i - window]
            sxy = cxy[i] - cxy[i - window]
        else:
            sx = cx[i]
            sy = cy[i]
            sxx = cxx[i]
            sxy = cxy[i]
        var_x = sxx - sx * sx / window
        if var_x < 1e-12:
            out[i] = 0.0
        else:
            out[i] = (sxy - sx * sy / window) / var_x
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_b120 = "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta120"
    col_b20 = "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta20"
    col_chg = "ff06290014i_factor_beta_to_squared_dispersion_4index_120_beta_chg"

    df[col_b120] = np.nan
    df[col_b20] = np.nan
    df[col_chg] = np.nan

    if len(df) < _WIN_LONG + 2:
        return df

    # -- fetch index closes --
    index_rets = {}
    for sym in _SYMS:
        try:
            s = _indexes.index_close(sym)
            if s is None or len(s) == 0:
                continue
            index_rets[sym] = s.pct_change()
        except Exception:
            continue

    if len(index_rets) < 2:
        return df

    # Build a DataFrame of index returns aligned to df dates
    dates = pd.to_datetime(df["Date"])
    idx_df = pd.DataFrame(index_rets)
    idx_df.index = pd.to_datetime(idx_df.index)
    idx_df = idx_df.sort_index()

    # Align to ticker dates via merge_asof (backward)
    df_dates = pd.DataFrame({"Date": dates.values}).sort_values("Date")
    idx_df_reset = idx_df.reset_index().rename(columns={"index": "Date", "Date": "Date"})
    # reset_index gives us the DatetimeIndex as a column
    idx_df2 = idx_df.copy()
    idx_df2.index.name = "Date"
    idx_df2 = idx_df2.reset_index()
    idx_df2["Date"] = pd.to_datetime(idx_df2["Date"])

    merged = pd.merge_asof(
        df_dates.assign(_orig_order=np.arange(len(df_dates))),
        idx_df2,
        on="Date",
        direction="backward",
    )
    # restore original row order
    merged = merged.sort_values("_orig_order").reset_index(drop=True)

    avail_cols = [c for c in _SYMS if c in merged.columns]
    if len(avail_cols) < 2:
        return df

    # f = std across available index returns (per day), then squared
    mat = merged[avail_cols].values.astype(float)  # shape (n, k)
    # std across columns (axis=1); ddof=1
    with np.errstate(invalid="ignore"):
        f = np.nanstd(mat, axis=1, ddof=1)  # shape (n,)

    f2 = f ** 2

    # Demean f^2 using expanding mean to stay causal
    # We need a causal demeaning: use the rolling mean over the same window
    # For the regression predictor use global (expanding) mean — but to avoid
    # lookahead we subtract the *in-window* mean during each OLS window.
    # The OLS slope with demeaned x is identical to non-demeaned slope (centering
    # doesn't change the slope), so we can just use f^2 directly as x.

    # Ticker daily returns
    close = df["Close"].values.astype(float)
    r = np.empty_like(close)
    r[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = np.where(close[:-1] != 0, (close[1:] - close[:-1]) / close[:-1], np.nan)

    # Replace any inf
    r = np.where(np.isfinite(r), r, np.nan)
    f2 = np.where(np.isfinite(f2), f2, np.nan)

    # Fill NaN in f2 / r with 0 only inside the rolling window computation
    # We'll mask NaN windows in the OLS helper instead
    # For speed use a vectorised O(n) cumsum approach (handles NaN by masking)
    # Build NaN-safe versions: replace NaN with 0 and track validity
    valid = np.isfinite(r) & np.isfinite(f2)
    r_c = np.where(valid, r, 0.0)
    f2_c = np.where(valid, f2, 0.0)
    cnt = valid.astype(float)

    n = len(r_c)
    cx = np.cumsum(f2_c)
    cy = np.cumsum(r_c)
    cxx = np.cumsum(f2_c * f2_c)
    cxy = np.cumsum(f2_c * r_c)
    cc = np.cumsum(cnt)

    def _slope_at(i, window):
        if i < window - 1:
            return np.nan
        if i >= window:
            sx = cx[i] - cx[i - window]
            sy = cy[i] - cy[i - window]
            sxx = cxx[i] - cxx[i - window]
            sxy = cxy[i] - cxy[i - window]
            nc = cc[i] - cc[i - window]
        else:
            sx = cx[i]; sy = cy[i]; sxx = cxx[i]; sxy = cxy[i]; nc = cc[i]
        if nc < window * 0.5:
            return np.nan
        var_x = sxx - sx * sx / nc
        if var_x < 1e-12:
            return 0.0
        return (sxy - sx * sy / nc) / var_x

    # Vectorise via numpy operations over all valid i
    # Long window
    b120 = np.full(n, np.nan)
    b20 = np.full(n, np.nan)

    # Long window (120)
    irange = np.arange(_WIN_LONG - 1, n)
    lo = irange - _WIN_LONG  # = irange - 120; for i < 120 use 0 (handled below)
    lo_clip = np.maximum(lo, -1)  # -1 means use from 0

    for idx_i in irange:
        b120[idx_i] = _slope_at(idx_i, _WIN_LONG)

    # Short window (20)
    for idx_i in range(_WIN_SHORT - 1, n):
        b20[idx_i] = _slope_at(idx_i, _WIN_SHORT)

    # Delta: long-window slope minus lagged by 20 bars
    b_chg = np.full(n, np.nan)
    lag = 20
    if n > lag:
        b_chg[lag:] = np.where(
            np.isfinite(b120[lag:]) & np.isfinite(b120[:-lag]),
            b120[lag:] - b120[:-lag],
            np.nan,
        )

    df[col_b120] = b120
    df[col_b20] = b20
    df[col_chg] = b_chg

    return df
