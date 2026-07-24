"""
Huber-robust beta gap vs OLS beta on SPY over 120-day rolling window.
robust_beta - ols_beta captures the influence of outlier days on the beta estimate.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by path, no package import)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120",
    "description": (
        "Huber-robust beta minus OLS beta vs SPY over a 120-day rolling window. "
        "OLS beta is computed from daily log-returns. Robust beta is obtained via 3 "
        "IRLS (iteratively reweighted least squares) passes using Huber weights "
        "(tuning constant k=1.345*MAD of residuals, guarded MAD>1e-9). "
        "The gap (robust_beta - ols_beta) measures how much outlier sessions distort "
        "the naive beta estimate -- a large negative gap means the stock's measured "
        "co-movement with SPY is inflated by a few extreme co-move days; a large "
        "positive gap means outliers suppress the true systematic exposure. "
        "Also emits the robust beta level and its 20-day rolling z-score. "
        "Per-ticker proxy (no cross-sectional rank required)."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_gap",
        "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_robust",
        "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_zscore",
    ],
    "tags": ["beta", "risk", "robust", "huber", "spy", "outlier"],
    "version": "1.0",
    "author": "feature-factory ff06290014i",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 120          # rolling window in bars
_HUBER_K_MULT = 1.345  # Huber tuning constant multiplier on MAD
_IRLS_PASSES = 3       # number of IRLS reweighting iterations
_MIN_OBS = 30          # minimum valid observations before computing
_ZSCORE_WIN = 20       # window for rolling z-score of robust beta

# ---------------------------------------------------------------------------
# Core: Huber-IRLS beta for a (x, y) pair of 1D arrays
# ---------------------------------------------------------------------------

def _huber_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Returns (ols_beta, robust_beta) for arrays x (market returns) and y (stock returns).
    Both are 1D, same length, no NaN.
    Uses simple regression through origin approximation:
        beta = cov(x,y) / var(x)  (demeaned, OLS)
    Then IRLS with Huber weights.
    Returns (nan, nan) if degenerate.
    """
    n = len(x)
    if n < _MIN_OBS:
        return np.nan, np.nan

    # Demean
    xm = x - x.mean()
    ym = y - y.mean()

    var_x = np.dot(xm, xm)
    if var_x < 1e-12:
        return np.nan, np.nan

    ols_beta = np.dot(xm, ym) / var_x

    # IRLS: start from OLS, apply Huber weights iteratively
    beta_r = ols_beta
    for _ in range(_IRLS_PASSES):
        resid = ym - beta_r * xm
        mad = np.median(np.abs(resid - np.median(resid)))
        if mad < 1e-9:
            # Residuals essentially zero -> robust == OLS
            break
        k = _HUBER_K_MULT * mad
        # Huber weights: 1 if |resid| <= k, else k/|resid|
        abs_r = np.abs(resid)
        w = np.where(abs_r <= k, 1.0, k / (abs_r + 1e-14))
        # Weighted least squares step
        wxm = w * xm
        denom = np.dot(wxm, xm)
        if denom < 1e-12:
            break
        beta_r = np.dot(wxm, ym) / denom

    return ols_beta, beta_r


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    col_gap = "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_gap"
    col_rob = "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_robust"
    col_z   = "ff06290014i_beta_risk_robust_beta_gap_huber_spy_120_zscore"

    # Initialise outputs to NaN
    df[col_gap] = np.nan
    df[col_rob] = np.nan
    df[col_z]   = np.nan

    if len(df) < _MIN_OBS + 1:
        return df

    # --- Fetch SPY closes and align ---
    try:
        spy_series = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_series is None or spy_series.empty:
        return df

    # Build per-bar stock log-returns
    close = df["Close"].values.astype(float)
    if len(close) < 2:
        return df

    stock_ret = np.empty(len(close))
    stock_ret[0] = np.nan
    stock_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    # Align SPY returns to the stock's Date index
    dates = pd.to_datetime(df["Date"])

    spy_df = spy_series.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    stock_df = pd.DataFrame({"Date": dates, "stock_ret": stock_ret}).reset_index(drop=True)
    merged = pd.merge_asof(
        stock_df.sort_values("Date"),
        spy_df,
        on="Date",
        direction="backward",
    )
    # Restore original order
    merged = merged.set_index(stock_df.sort_values("Date").index).reindex(stock_df.index)

    spy_close_aligned = merged["spy_close"].values.astype(float)

    # Compute SPY log-returns in aligned space
    spy_ret = np.empty(len(spy_close_aligned))
    spy_ret[0] = np.nan
    spy_ret[1:] = np.log(
        spy_close_aligned[1:] / np.where(spy_close_aligned[:-1] > 0, spy_close_aligned[:-1], np.nan)
    )

    # Rolling window computation
    n = len(df)
    gap_arr  = np.full(n, np.nan)
    rob_arr  = np.full(n, np.nan)

    for end in range(_WINDOW - 1, n):
        start = end - _WINDOW + 1
        x_win = spy_ret[start: end + 1]
        y_win = stock_ret[start: end + 1]

        # Drop NaN pairs
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if mask.sum() < _MIN_OBS:
            continue

        ols_b, rob_b = _huber_beta(x_win[mask], y_win[mask])
        if np.isfinite(ols_b) and np.isfinite(rob_b):
            gap_arr[end]  = rob_b - ols_b
            rob_arr[end]  = rob_b

    df[col_gap] = gap_arr
    df[col_rob] = rob_arr

    # Rolling z-score of robust beta (20-bar)
    rob_s = pd.Series(rob_arr)
    roll_mean = rob_s.rolling(_ZSCORE_WIN, min_periods=max(2, _ZSCORE_WIN // 2)).mean()
    roll_std  = rob_s.rolling(_ZSCORE_WIN, min_periods=max(2, _ZSCORE_WIN // 2)).std()
    z = (rob_s - roll_mean) / roll_std.replace(0, np.nan)
    df[col_z] = z.values

    return df
