"""
Feature block: ff06282316b_factor_beta_dispersion_4index_120
Vein: factor
Batch: 06282316b

Computes the trailing 120-day OLS beta of the ticker to each of SPY, QQQ, IWM,
and DIA separately, then takes the standard deviation (dispersion) of those four
betas. Low dispersion = uniformly market-exposed (broad beta); high dispersion =
concentrated exposure in one index style (e.g. small-cap vs large-cap vs tech).

Per-ticker proxy -- this is inherently per-stock (one stock vs 4 indexes) so no
cross-sectional approximation is needed. Betas are computed via vectorised rolling
OLS using pre-computed covariance and variance over the aligned return series.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load _indexes helper by file path --
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282316b_factor_beta_dispersion_4index_120",
    "description": (
        "Trailing 120-day OLS beta dispersion across SPY/QQQ/IWM/DIA. "
        "Feature = std-dev of the four betas; captures whether the stock has "
        "uniform broad-market exposure or concentrated style exposure. "
        "Also emits the mean beta level and a 20-bar slope of the dispersion. "
        "Per-ticker; no cross-sectional data required."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282316b_factor_beta_dispersion_4index_120_disp",   # std of 4 betas
        "ff06282316b_factor_beta_dispersion_4index_120_mean",   # mean of 4 betas
        "ff06282316b_factor_beta_dispersion_4index_120_slope",  # 20-bar slope of disp
    ],
    "tags": ["factor", "beta", "dispersion", "index", "style"],
    "version": "1.0.0",
    "author": "feature-factory ff06282316b",
}

_WINDOW = 120
_SLOPE_WIN = 20
_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]


def _rolling_beta(
    stock_ret: np.ndarray,
    idx_ret: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Vectorised rolling OLS beta of stock_ret on idx_ret over `window` bars.
    beta = Cov(x, y) / Var(y)  where y = idx_ret, x = stock_ret.
    Returns array of same length; leading `window-1` values are NaN.
    Uses pandas rolling to handle NaN propagation correctly.
    """
    n = len(stock_ret)
    if n < window:
        return np.full(n, np.nan)

    s = pd.Series(stock_ret, dtype=float)
    idx = pd.Series(idx_ret, dtype=float)

    roll_cov = s.rolling(window).cov(idx)
    roll_var = idx.rolling(window).var(ddof=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(roll_var.values != 0, roll_cov.values / roll_var.values, np.nan)

    return beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on all code paths
    col_disp = "ff06282316b_factor_beta_dispersion_4index_120_disp"
    col_mean = "ff06282316b_factor_beta_dispersion_4index_120_mean"
    col_slope = "ff06282316b_factor_beta_dispersion_4index_120_slope"

    df[col_disp] = np.nan
    df[col_mean] = np.nan
    df[col_slope] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # Stock returns (log)
    close = df["Close"].values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        stock_ret = np.empty(len(close))
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(np.where(close[:-1] != 0, close[1:] / close[:-1], np.nan))

    # Pull index close series, align to df dates via merge_asof
    dates = pd.to_datetime(df["Date"])

    beta_matrix = []  # list of arrays, one per symbol

    for sym in _SYMBOLS:
        try:
            idx_close = _indexes.index_close(sym)
            if idx_close is None or len(idx_close) == 0:
                beta_matrix.append(np.full(len(df), np.nan))
                continue

            idx_df = idx_close.reset_index()
            idx_df.columns = ["Date", "idx_close"]
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df = idx_df.sort_values("Date").reset_index(drop=True)

            # Merge backward to align index dates to stock dates
            merged = pd.merge_asof(
                pd.DataFrame({"Date": dates}).sort_values("Date"),
                idx_df,
                on="Date",
                direction="backward",
            )

            ic = merged["idx_close"].values.astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                idx_ret = np.empty(len(ic))
                idx_ret[0] = np.nan
                idx_ret[1:] = np.log(np.where(ic[:-1] != 0, ic[1:] / ic[:-1], np.nan))

            betas = _rolling_beta(stock_ret, idx_ret, _WINDOW)
            beta_matrix.append(betas)

        except Exception:
            beta_matrix.append(np.full(len(df), np.nan))

    if not beta_matrix:
        return df

    # Stack into (n_symbols, n_bars), compute per-bar std and mean
    arr = np.vstack(beta_matrix)  # shape (4, n_bars)

    with np.errstate(invalid="ignore"):
        disp = np.nanstd(arr, axis=0, ddof=1)  # dispersion = std of 4 betas
        mean_b = np.nanmean(arr, axis=0)        # mean of 4 betas

    # Where fewer than 2 non-NaN betas exist, dispersion is undefined
    n_valid = np.sum(~np.isnan(arr), axis=0)
    disp = np.where(n_valid >= 2, disp, np.nan)
    mean_b = np.where(n_valid >= 1, mean_b, np.nan)

    # Replace inf/-inf with NaN
    disp = np.where(np.isfinite(disp), disp, np.nan)
    mean_b = np.where(np.isfinite(mean_b), mean_b, np.nan)

    df[col_disp] = disp
    df[col_mean] = mean_b

    # Slope of dispersion over last _SLOPE_WIN bars (linear trend coefficient)
    disp_series = pd.Series(disp, dtype=float)
    x = np.arange(_SLOPE_WIN, dtype=float)
    x -= x.mean()  # centre for numerical stability

    def _slope(y: np.ndarray) -> float:
        if np.sum(~np.isnan(y)) < _SLOPE_WIN // 2:
            return np.nan
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        xm = x[mask]
        ym = y[mask]
        denom = np.dot(xm, xm)
        if denom == 0.0:
            return np.nan
        return float(np.dot(xm, ym) / denom)

    slope_vals = disp_series.rolling(_SLOPE_WIN).apply(_slope, raw=True)
    df[col_slope] = slope_vals.values

    return df
