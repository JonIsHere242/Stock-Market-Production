"""
Candidate feature block: ext_downside_idio_vol
Downside idiosyncratic semivolatility -- rolling semideviation of market-model
residuals.  Captures crash asymmetry in idiosyncratic risk, orthogonal to the
parent osap_idiovolaht which measures total idiosyncratic vol.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path, never by package import)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_downside_idio_vol",
    "description": (
        "Downside idiosyncratic semivolatility.  For each stock, a rolling "
        "60-day OLS beta is estimated against SPY log-returns (using a trailing "
        "120-day window so the beta itself has 60 in-sample days of history). "
        "Daily idiosyncratic residuals = stock log-ret - beta * SPY log-ret. "
        "Three features are produced: "
        "(1) ext_downside_idio_vol_semi -- rolling 60d RMS of NEGATIVE residuals "
        "(downside semideviation, i.e. crash intensity); "
        "(2) ext_downside_idio_vol_ratio -- downside semideviation / upside "
        "semideviation (idiosyncratic crash asymmetry; >1 = fat left tail); "
        "(3) ext_downside_idio_vol_zscore -- 63-day rolling z-score of the "
        "downside semivol (captures regime shift / trend in crash risk). "
        "All windows are causal (no lookahead). Proxy note: beta is a simple "
        "rolling OLS per-ticker (not cross-sectional Fama-French); degrades to "
        "NaN when SPY is unavailable."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_downside_idio_vol_semi",
        "ext_downside_idio_vol_ratio",
        "ext_downside_idio_vol_zscore",
    ],
    "tags": ["volatility", "idiosyncratic", "downside", "semivariance", "risk"],
    "version": "1.0.0",
    "author": (
        "Extension/exploration of gate-validated winner osap_idiovolaht; "
        "downside semivariance concept from Ang, Chen & Xing (2006) "
        "'Downside Risk', Review of Financial Studies."
    ),
}

# ---------------------------------------------------------------------------
# Rolling OLS beta helper -- vectorised via numpy sliding windows
# ---------------------------------------------------------------------------
_BETA_WIN = 120   # days used to estimate rolling beta
_SEMI_WIN  = 60   # window for semideviation (last 60 of the 120 beta days)
_ZSCORE_WIN = 63  # look-back for z-scoring the semivol


def _rolling_ols_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal rolling OLS slope of y ~ x (no intercept beyond demeaning).
    Returns array of same length as y, NaN for first (window-1) rows.
    Uses vectorised cov/var accumulation via np.lib.stride_tricks.
    """
    n = len(y)
    beta = np.full(n, np.nan)
    if n < window:
        return beta

    # sliding window views -- shape (n-window+1, window)
    from numpy.lib.stride_tricks import sliding_window_view
    y_wins = sliding_window_view(y, window)  # shape (n-win+1, win)
    x_wins = sliding_window_view(x, window)

    y_mean = y_wins.mean(axis=1)
    x_mean = x_wins.mean(axis=1)

    dy = y_wins - y_mean[:, None]
    dx = x_wins - x_mean[:, None]

    cov_xy = (dy * dx).mean(axis=1)
    var_x  = (dx * dx).mean(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(var_x == 0, np.nan, cov_xy / var_x)

    beta[window - 1:] = b
    return beta


def _rolling_semivol(resid: np.ndarray, window: int):
    """
    Rolling RMS of NEGATIVE residuals and rolling RMS of POSITIVE residuals.
    Returns (down_semi, up_semi) arrays of same length as resid.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(resid)
    down_semi = np.full(n, np.nan)
    up_semi   = np.full(n, np.nan)

    if n < window:
        return down_semi, up_semi

    wins = sliding_window_view(resid, window)  # (n-win+1, win)

    neg = np.where(wins < 0, wins, 0.0)
    pos = np.where(wins > 0, wins, 0.0)

    neg_count = (wins < 0).sum(axis=1)
    pos_count = (wins > 0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        d_semi = np.where(
            neg_count > 0,
            np.sqrt((neg ** 2).sum(axis=1) / neg_count),
            np.nan,
        )
        u_semi = np.where(
            pos_count > 0,
            np.sqrt((pos ** 2).sum(axis=1) / pos_count),
            np.nan,
        )

    down_semi[window - 1:] = d_semi
    up_semi[window - 1:]   = u_semi
    return down_semi, up_semi


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- initialise output columns as NaN ----
    df["ext_downside_idio_vol_semi"]   = np.nan
    df["ext_downside_idio_vol_ratio"]  = np.nan
    df["ext_downside_idio_vol_zscore"] = np.nan

    if len(df) < _BETA_WIN + _SEMI_WIN:
        return df

    # ---- fetch SPY daily close ----
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    # ---- align SPY to df dates via merge_asof ----
    df_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # realign to original df row order
    merged = merged.set_index(df_dates.sort_values("Date").index)
    spy_aligned = merged["spy_close"].reindex(df.index)

    stock_close = df["Close"].values.astype(float)
    spy_vals    = spy_aligned.values.astype(float)

    # ---- log returns (NaN-safe) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        stock_ret = np.where(
            stock_close[:-1] > 0,
            np.diff(np.log(np.where(stock_close > 0, stock_close, np.nan))),
            np.nan,
        )
        spy_ret = np.where(
            spy_vals[:-1] > 0,
            np.diff(np.log(np.where(spy_vals > 0, spy_vals, np.nan))),
            np.nan,
        )

    # prepend NaN for day-0 (no prior day)
    stock_ret = np.concatenate([[np.nan], stock_ret])
    spy_ret   = np.concatenate([[np.nan], spy_ret])

    # ---- rolling beta (120d window) ----
    beta = _rolling_ols_beta(stock_ret, spy_ret, _BETA_WIN)

    # ---- idiosyncratic residuals ----
    with np.errstate(invalid="ignore"):
        resid = stock_ret - beta * spy_ret

    # ---- rolling 60d downside / upside semideviation ----
    down_semi, up_semi = _rolling_semivol(resid, _SEMI_WIN)

    # ---- ratio (downside / upside) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            (up_semi > 0) & np.isfinite(down_semi),
            down_semi / up_semi,
            np.nan,
        )

    # ---- 63d z-score of downside semivol ----
    n = len(down_semi)
    zscore = np.full(n, np.nan)
    if n >= _ZSCORE_WIN:
        from numpy.lib.stride_tricks import sliding_window_view
        wins_d = sliding_window_view(down_semi, _ZSCORE_WIN)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu  = np.nanmean(wins_d, axis=1)
            std = np.nanstd(wins_d, axis=1, ddof=1)
            z   = np.where(std > 0, (down_semi[_ZSCORE_WIN - 1:] - mu) / std, np.nan)
        zscore[_ZSCORE_WIN - 1:] = z

    # ---- guard inf / -inf ----
    def _clean(arr):
        return np.where(np.isfinite(arr), arr, np.nan)

    df["ext_downside_idio_vol_semi"]   = _clean(down_semi)
    df["ext_downside_idio_vol_ratio"]  = _clean(ratio)
    df["ext_downside_idio_vol_zscore"] = _clean(zscore)

    return df
