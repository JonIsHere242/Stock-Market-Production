"""
Idiosyncratic return skewness (lottery demand proxy) -- Boyer, Mitton & Vorkink (2010).
Regresses daily stock return on SPY return over a trailing 120-day window, takes residuals,
then computes rolling 60-day skewness of those residuals (per-ticker, causal).
A separate 20-day change in that skewness captures momentum in lottery demand.
Orthogonal to idiosyncratic VOLATILITY (osap_idiovolaht), which uses residual std-dev.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------- helper: SPY index ---------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------- metadata ------------------------------------------------------------
METADATA = {
    "name": "ext_idio_skew",
    "description": (
        "Idiosyncratic-return skewness (lottery proxy). "
        "Per-ticker: regress daily log-return on SPY log-return over a trailing "
        "120-day OLS window; compute 60-day rolling skewness of the residuals. "
        "Sign convention: high positive skew = lottery stock (typically mean-reverting "
        "negatively in next-day returns per Boyer-Mitton-Vorkink 2010). "
        "Also emits the 20-day change in that skewness to capture trend in lottery demand. "
        "Orthogonal to idiosyncratic volatility (osap_idiovolaht) which uses residual std. "
        "Cross-sectional ranking not possible per-ticker; proxy is the per-stock time-series."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext_idio_skew_60d",      # rolling 60-day skewness of idio residuals
        "ext_idio_skew_delta20",   # 20-day change in ext_idio_skew_60d
    ],
    "tags": ["skewness", "lottery", "idiosyncratic", "residual", "momentum"],
    "version": "1.0.0",
    "author": "Boyer, Mitton & Vorkink (2010) RFS; spec ext_idio_skew",
}

# ---------- constants -----------------------------------------------------------
_BETA_WIN = 120      # window for OLS beta estimation (days)
_SKEW_WIN = 60       # window for rolling skewness of residuals
_DELTA_WIN = 20      # look-back for skewness change
_MIN_PERIODS_BETA = 60
_MIN_PERIODS_SKEW = 30


def _rolling_ols_residuals(
    stock_ret: np.ndarray, spy_ret: np.ndarray, win: int, min_periods: int
) -> np.ndarray:
    """
    For each day t, fit OLS: stock_ret[t-win+1:t+1] ~ spy_ret[...].
    Return per-day residual (hat{e}_t = stock_ret[t] - beta_hat * spy_ret[t] - alpha_hat).
    Vectorised via cumsum trick for O(n) speed.
    """
    n = len(stock_ret)
    residuals = np.full(n, np.nan, dtype=np.float64)

    # prefix sums for O(1) window OLS
    s = np.zeros(n + 1)
    sx = np.zeros(n + 1)   # spy
    sy = np.zeros(n + 1)   # stock
    sxx = np.zeros(n + 1)  # spy^2
    sxy = np.zeros(n + 1)  # spy*stock

    for i in range(n):
        x = spy_ret[i]
        y = stock_ret[i]
        s[i + 1] = s[i] + (0 if np.isnan(x) or np.isnan(y) else 1)
        sx[i + 1] = sx[i] + (0.0 if np.isnan(x) else x)
        sy[i + 1] = sy[i] + (0.0 if np.isnan(y) else y)
        sxx[i + 1] = sxx[i] + (0.0 if np.isnan(x) else x * x)
        sxy[i + 1] = sxy[i] + (0.0 if np.isnan(x) or np.isnan(y) else x * y)

    for i in range(win - 1, n):
        lo = i + 1 - win
        n_w = s[i + 1] - s[lo]
        if n_w < min_periods:
            continue
        if np.isnan(stock_ret[i]) or np.isnan(spy_ret[i]):
            continue
        sum_x = sx[i + 1] - sx[lo]
        sum_y = sy[i + 1] - sy[lo]
        sum_xx = sxx[i + 1] - sxx[lo]
        sum_xy = sxy[i + 1] - sxy[lo]
        denom = n_w * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-14:
            continue
        beta = (n_w * sum_xy - sum_x * sum_y) / denom
        alpha = (sum_y - beta * sum_x) / n_w
        residuals[i] = stock_ret[i] - (alpha + beta * spy_ret[i])

    return residuals


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ---- stock log-returns -----------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret = np.full(len(close), np.nan)
        mask = (close[:-1] > 0) & (close[1:] > 0)
        stock_ret[1:] = np.where(mask, np.log(close[1:] / close[:-1]), np.nan)

    # ---- SPY log-returns -------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["spy_ret"] = np.log(
            spy_df["spy_close"] / spy_df["spy_close"].shift(1)
        )

        dates_df = df[["Date"]].copy()
        dates_df["Date"] = pd.to_datetime(dates_df["Date"])
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        merged = pd.merge_asof(
            dates_df.sort_values("Date"),
            spy_df[["Date", "spy_ret"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )
        merged = merged.set_index("Date").reindex(
            pd.to_datetime(dates_df["Date"])
        )
        spy_ret = merged["spy_ret"].values.astype(np.float64)
    except Exception:
        # degrade gracefully: without SPY we can't compute idio residuals
        df["ext_idio_skew_60d"] = np.nan
        df["ext_idio_skew_delta20"] = np.nan
        return df

    # ---- rolling OLS residuals -------------------------------------------------
    residuals = _rolling_ols_residuals(
        stock_ret, spy_ret, win=_BETA_WIN, min_periods=_MIN_PERIODS_BETA
    )

    # ---- rolling 60-day skewness of residuals ----------------------------------
    resid_s = pd.Series(residuals, index=df.index)

    def _skew(x: np.ndarray) -> float:
        """Unbiased Fisher skewness; returns NaN if < 3 valid values."""
        v = x[~np.isnan(x)]
        n = len(v)
        if n < 3:
            return np.nan
        mu = v.mean()
        sigma = v.std(ddof=1)
        if sigma < 1e-14:
            return np.nan
        return float(((v - mu) ** 3).mean() / sigma**3)

    skew_60 = (
        resid_s
        .rolling(window=_SKEW_WIN, min_periods=_MIN_PERIODS_SKEW)
        .apply(_skew, raw=True)
    )

    df["ext_idio_skew_60d"] = skew_60.values
    df["ext_idio_skew_delta20"] = (skew_60 - skew_60.shift(_DELTA_WIN)).values

    return df
