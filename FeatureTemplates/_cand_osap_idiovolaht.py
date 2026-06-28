"""
Idiosyncratic risk (AHT) — Ali, Hwang, and Trombley (2003).
Per-ticker proxy: std-dev of CAPM residuals (stock return minus beta*market return minus alpha)
over a rolling 252-day window.  Requires ≥100 non-missing observations.
Sign prediction: -1 (high idio-vol stocks under-perform).
"""
from __future__ import annotations

import warnings
import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_idiovolaht",
    "description": (
        "Idiosyncratic volatility (Ali, Hwang & Trombley 2003 / OpenSourceAP Chen-Zimmermann). "
        "Standard deviation of daily CAPM residuals over a rolling 252-day window (min 100 obs). "
        "Residual = stock_ret - (alpha + beta * market_ret) estimated by rolling OLS with SPY as market. "
        "Per-ticker proxy: inherently single-stock vs market; no cross-sectional ranking needed. "
        "Sign: -1 (high idio-vol predicts lower future returns cross-sectionally)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_idiovolaht_ivol",    # annualised idiosyncratic vol (main signal)
        "osap_idiovolaht_beta",    # rolling CAPM beta (auxiliary)
        "osap_idiovolaht_ivol_ch", # 63-day change in ivol (momentum of risk)
    ],
    "tags": ["volatility", "capm", "idiosyncratic", "aht", "opensourceap"],
    "version": "1.0",
    "author": "Ali, Hwang, and Trombley (2003) via OpenSourceAP (Chen-Zimmermann)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 252          # rolling estimation window (1 year of daily bars)
_MIN_OBS = 100         # AHT paper requirement
_ANNUALISE = np.sqrt(252)
_SLOPE_WIN = 63        # ~1 quarter change in ivol


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling idiosyncratic volatility via CAPM residuals.

    Steps:
    1. Compute daily log-returns for the stock and SPY (market proxy).
    2. For each day t, run OLS on the trailing 252-day window:
           stock_ret = alpha + beta * mkt_ret + epsilon
       using the closed-form rolling solution.
    3. ivol = std(epsilon) * sqrt(252)  [annualised].
    4. Emit ivol, beta, and 63-day change in ivol.
    """
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Stock log-returns
    # ------------------------------------------------------------------
    close = df["Close"].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock_ret = np.empty(n, dtype=np.float64)
        stock_ret[0] = np.nan
        stock_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    # ------------------------------------------------------------------
    # 2. Market (SPY) log-returns aligned to df dates
    # ------------------------------------------------------------------
    dates = df["Date"]
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        spy_df = spy_df.sort_values("Date")
        spy_df["spy_ret"] = np.log(spy_df["spy_close"] / spy_df["spy_close"].shift(1))

        tmp = pd.DataFrame({"Date": pd.to_datetime(dates)}).reset_index(drop=False)
        tmp = pd.merge_asof(tmp, spy_df[["Date", "spy_ret"]], on="Date", direction="backward")
        mkt_ret = tmp["spy_ret"].values.astype(np.float64)
    except Exception:
        # Degrade gracefully if index unavailable
        mkt_ret = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 3. Rolling OLS: closed-form via sliding window accumulators
    #    We need: sum(y), sum(x), sum(x^2), sum(xy), sum(y^2), count
    #    where x = mkt_ret, y = stock_ret.
    # ------------------------------------------------------------------
    x = mkt_ret      # shape (n,)
    y = stock_ret    # shape (n,)

    # Use pandas for rolling sums (handles NaN correctly)
    s = pd.DataFrame({"x": x, "y": y}, index=np.arange(n))

    roll = s.rolling(window=_WINDOW, min_periods=_MIN_OBS)
    cnt  = roll["x"].count().values          # non-nan pairs
    sx   = roll["x"].sum().values
    sy   = roll["y"].sum().values
    sxx  = (s["x"] ** 2).rolling(window=_WINDOW, min_periods=_MIN_OBS).sum().values
    sxy  = (s["x"] * s["y"]).rolling(window=_WINDOW, min_periods=_MIN_OBS).sum().values
    syy  = (s["y"] ** 2).rolling(window=_WINDOW, min_periods=_MIN_OBS).sum().values

    # OLS denominators
    denom = cnt * sxx - sx * sx           # n * Sxx - (Sx)^2

    with np.errstate(invalid="ignore", divide="ignore"):
        beta  = np.where(denom != 0, (cnt * sxy - sx * sy) / denom, np.nan)
        alpha = np.where(denom != 0, (sy - beta * sx) / cnt, np.nan)

        # Residual variance: Var(e) = (SSyy - beta*SSxy) / (n-2)
        # SSyy = n*syy - sy^2;  SSxy = n*sxy - sx*sy
        ss_yy = cnt * syy - sy * sy
        ss_xy = cnt * sxy - sx * sy
        ss_ee = ss_yy - beta * ss_xy          # sum of squared residuals (scaled by n)

        # Unscaled residual var: SS_ee/n^2 * 1/(n-2) ... simplify:
        # Var(e_i) = (SSyy/n - beta*(SSxy/n)) / (n-2)   ... use n*(n-2)
        res_var = np.where(
            (cnt > 2) & (denom != 0),
            ss_ee / (cnt * (cnt - 2)),
            np.nan
        )
        res_std = np.where(res_var > 0, np.sqrt(res_var), np.nan)
        ivol    = res_std * _ANNUALISE

    # ------------------------------------------------------------------
    # 4. 63-day change in ivol
    # ------------------------------------------------------------------
    ivol_series = pd.Series(ivol)
    ivol_ch = (ivol_series - ivol_series.shift(_SLOPE_WIN)).values

    # ------------------------------------------------------------------
    # 5. Assign back — no inf values
    # ------------------------------------------------------------------
    def _clean(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64)
        out[~np.isfinite(out)] = np.nan
        return out

    df["osap_idiovolaht_ivol"]    = _clean(ivol)
    df["osap_idiovolaht_beta"]    = _clean(beta)
    df["osap_idiovolaht_ivol_ch"] = _clean(ivol_ch)

    return df
