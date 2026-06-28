"""
Volatility of idiosyncratic volatility (vol-of-idio-vol).

Measures the INSTABILITY of idiosyncratic risk rather than its level.
Parent feature (osap_idiovolaht) captures the level; this block captures
whether that level is itself stable or erratic over time — a distinct axis.

Per-ticker proxy: market residuals are estimated by regressing rolling 20d
daily returns against SPY returns. The 20d rolling std of those residuals
(idio_vol) is then subjected to a 60d rolling std (vol-of-vol) and 60d
OLS slope (trend of idio-vol instability). Degrades gracefully to NaN when
SPY data is unavailable (uses raw return std as fallback).
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Optional: load _indexes helper
# ---------------------------------------------------------------------------
try:
    _spec = _ilu.spec_from_file_location(
        "_indexes", _P(__file__).resolve().parent / "_indexes.py"
    )
    _indexes = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_indexes)
    _HAS_INDEXES = True
except Exception:
    _HAS_INDEXES = False

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_idio_vol_of_vol",
    "description": (
        "Volatility of idiosyncratic volatility. Computes a 20-day rolling "
        "std of market-model residuals (idio-vol level), then applies a 60-day "
        "rolling std (vol-of-vol) and a 60-day OLS slope (trend). Captures the "
        "INSTABILITY of idiosyncratic risk, orthogonal to osap_idiovolaht which "
        "measures the level. Per-ticker proxy using SPY as the market factor. "
        "Degrades to raw-return-std residuals when SPY data is unavailable."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_idio_vol_of_vol_level",   # 60d std of rolling 20d idio-vol
        "ext_idio_vol_of_vol_slope",   # 60d OLS slope of rolling 20d idio-vol
    ],
    "tags": ["volatility", "idiosyncratic", "instability", "market-model", "extension"],
    "version": "1.0.0",
    "author": "Extension/exploration of gate-validated winner osap_idiovolaht",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SHORT_WIN = 20   # idio-vol estimation window
_LONG_WIN  = 60   # vol-of-vol / slope window
_MIN_LONG  = 30   # minimum obs for vol-of-vol to emit a value


def _rolling_ols_slope(series: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Vectorised rolling OLS slope using a 1-D x-axis (0..window-1)."""
    n = len(series)
    out = np.full(n, np.nan)
    # precompute centred x for the window
    x = np.arange(window, dtype=float)
    xm = x.mean()
    x_dev = x - xm
    x_ss = (x_dev ** 2).sum()
    if x_ss == 0:
        return out
    for i in range(window - 1, n):
        y = series[i - window + 1 : i + 1]
        if np.sum(~np.isnan(y)) < min_periods:
            continue
        # replace NaN with mean to avoid contamination; if all NaN skip
        valid = ~np.isnan(y)
        if valid.sum() < min_periods:
            continue
        ym = np.nanmean(y)
        y_filled = np.where(valid, y, ym)
        out[i] = (x_dev * (y_filled - ym)).sum() / x_ss
    return out


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Daily log returns for this ticker                                 #
    # ------------------------------------------------------------------ #
    close = df["Close"].values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_ret = np.full(len(close), np.nan)
        mask = (close[:-1] > 0) & (close[1:] > 0)
        log_ret[1:][mask] = np.log(close[1:][mask] / close[:-1][mask])

    # ------------------------------------------------------------------ #
    # 2. SPY returns aligned to df dates                                   #
    # ------------------------------------------------------------------ #
    spy_ret = np.full(len(df), np.nan)
    if _HAS_INDEXES:
        try:
            spy_close = _indexes.index_close("SPY")  # pd.Series, DatetimeIndex
            spy_df = spy_close.rename("_spy_close").reset_index()
            spy_df.columns = ["Date", "_spy_close"]
            spy_df = spy_df.sort_values("Date")

            tmp = df[["Date"]].copy()
            tmp = tmp.sort_values("Date")
            merged = pd.merge_asof(tmp, spy_df, on="Date", direction="backward")
            spy_c = merged["_spy_close"].values.astype(float)
            spy_r = np.full(len(spy_c), np.nan)
            m2 = (spy_c[:-1] > 0) & (spy_c[1:] > 0)
            spy_r[1:][m2] = np.log(spy_c[1:][m2] / spy_c[:-1][m2])
            spy_ret = spy_r
        except Exception:
            pass  # fallback: spy_ret stays NaN

    # ------------------------------------------------------------------ #
    # 3. Rolling 20-day idio-vol (std of residuals)                        #
    # ------------------------------------------------------------------ #
    n = len(log_ret)
    idio_vol = np.full(n, np.nan)

    for i in range(_SHORT_WIN - 1, n):
        y_win = log_ret[i - _SHORT_WIN + 1 : i + 1]
        m_win = spy_ret[i - _SHORT_WIN + 1 : i + 1]

        valid = ~(np.isnan(y_win) | np.isnan(m_win))
        if valid.sum() >= max(5, _SHORT_WIN // 2):
            yv = y_win[valid]
            mv = m_win[valid]
            # OLS beta
            mv_c = mv - mv.mean()
            ss = (mv_c ** 2).sum()
            if ss > 0:
                beta = (mv_c * (yv - yv.mean())).sum() / ss
                resid = yv - (yv.mean() + beta * mv_c)
            else:
                resid = yv - yv.mean()
            idio_vol[i] = resid.std(ddof=1) if len(resid) > 1 else np.nan
        else:
            # fallback: use raw log_ret std if no SPY alignment
            yv = y_win[~np.isnan(y_win)]
            if len(yv) >= 5:
                idio_vol[i] = yv.std(ddof=1)

    # ------------------------------------------------------------------ #
    # 4a. 60-day rolling std of idio_vol  (vol-of-vol level)              #
    # ------------------------------------------------------------------ #
    vol_of_vol = np.full(n, np.nan)
    for i in range(_LONG_WIN - 1, n):
        window = idio_vol[i - _LONG_WIN + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= _MIN_LONG:
            vol_of_vol[i] = valid.std(ddof=1)

    # ------------------------------------------------------------------ #
    # 4b. 60-day OLS slope of idio_vol  (trend of instability)            #
    # ------------------------------------------------------------------ #
    vov_slope = _rolling_ols_slope(idio_vol, window=_LONG_WIN, min_periods=_MIN_LONG)

    # ------------------------------------------------------------------ #
    # 5. Guard: replace inf/-inf with NaN                                  #
    # ------------------------------------------------------------------ #
    vol_of_vol = np.where(np.isfinite(vol_of_vol), vol_of_vol, np.nan)
    vov_slope  = np.where(np.isfinite(vov_slope),  vov_slope,  np.nan)

    # ------------------------------------------------------------------ #
    # 6. Attach columns and return                                         #
    # ------------------------------------------------------------------ #
    df["ext_idio_vol_of_vol_level"] = vol_of_vol
    df["ext_idio_vol_of_vol_slope"] = vov_slope
    return df
