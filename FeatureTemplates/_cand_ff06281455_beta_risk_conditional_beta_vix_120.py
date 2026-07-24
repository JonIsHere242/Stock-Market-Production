from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Load _indexes helper (by path, no package import)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06281455_beta_risk_conditional_beta_vix_120",
    "description": (
        "VIX-regime-conditional beta to SPY over a 120-day rolling window. "
        "The VIX regime (high vs low) is determined by comparing today's VIX to "
        "the trailing 250-day median VIX. Produces: beta in high-VIX periods, "
        "beta in low-VIX periods, and the stress-spread (high minus low). "
        "This is a per-ticker proxy -- cross-sectional ranking is not used. "
        "Captures the asymmetric market-sensitivity of a stock under stress vs calm."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06281455_beta_risk_conditional_beta_vix_120_high",
        "ff06281455_beta_risk_conditional_beta_vix_120_low",
        "ff06281455_beta_risk_conditional_beta_vix_120_spread",
    ],
    "tags": ["beta", "vix", "regime", "risk", "conditional"],
    "version": "1.0.0",
    "author": "feature-factory ff06281455",
}

_WINDOW = 120        # rolling beta window (bars)
_VIX_LOOKBACK = 250  # bars for trailing VIX median (regime boundary)
_MIN_OBS = 20        # minimum observations per regime bucket for valid beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    col_high   = "ff06281455_beta_risk_conditional_beta_vix_120_high"
    col_low    = "ff06281455_beta_risk_conditional_beta_vix_120_low"
    col_spread = "ff06281455_beta_risk_conditional_beta_vix_120_spread"

    df[col_high]   = np.nan
    df[col_low]    = np.nan
    df[col_spread] = np.nan

    if len(df) < _MIN_OBS + 1:
        return df

    # -----------------------------------------------------------------------
    # 1. Pull SPY close and VIX; align backward (lookahead-safe)
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")  # Series, DatetimeIndex
    except Exception:
        return df

    try:
        vix_df = _indexes.vix_daily_close()      # DataFrame[Date, vix_close]
    except Exception:
        return df

    # Ensure Date column is datetime for merge
    df_work = df[["Date", "Close"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])

    spy_frame = spy_close.reset_index()
    spy_frame.columns = ["Date", "spy_close"]
    spy_frame["Date"] = pd.to_datetime(spy_frame["Date"])

    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    # merge_asof requires sorted keys
    df_work = df_work.sort_values("Date").reset_index(drop=True)
    spy_frame = spy_frame.sort_values("Date")
    vix_df = vix_df.sort_values("Date")

    df_work = pd.merge_asof(df_work, spy_frame, on="Date", direction="backward")
    df_work = pd.merge_asof(df_work, vix_df,   on="Date", direction="backward")

    # -----------------------------------------------------------------------
    # 2. Daily log returns (stock and SPY)
    # -----------------------------------------------------------------------
    stock_ret = np.log(df_work["Close"]   / df_work["Close"].shift(1))
    spy_ret   = np.log(df_work["spy_close"] / df_work["spy_close"].shift(1))
    vix       = df_work["vix_close"].values.astype(float)

    n = len(df_work)

    # -----------------------------------------------------------------------
    # 3. Trailing VIX median for regime classification (250-bar expanding up
    #    to full lookback, then rolling)
    # -----------------------------------------------------------------------
    vix_series = pd.Series(vix)
    # Rolling median of the PAST _VIX_LOOKBACK bars (inclusive of today)
    vix_med = vix_series.rolling(_VIX_LOOKBACK, min_periods=30).median().values

    stock_ret_arr = stock_ret.values
    spy_ret_arr   = spy_ret.values

    # Output arrays
    beta_high_arr   = np.full(n, np.nan)
    beta_low_arr    = np.full(n, np.nan)
    beta_spread_arr = np.full(n, np.nan)

    # -----------------------------------------------------------------------
    # 4. Rolling conditional beta over _WINDOW bars
    #    For each bar t (1-indexed from 0), use the window [t-WINDOW, t].
    #    Within that window, classify each bar as high-VIX or low-VIX based
    #    on whether vix[i] >= vix_med[t] (the regime boundary AT time t).
    # -----------------------------------------------------------------------
    for t in range(_WINDOW - 1, n):
        med_t = vix_med[t]
        if np.isnan(med_t):
            continue

        w_stock = stock_ret_arr[t - _WINDOW + 1 : t + 1]
        w_spy   = spy_ret_arr[t - _WINDOW + 1 : t + 1]
        w_vix   = vix[t - _WINDOW + 1 : t + 1]

        valid = np.isfinite(w_stock) & np.isfinite(w_spy) & np.isfinite(w_vix)

        high_mask = valid & (w_vix >= med_t)
        low_mask  = valid & (w_vix <  med_t)

        def _beta(x: np.ndarray, y: np.ndarray) -> float:
            """OLS beta of x on y (no intercept robust via demeaned cov/var)."""
            if len(x) < _MIN_OBS:
                return np.nan
            xm = x - x.mean()
            ym = y - y.mean()
            var_y = (ym * ym).sum()
            if var_y == 0.0 or not np.isfinite(var_y):
                return np.nan
            return float((xm * ym).sum() / var_y)

        bh = _beta(w_stock[high_mask], w_spy[high_mask])
        bl = _beta(w_stock[low_mask],  w_spy[low_mask])

        beta_high_arr[t]   = bh
        beta_low_arr[t]    = bl
        if np.isfinite(bh) and np.isfinite(bl):
            beta_spread_arr[t] = bh - bl

    # -----------------------------------------------------------------------
    # 5. Map back to original df index using Date alignment
    # -----------------------------------------------------------------------
    result = pd.Series(df_work["Date"].values, name="Date").to_frame()
    result[col_high]   = beta_high_arr
    result[col_low]    = beta_low_arr
    result[col_spread] = beta_spread_arr

    # Align back on original df (Date may have been reordered)
    orig_dates = pd.to_datetime(df["Date"])
    date_to_high   = dict(zip(df_work["Date"], beta_high_arr))
    date_to_low    = dict(zip(df_work["Date"], beta_low_arr))
    date_to_spread = dict(zip(df_work["Date"], beta_spread_arr))

    df[col_high]   = orig_dates.map(date_to_high)
    df[col_low]    = orig_dates.map(date_to_low)
    df[col_spread] = orig_dates.map(date_to_spread)

    return df
