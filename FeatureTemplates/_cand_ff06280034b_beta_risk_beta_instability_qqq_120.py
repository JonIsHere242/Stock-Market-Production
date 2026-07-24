"""
Beta instability vs QQQ: rolling 40d beta computed at every bar, then
120d std / range / up-vs-down regime split of that beta series.

Per-ticker proxy: all computation is purely time-series per stock using
OHLCV returns merged with QQQ index returns -- no cross-sectional data needed.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06280034b_beta_risk_beta_instability_qqq_120",
    "description": (
        "Beta instability vs QQQ. Computes a rolling 40-day OLS beta at every bar, "
        "then summarises the 120-day std (instability), 120-day range, and the "
        "difference in mean beta between QQQ-up and QQQ-down sub-periods over 120d. "
        "High instability signals regime-unstable factor exposure. Per-ticker, "
        "causal, no cross-sectional data required."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06280034b_beta_instability_std",   # 120d std of rolling-40d beta
        "ff06280034b_beta_instability_range",  # 120d range (max-min) of rolling-40d beta
        "ff06280034b_beta_regime_spread",      # mean beta in QQQ-up days minus QQQ-down days (120d)
    ],
    "tags": ["beta", "risk", "qqq", "instability", "regime"],
    "version": "1.0",
    "author": "feature-factory ff06280034b",
}

# rolling OLS beta window
_BETA_WIN = 40
# outer summarisation window
_OUTER_WIN = 120


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN up front
    for col in METADATA["produces"]:
        df[col] = np.nan

    if len(df) < _BETA_WIN + 1:
        return df

    # ------------------------------------------------------------------
    # Fetch QQQ daily close, align to this stock's dates
    # ------------------------------------------------------------------
    try:
        qqq_series = _indexes.index_close("QQQ")  # pd.Series with DatetimeIndex
    except Exception:
        return df

    if qqq_series is None or len(qqq_series) == 0:
        return df

    # Build a temporary df for merge_asof
    qqq_df = qqq_series.rename("qqq_close").reset_index()
    qqq_df.columns = ["Date", "qqq_close"]
    qqq_df["Date"] = pd.to_datetime(qqq_df["Date"])

    stock_dates = pd.to_datetime(df["Date"])
    merge_left = pd.DataFrame({"Date": stock_dates})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = pd.merge_asof(
            merge_left.sort_values("Date"),
            qqq_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )

    # Restore original order using the original index
    aligned = aligned.set_index("Date").reindex(stock_dates.values)["qqq_close"].values

    # ------------------------------------------------------------------
    # Compute log returns
    # ------------------------------------------------------------------
    stock_close = df["Close"].values.astype(np.float64)
    stock_ret = np.empty(len(df))
    stock_ret[:] = np.nan
    stock_ret[1:] = np.diff(np.log(np.where(stock_close > 0, stock_close, np.nan)))

    qqq_close = aligned.astype(np.float64)
    qqq_ret = np.empty(len(df))
    qqq_ret[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid_qqq = qqq_close > 0
        log_qqq = np.where(valid_qqq, np.log(qqq_close), np.nan)
    qqq_ret[1:] = np.diff(log_qqq)

    n = len(df)

    # ------------------------------------------------------------------
    # Rolling 40-day OLS beta at every bar using vectorised sliding sum
    # ------------------------------------------------------------------
    # beta = Cov(r_stock, r_qqq) / Var(r_qqq)  over window
    # Use pandas rolling for clean NaN handling
    sr = pd.Series(stock_ret)
    qr = pd.Series(qqq_ret)

    win = _BETA_WIN

    roll_cov = sr.rolling(win).cov(qr)          # rolling covariance
    roll_var = qr.rolling(win).var()             # rolling variance of QQQ

    rolling_beta = np.where(
        roll_var.values != 0,
        roll_cov.values / roll_var.values,
        np.nan,
    )
    rolling_beta = np.where(np.isfinite(rolling_beta), rolling_beta, np.nan)

    rb_series = pd.Series(rolling_beta)

    # ------------------------------------------------------------------
    # 120d std and range of the rolling-40d beta series
    # ------------------------------------------------------------------
    outer = _OUTER_WIN

    beta_std = rb_series.rolling(outer).std().values
    beta_max = rb_series.rolling(outer).max().values
    beta_min = rb_series.rolling(outer).min().values
    beta_range = beta_max - beta_min

    # Guard
    beta_std = np.where(np.isfinite(beta_std), beta_std, np.nan)
    beta_range = np.where(np.isfinite(beta_range), beta_range, np.nan)

    # ------------------------------------------------------------------
    # Regime spread: mean(beta | QQQ_up) - mean(beta | QQQ_down) over 120d
    # QQQ_up days: qqq_ret > 0
    # ------------------------------------------------------------------
    qqq_up = (qr > 0).astype(float)   # 1 on up days, 0 on down days (NaN days = 0)
    qqq_dn = (qr <= 0).astype(float)

    # Replace NaN qqq_ret days with neither up nor down
    qqq_nan_mask = qr.isna().values
    qqq_up_arr = qqq_up.values.copy()
    qqq_dn_arr = qqq_dn.values.copy()
    qqq_up_arr[qqq_nan_mask] = np.nan
    qqq_dn_arr[qqq_nan_mask] = np.nan

    rb_arr = rb_series.values.copy()

    # rolling mean of beta on up days = rolling sum(beta * is_up) / rolling sum(is_up)
    rb_s = pd.Series(rb_arr)
    up_s = pd.Series(qqq_up_arr)
    dn_s = pd.Series(qqq_dn_arr)

    # product: beta value when up, else NaN
    beta_on_up = np.where(up_s.values == 1.0, rb_arr, np.nan)
    beta_on_dn = np.where(dn_s.values == 1.0, rb_arr, np.nan)

    beta_on_up_s = pd.Series(beta_on_up)
    beta_on_dn_s = pd.Series(beta_on_dn)

    # rolling nanmean via sum / count
    roll_up_sum = beta_on_up_s.rolling(outer, min_periods=5).sum()
    roll_up_cnt = pd.Series(np.where(~np.isnan(beta_on_up), 1.0, 0.0)).rolling(outer, min_periods=5).sum()
    roll_dn_sum = beta_on_dn_s.rolling(outer, min_periods=5).sum()
    roll_dn_cnt = pd.Series(np.where(~np.isnan(beta_on_dn), 1.0, 0.0)).rolling(outer, min_periods=5).sum()

    mean_beta_up = np.where(roll_up_cnt.values > 0, roll_up_sum.values / roll_up_cnt.values, np.nan)
    mean_beta_dn = np.where(roll_dn_cnt.values > 0, roll_dn_sum.values / roll_dn_cnt.values, np.nan)

    regime_spread = mean_beta_up - mean_beta_dn
    regime_spread = np.where(np.isfinite(regime_spread), regime_spread, np.nan)

    # ------------------------------------------------------------------
    # Assign back to df
    # ------------------------------------------------------------------
    df["ff06280034b_beta_instability_std"] = beta_std
    df["ff06280034b_beta_instability_range"] = beta_range
    df["ff06280034b_beta_regime_spread"] = regime_spread

    return df
