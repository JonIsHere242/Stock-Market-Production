"""
CoVaR-style stress co-movement feature block.

Per-ticker proxy: compare stock return behaviour during SPY left-tail days
vs all days, using a 250-day rolling window.  Produces:
  - delta_covar        : (mean_ret|stress - mean_ret|all) / std_all
  - stress_vol_ratio   : std(ret|stress) / std(ret|no_stress)
  - delta_covar_chg    : level minus 60-bar-ago value
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: load _indexes for SPY
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282240_factor_covar_conditional_meanret",
    "description": (
        "CoVaR-style stress co-movement. Over a 250-day trailing window, "
        "identifies SPY left-tail days (below 10th percentile of SPY return). "
        "delta_covar = (mean stock ret on stress days - mean stock ret all days) "
        "normalised by stock return std. stress_vol_ratio = std(stock|stress) / "
        "std(stock|no-stress). delta_covar_chg = level minus 60-bar-ago. "
        "Requires >= 12 stress days else NaN. Recomputed every 5 bars (stride grid "
        "fixed from series start) and forward-filled for efficiency. "
        "Per-ticker proxy for cross-sectional CoVaR; captures tail-beta / "
        "systemic-stress amplification orthogonal to unconditional beta."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282240_factor_covar_conditional_meanret_delta_covar",
        "ff06282240_factor_covar_conditional_meanret_stress_vol_ratio",
        "ff06282240_factor_covar_conditional_meanret_delta_covar_chg",
    ],
    "tags": ["factor", "covar", "tail_risk", "stress", "market_conditional"],
    "version": "1.0.0",
    "author": "feature-factory ff06282240",
}

_WINDOW = 250
_STRESS_PCT = 0.10
_MIN_STRESS = 12
_STRIDE = 5
_CHG_LAG = 60

_COL_DC = "ff06282240_factor_covar_conditional_meanret_delta_covar"
_COL_SVR = "ff06282240_factor_covar_conditional_meanret_stress_vol_ratio"
_COL_CHG = "ff06282240_factor_covar_conditional_meanret_delta_covar_chg"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN on every code path
    df[_COL_DC] = np.nan
    df[_COL_SVR] = np.nan
    df[_COL_CHG] = np.nan

    n = len(df)
    if n < _WINDOW:
        return df

    # ------------------------------------------------------------------
    # Fetch SPY close and align to df by Date (backward merge_asof)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        if spy_close is None or spy_close.empty:
            return df

        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        df_dates = df[["Date"]].copy()
        df_dates["Date"] = pd.to_datetime(df_dates["Date"])

        merged = pd.merge_asof(
            df_dates.sort_values("Date"),
            spy_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        # Restore original order
        merged = merged.set_index(df_dates.sort_values("Date").index)
        spy_arr = merged["spy_close"].values.astype(float)
    except Exception:
        return df

    # SPY daily returns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spy_ret = np.empty(n, dtype=float)
        spy_ret[0] = np.nan
        spy_ret[1:] = np.where(
            spy_arr[:-1] > 0,
            spy_arr[1:] / spy_arr[:-1] - 1.0,
            np.nan,
        )

    # Stock daily returns
    close_arr = df["Close"].values.astype(float)
    stk_ret = np.empty(n, dtype=float)
    stk_ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        stk_ret[1:] = np.where(
            close_arr[:-1] > 0,
            close_arr[1:] / close_arr[:-1] - 1.0,
            np.nan,
        )

    # ------------------------------------------------------------------
    # Rolling computation on stride grid (fixed from series start)
    # ------------------------------------------------------------------
    dc_vals = np.full(n, np.nan)
    svr_vals = np.full(n, np.nan)

    for i in range(n):
        if i % _STRIDE != 0:
            continue
        if i < _WINDOW - 1:
            continue

        w_spy = spy_ret[i - _WINDOW + 1 : i + 1]
        w_stk = stk_ret[i - _WINDOW + 1 : i + 1]

        valid_mask = np.isfinite(w_spy) & np.isfinite(w_stk)
        spy_v = w_spy[valid_mask]
        stk_v = w_stk[valid_mask]

        if len(spy_v) < _WINDOW // 2:
            continue

        # 10th percentile threshold for SPY stress
        threshold = np.percentile(spy_v, _STRESS_PCT * 100)
        stress_mask = spy_v <= threshold
        no_stress_mask = ~stress_mask

        n_stress = int(stress_mask.sum())
        if n_stress < _MIN_STRESS:
            continue

        stk_stress = stk_v[stress_mask]
        stk_no_stress = stk_v[no_stress_mask]
        stk_all = stk_v

        mean_all = stk_all.mean()
        std_all = stk_all.std()
        if not np.isfinite(std_all) or std_all < 1e-12:
            continue

        mean_stress = stk_stress.mean()
        delta_covar = (mean_stress - mean_all) / std_all

        std_stress = stk_stress.std() if len(stk_stress) > 1 else np.nan
        std_no_stress = stk_no_stress.std() if len(stk_no_stress) > 1 else np.nan

        if (
            np.isfinite(std_stress)
            and np.isfinite(std_no_stress)
        ):
            svr = std_stress / (std_no_stress + 1e-9)
        else:
            svr = np.nan

        dc_vals[i] = delta_covar
        svr_vals[i] = svr

    # Forward-fill between stride points
    for i in range(1, n):
        if np.isnan(dc_vals[i]) and not np.isnan(dc_vals[i - 1]):
            dc_vals[i] = dc_vals[i - 1]
        if np.isnan(svr_vals[i]) and not np.isnan(svr_vals[i - 1]):
            svr_vals[i] = svr_vals[i - 1]

    # Guard against inf
    dc_vals = np.where(np.isfinite(dc_vals), dc_vals, np.nan)
    svr_vals = np.where(np.isfinite(svr_vals), svr_vals, np.nan)

    # 60-bar change
    chg_vals = np.full(n, np.nan)
    for i in range(_CHG_LAG, n):
        if np.isfinite(dc_vals[i]) and np.isfinite(dc_vals[i - _CHG_LAG]):
            chg_vals[i] = dc_vals[i] - dc_vals[i - _CHG_LAG]

    df[_COL_DC] = dc_vals
    df[_COL_SVR] = svr_vals
    df[_COL_CHG] = chg_vals

    return df
