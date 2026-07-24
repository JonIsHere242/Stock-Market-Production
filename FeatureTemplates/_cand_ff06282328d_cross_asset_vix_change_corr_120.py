from __future__ import annotations
import pandas as pd
import numpy as np
import importlib.util as _ilu
from pathlib import Path as _P

# Load _indexes helper
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282328d_cross_asset_vix_change_corr_120",
    "description": (
        "Trailing-120d Pearson correlation between daily stock returns and daily VIX changes. "
        "Negative values flag vol-fragile names (stock falls when fear spikes). "
        "Also produces a 20d rolling mean of VIX changes to capture the vol-sensitivity regime. "
        "Per-ticker proxy; joined to VIX via backward merge_asof (causal). "
        "Guard: if either series has zero variance in the window, result is NaN."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282328d_cross_asset_vix_change_corr_120",
        "ff06282328d_cross_asset_vix_corr_slope_20",
    ],
    "tags": ["cross_asset", "vix", "correlation", "vol_sensitivity"],
    "version": "1.0.0",
    "author": "feature-factory/ff06282328d",
}

_WINDOW = 120
_SLOPE_WINDOW = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-initialise all produced columns to NaN on every code path
    df["ff06282328d_cross_asset_vix_change_corr_120"] = np.nan
    df["ff06282328d_cross_asset_vix_corr_slope_20"] = np.nan

    if len(df) < 2:
        return df

    # Load VIX daily close and merge asof
    try:
        vix_df = _indexes.vix_daily_close()
    except Exception:
        return df

    if vix_df is None or len(vix_df) == 0:
        return df

    # Ensure Date columns are datetime
    df_work = df.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    merged = pd.merge_asof(
        df_work[["Date"]].sort_values("Date"),
        vix_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Align back to original df order
    merged = merged.set_index("Date")

    # Daily return of the stock
    close = df_work["Close"].values.astype(float)
    dates = df_work["Date"].values

    ret = np.empty(len(close))
    ret[0] = np.nan
    ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] == 0, np.nan, close[:-1])

    # VIX daily change aligned to df rows
    vix_vals = merged.reindex(df_work["Date"].values)["vix_close"].values.astype(float)
    vix_chg = np.empty(len(vix_vals))
    vix_chg[0] = np.nan
    vix_chg[1:] = vix_vals[1:] - vix_vals[:-1]

    n = len(df)

    # Rolling 120d Pearson correlation (vectorised with pandas)
    ret_s = pd.Series(ret)
    vix_chg_s = pd.Series(vix_chg)

    # rolling corr uses pairwise complete obs, returns NaN if var=0
    corr_120 = ret_s.rolling(_WINDOW, min_periods=max(30, _WINDOW // 4)).corr(vix_chg_s)

    # Replace inf with NaN just in case
    corr_120 = corr_120.replace([np.inf, -np.inf], np.nan)

    # Slope: 20d rolling mean of the corr series (how corr is trending recently)
    corr_slope = corr_120.rolling(_SLOPE_WINDOW, min_periods=5).mean()
    corr_slope = corr_slope.replace([np.inf, -np.inf], np.nan)

    df["ff06282328d_cross_asset_vix_change_corr_120"] = corr_120.values
    df["ff06282328d_cross_asset_vix_corr_slope_20"] = corr_slope.values

    return df
