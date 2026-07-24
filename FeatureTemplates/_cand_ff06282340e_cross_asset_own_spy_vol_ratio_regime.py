from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# Load _indexes helper by file path
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282340e_cross_asset_own_spy_vol_ratio_regime",
    "description": (
        "Cross-asset vol regime signal. Computes the ticker's trailing-20d realized volatility "
        "divided by SPY's trailing-20d realized volatility, then subtracts the rolling 120-day "
        "median of that ratio to isolate whether the stock is currently unusually agitated "
        "relative to the broad market. A second variant is the z-score of the ratio over the "
        "same 120-day window for a standardized view. Per-ticker proxy; causal/no-lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282340e_vol_ratio_vs_spy",          # raw ratio: ticker_vol20 / spy_vol20
        "ff06282340e_vol_ratio_demean",           # ratio minus its 120d rolling median
        "ff06282340e_vol_ratio_zscore",           # (ratio - 120d mean) / 120d std
    ],
    "tags": ["cross_asset", "volatility", "regime", "spy", "relative"],
    "version": "1.0.0",
    "author": "feature-factory ff06282340e",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise all produced columns to NaN upfront so every code path is covered
    for col in METADATA["produces"]:
        df[col] = np.nan

    if len(df) < 2:
        return df

    # --- ticker log-returns and 20d realized vol ---
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    ticker_vol20 = log_ret.rolling(20, min_periods=10).std()

    # --- SPY series ---
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    if spy_close is None or spy_close.empty:
        return df

    # Align SPY to the ticker's dates via merge_asof
    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    ticker_dates = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(
        ticker_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Restore original ticker row order
    merged.index = df.index[ticker_dates.sort_values("Date").index]
    spy_aligned = merged["spy_close"].reindex(df.index)

    spy_log_ret = np.log(spy_aligned / spy_aligned.shift(1))
    spy_vol20 = spy_log_ret.rolling(20, min_periods=10).std()

    # --- vol ratio ---
    spy_vol20_safe = spy_vol20.replace(0, np.nan)
    ratio = ticker_vol20 / spy_vol20_safe

    # --- demeaned: ratio minus 120d rolling median ---
    ratio_med120 = ratio.rolling(120, min_periods=30).median()
    ratio_demean = ratio - ratio_med120

    # --- z-score: (ratio - 120d mean) / 120d std ---
    ratio_mean120 = ratio.rolling(120, min_periods=30).mean()
    ratio_std120 = ratio.rolling(120, min_periods=30).std()
    ratio_std120_safe = ratio_std120.replace(0, np.nan)
    ratio_zscore = (ratio - ratio_mean120) / ratio_std120_safe

    # Replace inf/-inf with NaN
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio_demean = ratio_demean.replace([np.inf, -np.inf], np.nan)
    ratio_zscore = ratio_zscore.replace([np.inf, -np.inf], np.nan)

    df["ff06282340e_vol_ratio_vs_spy"] = ratio
    df["ff06282340e_vol_ratio_demean"] = ratio_demean
    df["ff06282340e_vol_ratio_zscore"] = ratio_zscore

    return df
