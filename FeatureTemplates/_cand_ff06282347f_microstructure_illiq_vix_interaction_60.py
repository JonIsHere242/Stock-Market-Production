from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# -- load _indexes helper -------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06282347f_microstructure_illiq_vix_interaction_60",
    "description": (
        "Illiquidity-VIX interaction: 60-day rolling Pearson correlation between the "
        "Amihud daily illiquidity ratio log1p(|ret|/(close*volume)) and the VIX level. "
        "Positive values indicate the stock's price-impact cost spikes in sync with "
        "market fear (high liquidity-risk loading); values near zero indicate "
        "idiosyncratic illiquidity. Also produces an exponentially-smoothed (32-day) "
        "level of the Amihud ratio itself as an auxiliary signal. "
        "Per-ticker proxy; no cross-sectional ranking needed."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06282347f_microstructure_illiq_vix_interaction_60_corr",
        "ff06282347f_microstructure_illiq_vix_interaction_60_illiq_ema",
        "ff06282347f_microstructure_illiq_vix_interaction_60_corr_delta",
    ],
    "tags": ["microstructure", "illiquidity", "vix", "amihud", "correlation", "liquidity_risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06282347f",
}

_WINDOW = 60
_EMA_SPAN = 32
_DELTA_LAG = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out_corr = "ff06282347f_microstructure_illiq_vix_interaction_60_corr"
    out_illiq = "ff06282347f_microstructure_illiq_vix_interaction_60_illiq_ema"
    out_delta = "ff06282347f_microstructure_illiq_vix_interaction_60_corr_delta"

    # Initialise all produced columns to NaN so every code path is covered.
    df[out_corr] = np.nan
    df[out_illiq] = np.nan
    df[out_delta] = np.nan

    if len(df) < 2:
        return df

    # -- Amihud illiquidity ratio -----------------------------------------------
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)

    ret = np.empty(len(close))
    ret[0] = np.nan
    ret[1:] = np.diff(np.log(np.where(close > 0, close, np.nan)))

    dollar_vol = close * volume  # close * volume (proxy for $ volume)
    dollar_vol = np.where(dollar_vol > 0, dollar_vol, np.nan)

    abs_ret = np.abs(ret)
    illiq_raw = np.log1p(abs_ret / dollar_vol)  # Amihud ratio

    illiq_s = pd.Series(illiq_raw, index=df.index)

    # EMA of illiquidity level (auxiliary feature)
    df[out_illiq] = illiq_s.ewm(span=_EMA_SPAN, min_periods=_EMA_SPAN // 2).mean().to_numpy()

    # -- Merge VIX (backward-safe) ----------------------------------------------
    try:
        vix_df = _indexes.vix_daily_close()
        # vix_df has columns ["Date", "vix_close"]
        tmp = df[["Date"]].copy().reset_index(drop=True)
        tmp["_orig_idx"] = np.arange(len(tmp))
        tmp_sorted = tmp.sort_values("Date")
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        tmp_sorted["Date"] = pd.to_datetime(tmp_sorted["Date"])
        merged = pd.merge_asof(tmp_sorted, vix_df, on="Date", direction="backward")
        merged = merged.sort_values("_orig_idx")
        vix_vals = merged["vix_close"].to_numpy(dtype=float)
    except Exception:
        return df

    vix_s = pd.Series(vix_vals, index=df.index)

    # -- 60-day rolling correlation between illiq and vix -----------------------
    # Use pandas rolling corr which is O(n) per bar internally.
    corr_s = illiq_s.rolling(window=_WINDOW, min_periods=_WINDOW // 2).corr(vix_s)
    df[out_corr] = corr_s.to_numpy()

    # -- Delta: change in correlation over _DELTA_LAG bars ----------------------
    df[out_delta] = corr_s.diff(_DELTA_LAG).to_numpy()

    return df
