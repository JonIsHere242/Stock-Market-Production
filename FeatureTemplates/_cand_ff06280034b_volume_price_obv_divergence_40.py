from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06280034b_volume_price_obv_divergence_40",
    "description": (
        "OBV-price divergence: cumulative OBV = cumsum(sign(daily_ret) * Volume). "
        "Computes rolling 40-bar correlation between OBV trend (OBV slope via linear regression residual proxy) "
        "and price trend -- negative values signal distribution (price rising while OBV falling). "
        "Also provides 20-bar change of that correlation (momentum of divergence signal) "
        "and a 40-bar z-score of OBV to flag extreme accumulation/distribution. "
        "All per-ticker, causal, no lookahead."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ff06280034b_volume_price_obv_divergence_40_corr",
        "ff06280034b_volume_price_obv_divergence_40_corr_chg20",
        "ff06280034b_volume_price_obv_divergence_40_obv_zscore",
    ],
    "tags": ["volume", "obv", "divergence", "distribution", "accumulation"],
    "version": "1.0.0",
    "author": "feature-factory ff06280034b",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    out_corr = "ff06280034b_volume_price_obv_divergence_40_corr"
    out_chg = "ff06280034b_volume_price_obv_divergence_40_corr_chg20"
    out_z = "ff06280034b_volume_price_obv_divergence_40_obv_zscore"

    # Initialise all produced columns to NaN on every code path
    df[out_corr] = np.nan
    df[out_chg] = np.nan
    df[out_z] = np.nan

    n = len(df)
    if n < 2:
        return df

    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Daily return sign: +1 if up, -1 if down, 0 if flat
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    ret[1:] = close[1:] - close[:-1]
    sign_ret = np.sign(ret)  # -1, 0, +1

    # OBV: cumulative sum of sign(ret) * Volume
    obv = np.cumsum(sign_ret * volume)

    # Rolling 40-bar correlation between OBV and Close using pandas for correctness
    window = 40
    s_obv = pd.Series(obv, index=df.index)
    s_close = pd.Series(close, index=df.index)

    corr_series = s_obv.rolling(window).corr(s_close)

    # Guard: clip to [-1, 1], coerce inf to nan
    corr_arr = corr_series.to_numpy(dtype=np.float64)
    corr_arr = np.where(np.isinf(corr_arr), np.nan, corr_arr)
    corr_arr = np.clip(corr_arr, -1.0, 1.0)

    df[out_corr] = corr_arr

    # 20-bar change in the correlation (divergence momentum)
    chg_series = corr_series.diff(20)
    chg_arr = chg_series.to_numpy(dtype=np.float64)
    chg_arr = np.where(np.isinf(chg_arr), np.nan, chg_arr)
    df[out_chg] = chg_arr

    # 40-bar OBV z-score: (OBV - rolling_mean) / rolling_std
    roll_mean = s_obv.rolling(window).mean()
    roll_std = s_obv.rolling(window).std(ddof=1)
    roll_std_arr = roll_std.to_numpy(dtype=np.float64)
    roll_mean_arr = roll_mean.to_numpy(dtype=np.float64)

    # Guard divide-by-zero
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(
            roll_std_arr == 0,
            np.nan,
            (obv - roll_mean_arr) / roll_std_arr,
        )
    zscore = np.where(np.isinf(zscore), np.nan, zscore)
    df[out_z] = zscore

    return df
