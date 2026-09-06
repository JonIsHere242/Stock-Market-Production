from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext5_abnormal_vol_response",
    "description": (
        "Price discovery efficiency per unit of abnormal volume. "
        "Abnormal volume = Volume / 20d rolling mean volume. "
        "Produces three per-ticker signals: "
        "(1) ext5_abnormal_vol_response_ratio: rolling 60d ratio of cumulative |return| to cumulative "
        "abnormal volume -- high = thin/jumpy market (each unit of excess volume moves price a lot), "
        "low = absorbing/liquid. "
        "(2) ext5_abnormal_vol_response_corr: rolling 60d Pearson correlation of abnormal volume with "
        "same-day |return| -- measures how tightly price impact tracks volume surges. "
        "(3) ext5_abnormal_vol_response_slope: 20d z-score of the ratio, capturing acceleration / "
        "regime change in price-discovery efficiency. "
        "Pure OHLCV; no lookahead. Proxy is per-ticker (cross-sectional rank not available at block level)."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext5_abnormal_vol_response_ratio",
        "ext5_abnormal_vol_response_corr",
        "ext5_abnormal_vol_response_slope",
    ],
    "tags": ["volume", "price_impact", "price_discovery", "abnormal_volume", "liquidity"],
    "version": "1.0",
    "author": "Round-6 fresh vein spec (NEW: volume-price), Source: ext5_abnormal_vol_response spec",
}

_VOL_WIN = 20      # window for average volume baseline
_ROLL_WIN = 60     # main rolling window for ratio + corr
_SLOPE_WIN = 20    # z-score window for slope/regime


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Absolute daily return (close-to-close, no lookahead) ---
    close = df["Close"]
    abs_ret = close.pct_change().abs()            # NaN on first row -- expected

    # --- 2. Abnormal volume = Volume / 20d rolling mean volume ---
    vol = df["Volume"].astype(float)
    avg_vol = vol.rolling(_VOL_WIN, min_periods=1).mean()
    # Guard: avg_vol == 0 → NaN
    avg_vol = avg_vol.where(avg_vol > 0, np.nan)
    abnorm_vol = vol / avg_vol                    # ≥ 0; NaN where avg_vol is NaN

    # --- 3. Rolling 60d ratio: cumulative |return| / cumulative abnormal volume ---
    cum_abs_ret = abs_ret.rolling(_ROLL_WIN, min_periods=_ROLL_WIN // 2).sum()
    cum_abnorm_vol = abnorm_vol.rolling(_ROLL_WIN, min_periods=_ROLL_WIN // 2).sum()
    # Guard division
    denom = cum_abnorm_vol.where(cum_abnorm_vol > 0, np.nan)
    ratio = cum_abs_ret / denom

    # Replace inf/-inf just in case
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # --- 4. Rolling 60d Pearson correlation: abnorm_vol vs |return| ---
    # Use pandas rolling corr (vectorised)
    corr = abnorm_vol.rolling(_ROLL_WIN, min_periods=_ROLL_WIN // 2).corr(abs_ret)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    # --- 5. 20d z-score of the ratio (slope / regime change in efficiency) ---
    ratio_mean = ratio.rolling(_SLOPE_WIN, min_periods=_SLOPE_WIN // 2).mean()
    ratio_std = ratio.rolling(_SLOPE_WIN, min_periods=_SLOPE_WIN // 2).std()
    ratio_std = ratio_std.where(ratio_std > 0, np.nan)
    slope = (ratio - ratio_mean) / ratio_std
    slope = slope.replace([np.inf, -np.inf], np.nan)

    df["ext5_abnormal_vol_response_ratio"] = ratio
    df["ext5_abnormal_vol_response_corr"] = corr
    df["ext5_abnormal_vol_response_slope"] = slope

    return df
