"""
Higher Moments Matter  —  DOI:10.1002/rfe.1121
"Higher moments matter! Cross-sectional (higher) moments and the predictability of stock returns"

The paper investigates cross-sectional volatility, skewness, and kurtosis (computed
ACROSS all stocks each day) as predictors of the equity premium (index-level return).
This is a cross-sectional signal that CANNOT be replicated per-ticker without access
to all other tickers on the same day.

Per-ticker proxy implemented here: rolling TIME-SERIES moments of the single return
series — capturing the LOCAL distributional dynamics the cross-sectional paper identifies
as economically meaningful. The paper's PCA-based aggregation of the three moments is
also proxied by computing each individually and a composite.

  * hmm_ts_vol_22 / _60: rolling realized volatility (std of daily log-returns).
  * hmm_ts_skew_22 / _60: rolling realized skewness of log-returns.
  * hmm_ts_kurt_22 / _60: rolling realized excess kurtosis of log-returns.
  * hmm_vol_of_vol_60: volatility of volatility (std of rolling 10d vol over 60d window)
    — a per-ticker proxy for the "cross-sectional vol" dynamic that the paper links to
    equity premium predictability.
  * hmm_skew_change_22: first difference of 22d skewness (momentum in skewness state).

NOTE: The cross-sectional aggregation across stocks is dropped. All signals are
computed from the single stock's own return history only.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_oalex_rfe1121_higher_moments_matter",
    "description": (
        "Per-ticker rolling time-series volatility, skewness, and kurtosis proxies for "
        "the cross-sectional higher-moments equity premium signals (DOI:10.1002/rfe.1121). "
        "Cross-sectional aggregation across stocks dropped; per-ticker distributional dynamics used."
    ),
    "requires": ["Close"],
    "produces": [
        "hmm_ts_vol_22",
        "hmm_ts_vol_60",
        "hmm_ts_skew_22",
        "hmm_ts_skew_60",
        "hmm_ts_kurt_22",
        "hmm_ts_kurt_60",
        "hmm_vol_of_vol_60",
        "hmm_skew_change_22",
    ],
    "tags": ["volatility", "momentum", "experimental"],
    "version": "1.0",
    "author": (
        "paper:10.1002/rfe.1121 (per-ticker time-series proxy — cross-sectional "
        "moments across stocks dropped; single-ticker rolling distributional dynamics used)"
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling realized higher moments per-ticker.

    Uses log-returns. All windows are causal (rolling backward only).
    Leading NaNs during warm-up are expected and not filled.
    """
    close = df["Close"].astype(np.float64)
    ret   = np.log(close / close.shift(1))

    # ---- Realized volatility (annualized std) --------------------------------
    df["hmm_ts_vol_22"] = ret.rolling(22, min_periods=10).std() * np.sqrt(252)
    df["hmm_ts_vol_60"] = ret.rolling(60, min_periods=20).std() * np.sqrt(252)

    # ---- Realized skewness ---------------------------------------------------
    # scipy not used to avoid pandas dependency on it; compute from rolling moments
    # skew = E[(r - mu)^3] / sigma^3
    def _rolling_skew(r: pd.Series, window: int, min_p: int) -> pd.Series:
        mu    = r.rolling(window, min_periods=min_p).mean()
        sigma = r.rolling(window, min_periods=min_p).std()
        # center the returns within each window
        cube  = (r - mu) ** 3
        m3    = cube.rolling(window, min_periods=min_p).mean()
        return m3 / (sigma ** 3).replace(0, np.nan)

    df["hmm_ts_skew_22"] = _rolling_skew(ret, 22, 10)
    df["hmm_ts_skew_60"] = _rolling_skew(ret, 60, 20)

    # ---- Realized excess kurtosis -------------------------------------------
    # kurt = E[(r - mu)^4] / sigma^4 - 3
    def _rolling_kurt(r: pd.Series, window: int, min_p: int) -> pd.Series:
        mu    = r.rolling(window, min_periods=min_p).mean()
        sigma = r.rolling(window, min_periods=min_p).std()
        quad  = (r - mu) ** 4
        m4    = quad.rolling(window, min_periods=min_p).mean()
        return m4 / (sigma ** 4).replace(0, np.nan) - 3.0

    df["hmm_ts_kurt_22"] = _rolling_kurt(ret, 22, 10)
    df["hmm_ts_kurt_60"] = _rolling_kurt(ret, 60, 20)

    # ---- Vol-of-vol (proxy for cross-sectional vol dynamics) ----------------
    # Compute 10d vol rolling, then take 60d std of that series
    vol_10d = ret.rolling(10, min_periods=5).std() * np.sqrt(252)
    df["hmm_vol_of_vol_60"] = vol_10d.rolling(60, min_periods=20).std()

    # ---- Skewness momentum (first difference of 22d skew) -------------------
    df["hmm_skew_change_22"] = df["hmm_ts_skew_22"].diff()

    return df
