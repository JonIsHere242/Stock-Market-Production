from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_semivariance_ratio",
    "description": (
        "Rolling downside/upside semivariance ratio (Markowitz semivariance). "
        "Computes the ratio of downside semideviation (RMS of negative daily returns) "
        "to upside semideviation (RMS of positive daily returns) over a 60-day window. "
        "Ratio > 1 means downside volatility dominates (crash-prone); ratio < 1 means "
        "upside volatility dominates. Also produces a 20-day short-window variant and a "
        "slope (60d ratio minus its own 20-day rolling mean) to capture regime change. "
        "Per-ticker time-series proxy for the Markowitz cross-sectional semivariance signal."
    ),
    "requires": ["Close"],
    "produces": [
        "xdom2_semivariance_ratio_60d",
        "xdom2_semivariance_ratio_20d",
        "xdom2_semivariance_ratio_slope",
    ],
    "tags": ["risk", "semivariance", "downside", "volatility", "asymmetry", "cross-domain"],
    "version": "1.0",
    "author": "Cross-domain / practitioner method transfer (batch 2); Markowitz semivariance",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling downside/upside semivariance ratios.

    Uses daily log returns; for each rolling window:
      - downside_semivol = sqrt(mean(r^2 for r < 0))
      - upside_semivol   = sqrt(mean(r^2 for r > 0))
      - ratio = downside_semivol / upside_semivol
    Windows with no negative or no positive return get NaN.
    """
    ret = np.log(df["Close"] / df["Close"].shift(1))

    n60 = 60
    n20 = 20

    def _semivar_ratio(returns: pd.Series, window: int) -> pd.Series:
        """Rolling semivariance ratio using numpy sliding windows."""
        arr = returns.to_numpy(dtype=np.float64)
        n = len(arr)
        out = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_slice = arr[i - window + 1 : i + 1]
            # Exclude leading NaN from log-return at row 0
            valid = window_slice[~np.isnan(window_slice)]
            if len(valid) < 2:
                continue
            neg = valid[valid < 0]
            pos = valid[valid > 0]
            if len(neg) == 0 or len(pos) == 0:
                continue
            down = np.sqrt(np.mean(neg ** 2))
            up = np.sqrt(np.mean(pos ** 2))
            if up == 0.0:
                out[i] = np.nan
            else:
                out[i] = down / up

        return pd.Series(out, index=returns.index)

    ratio_60 = _semivar_ratio(ret, n60)
    ratio_20 = _semivar_ratio(ret, n20)

    # Slope: current 60d ratio minus its rolling 20-day mean (trend in asymmetry)
    ratio_60_trend = ratio_60.rolling(window=20, min_periods=10).mean()
    slope = ratio_60 - ratio_60_trend

    df["xdom2_semivariance_ratio_60d"] = ratio_60
    df["xdom2_semivariance_ratio_20d"] = ratio_20
    df["xdom2_semivariance_ratio_slope"] = slope

    return df
