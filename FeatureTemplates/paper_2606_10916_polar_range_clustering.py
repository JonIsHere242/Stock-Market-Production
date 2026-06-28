"""
Polar Range Clustering Features  —  arxiv:2606.10916
"Range Penalization: Theoretical Insights with Applications in Federated Learning"

The paper introduces range regularization that induces "polar clustering" — weights
adaptively cluster at extreme (polar) values. Features with shared weights are
regularized toward center; personalized features cluster at extremes.

OHLCV translation:
  The COMPUTABLE METHOD is POLAR CLUSTERING of a time series:
  identifying which observations fall near the extreme poles of a rolling
  distribution vs. which cluster near center.

  Applied to OHLCV:
    - Define the "polar zone" as observations in the top or bottom quintile
      of the rolling distribution over W bars
    - Track the density and persistence of polar-zone occupancy
    - Measure the "range penalty" — the ratio of range to IQR (extreme-sensitivity)
    - Detect when price breaks out of the central "shared" zone into polar territory
    - Multi-window: 20, 40, 60 bars

  Features:
    - polar_upper_density_20: fraction of last 20 bars where Close was in upper quintile
    - polar_lower_density_20: fraction of last 20 bars in lower quintile
    - polar_range_iqr_ratio_20: rolling range / IQR (range-to-interquartile ratio)
    - polar_range_iqr_ratio_60: same, 60-bar
    - polar_center_escape_20: signed breakout from central band [Q25, Q75]
    - polar_persistence_streak: days continuously in polar zone (upper or lower)
    - polar_vol_asymmetry: ratio of upper-polar return std to lower-polar return std

  Produces 7 columns prefixed "polar_".
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_10916_polar_range_clustering",
    "description": (
        "Polar-clustering and range-penalization features: rolling polar-zone "
        "occupancy, range/IQR ratios, and breakout intensity; inspired by "
        "range regularization in arxiv 2606.10916 (Federated Learning range penalty)."
    ),
    "requires": ["Close", "High", "Low"],
    "produces": [
        "polar_upper_density_20",
        "polar_lower_density_20",
        "polar_range_iqr_ratio_20",
        "polar_range_iqr_ratio_60",
        "polar_center_escape_20",
        "polar_persistence_streak",
        "polar_vol_asymmetry",
        "polar_net_occupancy_10",
    ],
    "tags": ["momentum", "volatility", "experimental"],
    "version": "1.0",
    "author": "paper:2606.10916",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(close)

    close_s = pd.Series(close)

    # ---- Rolling quantiles for polar zone boundaries ---------------------------
    q20_20 = close_s.rolling(20, min_periods=6).quantile(0.20)
    q80_20 = close_s.rolling(20, min_periods=6).quantile(0.80)
    q25_20 = close_s.rolling(20, min_periods=6).quantile(0.25)
    q75_20 = close_s.rolling(20, min_periods=6).quantile(0.75)

    q20_60 = close_s.rolling(60, min_periods=15).quantile(0.20)
    q80_60 = close_s.rolling(60, min_periods=15).quantile(0.80)
    q25_60 = close_s.rolling(60, min_periods=15).quantile(0.25)
    q75_60 = close_s.rolling(60, min_periods=15).quantile(0.75)

    roll_max_20 = close_s.rolling(20, min_periods=6).max()
    roll_min_20 = close_s.rolling(20, min_periods=6).min()
    roll_max_60 = close_s.rolling(60, min_periods=15).max()
    roll_min_60 = close_s.rolling(60, min_periods=15).min()

    # ---- Upper and lower polar occupancy (fraction in polar zone) ---------------
    in_upper_20 = (close_s > q80_20).astype(float)
    in_lower_20 = (close_s < q20_20).astype(float)

    df["polar_upper_density_20"] = in_upper_20.rolling(20, min_periods=6).mean()
    df["polar_lower_density_20"] = in_lower_20.rolling(20, min_periods=6).mean()

    # ---- Range / IQR ratio: extreme sensitivity metric -------------------------
    range20 = roll_max_20 - roll_min_20
    iqr20 = (q75_20 - q25_20).replace(0, np.nan)
    df["polar_range_iqr_ratio_20"] = range20 / iqr20

    range60 = roll_max_60 - roll_min_60
    iqr60 = (q75_60 - q25_60).replace(0, np.nan)
    df["polar_range_iqr_ratio_60"] = range60 / iqr60

    # ---- Center escape: signed distance from central band ----------------------
    # Positive = above upper boundary Q75, Negative = below lower boundary Q25
    # Zero = inside central [Q25, Q75] band
    above = (close_s - q75_20).clip(lower=0)   # positive when above upper center
    below = (q25_20 - close_s).clip(lower=0)   # positive when below lower center
    # Normalize by current IQR
    escape = (above - below) / iqr20.replace(0, np.nan)
    df["polar_center_escape_20"] = escape

    # ---- Persistence streak: days continuously in any polar zone (upper or lower) -
    in_polar_20 = ((close_s > q80_20) | (close_s < q20_20)).astype(float)
    in_polar_arr = in_polar_20.values
    q20_arr = q20_20.values
    streak = np.zeros(n)
    for t in range(1, n):
        ip = in_polar_arr[t]
        if np.isnan(ip) or np.isnan(q20_arr[t]):
            streak[t] = np.nan
        elif ip == 1.0:
            streak[t] = streak[t - 1] + 1 if np.isfinite(streak[t - 1]) else 1
        else:
            streak[t] = 0.0
    df["polar_persistence_streak"] = streak

    # ---- Vol asymmetry: std(returns | upper polar) / std(returns | lower polar) -
    log_ret = np.empty(n)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan))

    # Rolling vol asymmetry over 60-bar windows
    vol_asym = np.full(n, np.nan)
    if n >= 60:
        # Build sliding 60-bar windows once and batch the expensive quantiles.
        seg_close_mat = np.lib.stride_tricks.sliding_window_view(close, 60)
        seg_ret_mat = np.lib.stride_tricks.sliding_window_view(log_ret, 60)
        finite_mat = np.isfinite(seg_ret_mat)
        # q80/q20 per window (close has no NaN, so this matches np.nanquantile)
        q80_all = np.nanquantile(seg_close_mat, 0.80, axis=1)
        q20_all = np.nanquantile(seg_close_mat, 0.20, axis=1)
        finite_counts = finite_mat.sum(axis=1)
        for i in range(seg_close_mat.shape[0]):
            if finite_counts[i] < 20:
                continue
            seg_close = seg_close_mat[i]
            seg_ret = seg_ret_mat[i]
            fin = finite_mat[i]
            ret_upper = seg_ret[(seg_close >= q80_all[i]) & fin]
            ret_lower = seg_ret[(seg_close <= q20_all[i]) & fin]
            if len(ret_upper) >= 3 and len(ret_lower) >= 3:
                std_upper = np.std(ret_upper)
                std_lower = np.std(ret_lower)
                if std_lower > 1e-10:
                    vol_asym[i + 59] = std_upper / std_lower

    df["polar_vol_asymmetry"] = vol_asym

    # ---- Net polar occupancy: lower zone - upper zone over 10-bar window -------
    # Positive = more time in lower polar zone than upper → mean-reversion up signal
    # Negative = more time in upper polar zone → overbought, potential reversal down
    df["polar_net_occupancy_10"] = (
        in_lower_20.rolling(10, min_periods=3).mean()
        - in_upper_20.rolling(10, min_periods=3).mean()
    )

    return df
