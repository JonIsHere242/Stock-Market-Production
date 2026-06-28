"""
Allan deviation of signed order-flow (per-ticker proxy).

Spec: ext2_allan_signed_flow
Source: Round-3 deep exploration of a rich winner vein (xdom_allan_variance).
Extends parent feature: xdom_allan_variance.

Method:
  sv_t = sign(daily return) * detrended log-volume
        where detrended = log(Volume) - rolling_5d_mean(log(Volume))

  Allan deviation at tau=5 over an 80d rolling window:
    - Split the 80 most-recent bars into non-overlapping 5-bar blocks (16 blocks).
    - Compute block means of sv_t.
    - sigma_A = sqrt(0.5 * mean(diff(block_means)^2))

  Produces:
    ext2_allan_signed_flow_dev  : rolling Allan deviation of signed flow
    ext2_allan_signed_flow_zscore : 20d z-score of the Allan deviation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "ext2_allan_signed_flow",
    "description": (
        "Allan deviation (tau=5, 80d rolling window) of signed order-flow "
        "sv_t = sign(return) * detrended_log_volume. Measures frequency-stability "
        "and drift character of directional trading flow. Per-ticker proxy; no "
        "cross-sectional component. Produces the Allan deviation and its 20d z-score."
    ),
    "requires": ["Close", "Volume"],
    "produces": [
        "ext2_allan_signed_flow_dev",
        "ext2_allan_signed_flow_zscore",
    ],
    "tags": ["volume", "flow", "allan_deviation", "volatility", "stability"],
    "version": "1.0.0",
    "author": "Spec: Round-3 deep exploration of xdom_allan_variance vein",
}

_TAU = 5          # block size
_WIN = 80         # rolling window (must be divisible by _TAU => 16 blocks)
_N_BLOCKS = _WIN // _TAU   # 16
_ZSCORE_WIN = 20


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    # ------------------------------------------------------------------ #
    # 1. Build signed-flow series                                          #
    # ------------------------------------------------------------------ #
    close = df["Close"].to_numpy(dtype=np.float64)
    volume = df["Volume"].to_numpy(dtype=np.float64)

    # Daily return (shift forward by 1 so index t uses Close[t-1])
    # ret[0] = NaN because no prior bar
    ret = np.empty(n, dtype=np.float64)
    ret[0] = np.nan
    ret[1:] = close[1:] / close[:-1] - 1.0

    sign_ret = np.sign(ret)  # -1, 0, or +1; NaN propagates as 0 from np.sign

    # Log volume (guard zero volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.where(volume > 0, np.log(volume), np.nan)

    # Detrended log-volume: subtract rolling 5d mean (causal, min_periods=1)
    log_vol_s = pd.Series(log_vol)
    log_vol_5d_mean = log_vol_s.rolling(window=5, min_periods=1).mean().to_numpy()
    detrended_lv = log_vol - log_vol_5d_mean

    # Signed flow
    sv = sign_ret * detrended_lv  # NaN where ret is NaN

    # ------------------------------------------------------------------ #
    # 2. Rolling Allan deviation over 80 bars, tau=5                      #
    # ------------------------------------------------------------------ #
    # We need exactly _WIN = 80 bars to form _N_BLOCKS = 16 non-overlapping
    # blocks of size _TAU = 5. Compute for each t >= _WIN-1.
    # Vectorise with numpy sliding_window_view.

    allan_dev = np.full(n, np.nan, dtype=np.float64)

    if n >= _WIN:
        from numpy.lib.stride_tricks import sliding_window_view

        sv_mat = sliding_window_view(sv, _WIN)  # shape (n - _WIN + 1, _WIN)
        # sv_mat[i] = bars [i .. i+_WIN-1]
        # Reshape into blocks: (num_windows, _N_BLOCKS, _TAU)
        sv_blocks = sv_mat.reshape(sv_mat.shape[0], _N_BLOCKS, _TAU)
        # Block means: (num_windows, _N_BLOCKS)
        block_means = np.nanmean(sv_blocks, axis=2)

        # Diff of block means along block axis
        diff_bm = np.diff(block_means, axis=1)  # shape (num_windows, _N_BLOCKS-1)

        # Allan deviation: sqrt(0.5 * mean(diff^2))
        with np.errstate(invalid="ignore"):
            mean_sq = np.nanmean(diff_bm ** 2, axis=1)
            adev = np.sqrt(0.5 * mean_sq)

        # Assign: first valid output index is _WIN - 1
        allan_dev[_WIN - 1:] = adev

    # ------------------------------------------------------------------ #
    # 3. 20d z-score of Allan deviation                                   #
    # ------------------------------------------------------------------ #
    adev_s = pd.Series(allan_dev)
    roll_mean = adev_s.rolling(window=_ZSCORE_WIN, min_periods=_ZSCORE_WIN).mean()
    roll_std = adev_s.rolling(window=_ZSCORE_WIN, min_periods=_ZSCORE_WIN).std(ddof=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = ((adev_s - roll_mean) / roll_std.replace(0, np.nan)).to_numpy()

    # ------------------------------------------------------------------ #
    # 4. Assign produced columns                                           #
    # ------------------------------------------------------------------ #
    df["ext2_allan_signed_flow_dev"] = allan_dev
    df["ext2_allan_signed_flow_zscore"] = zscore

    return df
