from __future__ import annotations

import numpy as np
import pandas as pd

METADATA = {
    "name": "xdom2_entropy_volume",
    "description": (
        "Rolling 60-day Shannon entropy of normalized daily volume shares (v_i / sum_v). "
        "Low entropy => volume concentrated on few days (event-driven / informed trading). "
        "High entropy => volume evenly spread (routine, uninformed flow). "
        "Produces: level (entropy_60), its 20-day change (delta_20), and a z-score of the "
        "level over the same 60-day window to make the signal stationary. "
        "Per-ticker time-series proxy -- no cross-sectional component."
    ),
    "requires": ["Volume"],
    "produces": [
        "xdom2_entropy_volume_60",
        "xdom2_entropy_volume_delta20",
        "xdom2_entropy_volume_zscore",
    ],
    "tags": ["volume", "entropy", "cross-domain", "information", "clustering"],
    "version": "1.0.0",
    "author": "Spec: Shannon entropy of the volume distribution — Cross-domain/practitioner method transfer (batch 2)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling Shannon entropy of the volume distribution."""
    window = 60
    delta_window = 20
    zscore_window = 126  # ~6-month z-score to centre the level signal

    vol = df["Volume"].to_numpy(dtype=np.float64)
    n = len(vol)

    entropy_vals = np.full(n, np.nan)

    if n >= window:
        for i in range(window - 1, n):
            block = vol[i - window + 1 : i + 1]
            # Replace non-positive with nan before summing
            block = np.where(block > 0, block, np.nan)
            total = np.nansum(block)
            if total <= 0 or np.isnan(total):
                # Can't normalise -- leave as NaN
                continue
            shares = block / total
            # Shannon entropy: -sum(p * log(p)), treat NaN shares as zero contribution
            log_shares = np.where(shares > 0, np.log(shares), 0.0)
            entropy_vals[i] = -np.nansum(shares * log_shares)

    df["xdom2_entropy_volume_60"] = entropy_vals

    # 20-day change in entropy level
    ent_series = pd.Series(entropy_vals, index=df.index)
    df["xdom2_entropy_volume_delta20"] = ent_series.diff(delta_window)

    # Rolling z-score of the entropy level (centre & scale over ~6 months)
    roll = ent_series.rolling(zscore_window, min_periods=window)
    roll_mean = roll.mean()
    roll_std = roll.std(ddof=1)
    zscore = (ent_series - roll_mean) / roll_std.replace(0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    df["xdom2_entropy_volume_zscore"] = zscore.to_numpy()

    return df
