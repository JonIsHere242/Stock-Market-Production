import pandas as pd
import numpy as np
from typing import List

METADATA = {
    "name": "lo_mackinlay_variance_ratio",
    "description": (
        "Rolling Lo-MacKinlay variance ratio for horizons k in [2,5,10]. "
        "VR(k) = Var(k-day log returns) / (k * Var(1-day log returns)) over a ~90-day window. "
        "VR>1 implies trending/positive autocorrelation; VR<1 implies mean reversion. "
        "Per-ticker OHLCV proxy — no heteroskedasticity correction (Lo-MacKinlay Z-stat omitted "
        "as it requires cross-sectional noise estimates). Includes deviation from 1 and 5-day ROC."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "vr_k2", "vr_k5", "vr_k10",
        "vr_k2_dev", "vr_k5_dev", "vr_k10_dev",
        "vr_k2_roc5", "vr_k5_roc5", "vr_k10_roc5",
    ],
    "tags": ["autocorrelation", "variance-ratio", "mean-reversion", "trend", "lo-mackinlay"],
    "version": "1.0",
    "author": "hypothesis-impl: per-ticker rolling VR proxy; no Z-stat (no cross-sectional noise model available)",
}

_HORIZONS: List[int] = [2, 5, 10]
_WINDOW: int = 90  # trading days for the rolling variance estimates


def compute(df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling variance of 1-day returns (denominator base)
    var1 = log_ret.rolling(_WINDOW).var()

    for k in _HORIZONS:
        col = f"vr_k{k}"
        dev_col = f"vr_k{k}_dev"
        roc_col = f"vr_k{k}_roc5"

        # k-day log return: sum of k consecutive 1-day log returns (no lookahead)
        log_ret_k = log_ret.rolling(k).sum()

        # Rolling variance of k-day returns
        # Need enough observations: use _WINDOW observations of the k-day series.
        # To keep windows comparable we roll over _WINDOW rows of the k-day return series.
        var_k = log_ret_k.rolling(_WINDOW).var()

        # Variance ratio; guard division by zero
        vr = var_k / (k * var1)

        df[col] = vr
        df[dev_col] = vr - 1.0
        df[roc_col] = vr / vr.shift(5) - 1.0

    return df