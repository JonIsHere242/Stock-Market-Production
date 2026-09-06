from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ff06282347f_distribution_conditional_skew_downup_60",
    "description": (
        "Conditional skewness asymmetry over trailing 60 days. "
        "Splits daily returns into down-days (return<0) and up-days (return>0), "
        "computes sample skewness of each subset separately, then emits: "
        "(1) down-day skewness, (2) the difference down_skew - up_skew as an "
        "asymmetry signal. Requires >=10 observations in each subset, else NaN. "
        "Per-ticker proxy; causal, no lookahead."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282347f_distribution_conditional_skew_downup_60_down_skew",
        "ff06282347f_distribution_conditional_skew_downup_60_asym",
    ],
    "tags": ["distribution", "skewness", "asymmetry", "returns", "tail"],
    "version": "1.0.0",
    "author": "feature-factory:ff06282347f",
}

_COL_DOWN = "ff06282347f_distribution_conditional_skew_downup_60_down_skew"
_COL_ASYM = "ff06282347f_distribution_conditional_skew_downup_60_asym"
_WINDOW = 60
_MIN_OBS = 10


def _sample_skew(arr: np.ndarray) -> float:
    """Unbiased sample skewness (same as scipy.stats.skew bias=False)."""
    n = len(arr)
    if n < 3:
        return np.nan
    mu = arr.mean()
    diff = arr - mu
    s2 = (diff ** 2).sum() / (n - 1)
    if s2 == 0.0:
        return np.nan
    s = np.sqrt(s2)
    m3 = (diff ** 3).sum() / n
    # bias correction: multiply by n*(n-1)^0.5 / (n-2)
    skew = (m3 / (s ** 3)) * (n * (n - 1) ** 0.5) / (n - 2)
    return skew


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise produced columns to NaN on every code path
    df[_COL_DOWN] = np.nan
    df[_COL_ASYM] = np.nan

    if len(df) < 2:
        return df

    ret = df["Close"].pct_change().to_numpy(dtype=float)
    n = len(ret)

    down_skew_vals = np.full(n, np.nan)
    asym_vals = np.full(n, np.nan)

    for i in range(_WINDOW - 1, n):
        window = ret[i - _WINDOW + 1: i + 1]  # shape (_WINDOW,)
        # exclude the first element which is NaN (pct_change at index 0)
        window = window[~np.isnan(window)]

        down = window[window < 0.0]
        up = window[window > 0.0]

        if len(down) < _MIN_OBS or len(up) < _MIN_OBS:
            continue

        ds = _sample_skew(down)
        us = _sample_skew(up)

        if not (np.isnan(ds) or np.isnan(us)):
            down_skew_vals[i] = ds
            asym_vals[i] = ds - us

    df[_COL_DOWN] = down_skew_vals
    df[_COL_ASYM] = asym_vals

    return df
