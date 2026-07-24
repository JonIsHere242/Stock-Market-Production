from __future__ import annotations
import pandas as pd
import numpy as np

METADATA = {
    "name": "ff06282352g_distribution_left_right_tail_var_gap_120",
    "description": (
        "Tail-size asymmetry over a trailing 120-day window. "
        "Left-tail VaR = abs(5th-percentile of returns); right-tail VaR = 95th-percentile. "
        "Produces the gap (left_var - right_var) and the ratio left_var/(right_var+eps). "
        "Negative gap means left tail dominates (downside heavier than upside). "
        "Requires >= 30 observations; leading NaNs expected. Per-ticker proxy."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282352g_distribution_left_right_tail_var_gap_120_gap",
        "ff06282352g_distribution_left_right_tail_var_gap_120_ratio",
    ],
    "tags": ["distribution", "tail", "asymmetry", "var", "risk"],
    "version": "1.0.0",
    "author": "feature-factory ff06282352g",
}

_WINDOW = 120
_MIN_OBS = 30
_EPS = 1e-8

_COL_GAP = "ff06282352g_distribution_left_right_tail_var_gap_120_gap"
_COL_RATIO = "ff06282352g_distribution_left_right_tail_var_gap_120_ratio"


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN so every code path produces the columns.
    df[_COL_GAP] = np.nan
    df[_COL_RATIO] = np.nan

    if len(df) < 2:
        return df

    ret = df["Close"].pct_change()  # log1p equivalent fine at daily scale

    # Rolling 5th and 95th percentile over _WINDOW bars, requiring _MIN_OBS.
    left_var = (
        ret.rolling(window=_WINDOW, min_periods=_MIN_OBS)
        .quantile(0.05)
        .abs()          # left-tail VaR = magnitude of downside extreme
    )
    right_var = (
        ret.rolling(window=_WINDOW, min_periods=_MIN_OBS)
        .quantile(0.95)
    )

    # Gap: positive  => left tail larger (fat downside)
    #       negative => right tail larger (fat upside)
    gap = left_var - right_var

    # Ratio: >1 => downside heavier; <1 => upside heavier
    ratio = left_var / (right_var.abs() + _EPS)

    # Guard: if right_var itself is negative (unusual but possible in strong
    # trending stocks) the ratio still makes sense via abs() denominator;
    # replace inf/-inf with NaN just in case.
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df[_COL_GAP] = gap.values
    df[_COL_RATIO] = ratio.values

    return df
