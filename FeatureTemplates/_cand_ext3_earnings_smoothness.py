"""
Earnings Smoothness (management proxy) — ext3_earnings_smoothness
Ratio of net_income_ttm volatility to operating_cash_flow_ttm volatility over a
trailing window.  A LOW ratio signals smoothed/managed earnings (net income is
suspiciously stable relative to cash-flow volatility), an earnings-management
red flag.  Produces level ratio + 4-quarter trend.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_earnings_smoothness",
    "description": (
        "Earnings smoothness ratio: std(net_income_ttm) / std(operating_cash_flow_ttm) "
        "over a rolling 8-quarter (~504 trading-day) window.  A low ratio indicates "
        "that accruals are being used to dampen income volatility relative to cash "
        "flows — a classical earnings-management red flag.  Produces (1) the ratio "
        "itself (ext3_earnings_smoothness_ratio) and (2) its 4-quarter rolling slope "
        "(ext3_earnings_smoothness_trend).  Implemented per-ticker via PIT "
        "_fundamentals.as_of; ~84% coverage (ETFs/foreign = NaN)."
    ),
    "requires": ["Close"],          # Close needed only so merge_asof has a valid key
    "produces": [
        "ext3_earnings_smoothness_ratio",
        "ext3_earnings_smoothness_trend",
    ],
    "tags": ["fundamentals", "earnings_quality", "accruals", "smoothness"],
    "version": "1.0",
    "author": "Round-4 expansion (osap_orgcap)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ~8 quarters of trading days; rolling std window for the ratio
_RATIO_WIN = 504
# ~4 quarters for trend estimation
_TREND_WIN = 252


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns [Date, Ticker, Open, High, Low, Close, Volume],
         one ticker, ascending by Date (~700 rows typical).

    Returns
    -------
    df with two new columns appended (may be all-NaN for ETFs/foreign).
    """
    # Pull PIT fundamentals — backward merge on filed_date (no lookahead)
    df = _fundamentals.as_of(df, fields=["net_income_ttm", "operating_cash_flow_ttm"])

    ni = df["fund_net_income_ttm"].astype(float)
    ocf = df["fund_operating_cash_flow_ttm"].astype(float)

    # ------------------------------------------------------------------
    # Rolling std of each series (min_periods = 4 to avoid noise on tiny windows)
    # ------------------------------------------------------------------
    std_ni = ni.rolling(window=_RATIO_WIN, min_periods=4).std()
    std_ocf = ocf.rolling(window=_RATIO_WIN, min_periods=4).std()

    # Smoothness ratio: std(NI_ttm) / std(OCF_ttm)
    # Guard: denominator zero or NaN → NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = np.where(
            (std_ocf.isna()) | (std_ocf == 0.0),
            np.nan,
            std_ni / std_ocf,
        )
    ratio = pd.Series(ratio, index=df.index, dtype=float)

    # ------------------------------------------------------------------
    # Trend: linear slope of ratio over trailing _TREND_WIN bars
    # We approximate slope via the difference between two rolling means
    # (mean of second half minus mean of first half), which is a fast
    # look-back proxy for the OLS slope direction and is fully vectorised.
    # ------------------------------------------------------------------
    half = _TREND_WIN // 2

    # Mean of ratio over the most-recent half window
    rm_recent = ratio.rolling(window=half, min_periods=2).mean()
    # Mean of ratio lagged by half-window (the earlier half)
    rm_earlier = ratio.shift(half).rolling(window=half, min_periods=2).mean()

    # slope proxy = (recent_mean - earlier_mean) / half  (units: ratio / bar)
    trend = (rm_recent - rm_earlier) / half

    # ------------------------------------------------------------------
    # Assign produced columns
    # ------------------------------------------------------------------
    df["ext3_earnings_smoothness_ratio"] = ratio
    df["ext3_earnings_smoothness_trend"] = trend

    # Drop scratch fund_ columns
    df.drop(
        columns=["fund_net_income_ttm", "fund_operating_cash_flow_ttm"],
        errors="ignore",
        inplace=True,
    )

    return df
