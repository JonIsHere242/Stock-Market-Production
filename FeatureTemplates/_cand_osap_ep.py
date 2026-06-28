"""
Earnings-to-Price Ratio (E/P) — Basu 1977 / OpenSourceAP (Chen-Zimmermann).

Per-ticker proxy: TTM net income (PIT-filed) divided by lagged market equity
(shares_outstanding * Close lagged ~126 trading days ≈ 6 calendar months).
Original paper used Dec-31 market equity and NYSE-only universe; we simulate
the 6-month lag on every ticker and drop rows where EP<0 (negative earnings
stocks get NaN per the original exclusion rule).

Cross-sectional note: the paper's return predictability is XS (rank within
universe). This block outputs the raw E/P ratio, a 63-day rolling trend
(slope of E/P), and a 126-day z-score so the downstream model can apply
its own XS ranking.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# --- load PIT fundamentals helper ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_ep",
    "description": (
        "Earnings-to-Price ratio (Basu 1977). "
        "TTM net income (PIT via filing date) divided by market equity lagged "
        "~126 trading days (~6 calendar months), approximating the original "
        "paper's use of Dec-31 lagged market cap. Negative E/P rows set to NaN "
        "per original exclusion rule. "
        "Per-ticker proxy — cross-sectional ranking is left to the model."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_ep_ratio",     # TTM E/P with 6m-lagged market cap
        "osap_ep_zscore",    # 126-day rolling z-score of the ratio
        "osap_ep_slope",     # 63-day rolling OLS slope (trend in E/P)
    ],
    "tags": ["valuation", "fundamentals", "earnings", "basu1977", "openSourceAP"],
    "version": "1.0",
    "author": "Basu 1977 / OpenSourceAP Chen-Zimmermann; block by Claude",
}

# Lag in trading days approximating 6 calendar months
_LAG_TD = 126
# Rolling window for z-score and slope
_Z_WIN = 126
_SLOPE_WIN = 63


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- Pull PIT fundamentals ---
    df = _fundamentals.as_of(df, fields=["net_income_ttm", "shares_outstanding"])

    # --- Market equity lagged ~6 months ---
    close_lagged = df["Close"].shift(_LAG_TD)
    shares = df["fund_shares_outstanding"]  # PIT shares (filed date)

    # Market equity = shares * lagged close
    mkt_eq = shares * close_lagged  # units: shares * $ = $

    # Guard against zero / missing market equity
    mkt_eq_safe = mkt_eq.where(mkt_eq > 0, other=np.nan)

    # E/P ratio: TTM net income / lagged market equity
    # net_income_ttm is in $ (same currency as Close * shares if shares in units)
    ep = df["fund_net_income_ttm"] / mkt_eq_safe

    # Original paper excludes negative earnings (EP < 0 -> NaN)
    ep = ep.where(ep >= 0, other=np.nan)

    df["osap_ep_ratio"] = ep

    # --- 126-day rolling z-score ---
    ep_mean = ep.rolling(_Z_WIN, min_periods=_Z_WIN // 2).mean()
    ep_std = ep.rolling(_Z_WIN, min_periods=_Z_WIN // 2).std()
    ep_std_safe = ep_std.where(ep_std > 0, other=np.nan)
    df["osap_ep_zscore"] = (ep - ep_mean) / ep_std_safe

    # --- 63-day rolling OLS slope (rate of change in E/P) ---
    # Use numpy polyfit via apply on rolling window; vectorised via stride trick
    n = _SLOPE_WIN

    def _slope(arr: np.ndarray) -> float:
        valid = ~np.isnan(arr)
        if valid.sum() < max(n // 2, 2):
            return np.nan
        x = np.arange(len(arr), dtype=np.float64)[valid]
        y = arr[valid]
        if x.std() == 0:
            return np.nan
        # OLS slope = cov(x,y)/var(x)
        x_dm = x - x.mean()
        denom = (x_dm ** 2).sum()
        if denom == 0:
            return np.nan
        return float((x_dm * (y - y.mean())).sum() / denom)

    ep_arr = ep.to_numpy(dtype=np.float64)
    slopes = np.full(len(ep_arr), np.nan)
    for i in range(n - 1, len(ep_arr)):
        slopes[i] = _slope(ep_arr[i - n + 1: i + 1])

    df["osap_ep_slope"] = slopes

    # Drop scratch fund_ columns not in produces
    for col in ["fund_net_income_ttm", "fund_shares_outstanding"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
