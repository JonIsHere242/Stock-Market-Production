"""
SG&A Efficiency Trend (Organizational Leverage)
Spec: ext_sga_efficiency_trend
Source: Extension/exploration of gate-validated winner (osap_orgcap)
"""
from __future__ import annotations
import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext_sga_efficiency_trend",
    "description": (
        "SG&A efficiency = revenue_ttm / SG&A_implied, where SG&A_implied is "
        "approximated as gross_profit_ttm - operating_income_ttm (the residual "
        "between gross profit and operating income, i.e. the SG&A + D&A wedge). "
        "Produces: (1) level ratio, (2) 252-day linear trend of that ratio "
        "(improving organizational leverage), and (3) gross_profit_ttm / SG&A_implied "
        "as an orthogonal angle. All computed per-ticker from PIT fundamentals; "
        "rising values signal better organizational capital utilization. "
        "Cross-sectional normalisation is not applied here (per-ticker proxy)."
    ),
    "requires": [],   # uses only PIT fundamentals + Date
    "produces": [
        "ext_sga_efficiency_trend_level",
        "ext_sga_efficiency_trend_slope",
        "ext_sga_efficiency_trend_gp_ratio",
    ],
    "tags": ["fundamentals", "organizational_capital", "efficiency", "sga", "trend"],
    "version": "1.0.0",
    "author": "Spec: Extension/exploration of osap_orgcap (gate-validated winner)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SG&A efficiency and its trend per ticker.

    SG&A_implied = gross_profit_ttm - operating_income_ttm
      (this is the SG&A + D&A wedge; a faithful PIT approximation since
       explicit SG&A is not in the fundamentals panel)

    Produced columns
    ----------------
    ext_sga_efficiency_trend_level    : revenue_ttm / SG&A_implied  (higher = more efficient)
    ext_sga_efficiency_trend_slope    : 252-bar rolling OLS slope of _level
    ext_sga_efficiency_trend_gp_ratio : gross_profit_ttm / SG&A_implied  (orthogonal angle)
    """
    # ---- pull PIT fundamentals ----
    df = _fundamentals.as_of(
        df,
        fields=["revenue_ttm", "gross_profit_ttm", "operating_income_ttm"],
    )

    rev = df["fund_revenue_ttm"]
    gp = df["fund_gross_profit_ttm"]
    oi = df["fund_operating_income_ttm"]

    # SG&A implied = gross_profit - operating_income  (always >= 0 for healthy co.)
    sga = gp - oi  # may be negative if unusual; we guard below

    # Guard: zero / negative SGA → NaN (can't interpret as efficiency denominator)
    sga_safe = sga.where(sga > 0, other=np.nan)

    # -- Level: revenue / SG&A_implied --
    level = rev / sga_safe  # NaN where coverage is missing

    # -- GP ratio: gross_profit / SG&A_implied --
    gp_ratio = gp / sga_safe

    # -- Trend: 252-bar rolling OLS slope of level --
    # Use a vectorised numpy approach via rolling apply on a small window.
    # The x-values are simply 0..n-1, which is equivalent to the slope of a
    # least-squares fit. We embed this in a tight rolling apply.
    WIN = 252

    level_arr = level.to_numpy(dtype=float)
    n = len(level_arr)
    slope_arr = np.full(n, np.nan)

    # Pre-compute OLS slope for a window of size w:
    #   slope = (n*sum(i*y) - sum(i)*sum(y)) / (n*sum(i^2) - sum(i)^2)
    # For window positions where there are enough non-NaN values (>=20).
    # We use pandas rolling with a raw numpy function to stay vectorised enough.
    def _ols_slope(y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        valid = mask.sum()
        if valid < 20:
            return np.nan
        idx = np.where(mask)[0].astype(float)
        yv = y[mask]
        n_ = float(valid)
        sx = idx.sum()
        sy = yv.sum()
        sxy = (idx * yv).sum()
        sxx = (idx * idx).sum()
        denom = n_ * sxx - sx * sx
        if denom == 0.0:
            return np.nan
        return (n_ * sxy - sx * sy) / denom

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope_series = (
            pd.Series(level_arr, index=df.index)
            .rolling(window=WIN, min_periods=20)
            .apply(_ols_slope, raw=True)
        )

    # ---- assign produced columns (never modify/drop existing) ----
    df["ext_sga_efficiency_trend_level"] = level.values
    df["ext_sga_efficiency_trend_slope"] = slope_series.values
    df["ext_sga_efficiency_trend_gp_ratio"] = gp_ratio.values

    # Drop scratch fund_ columns not in produces
    for col in ["fund_revenue_ttm", "fund_gross_profit_ttm", "fund_operating_income_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
