"""
osap_investment — Investment-to-revenue ratio scaled by 36-month rolling mean.

Based on Titman, Wei & Xie (2004) via Chen-Zimmermann OpenSourceAP.
Predicted sign: -1 (overinvestment → lower future returns).

Per-ticker implementation using PIT fundamentals (capex_ttm, revenue_ttm).
Cross-sectional ranking (original paper) is approximated by the within-ticker
time-series ratio vs. its own 36-month trailing mean, which captures the same
"abnormal investment" economic signal on a per-stock basis.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
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
    "name": "osap_investment",
    "description": (
        "Investment-to-revenue ratio (capex_ttm / revenue_ttm) relative to its own "
        "36-month (approx 756-trading-day) rolling mean. Based on Titman, Wei & Xie "
        "(2004); overinvestment signal. Predicted sign: -1 (high ratio → lower returns). "
        "Original paper is cross-sectional; this is a per-ticker time-series proxy "
        "capturing the same 'abnormal capital expenditure' economic signal. "
        "Observations with revenue_ttm < $10M are set to NaN per the spec exclusion."
    ),
    "requires": [],  # uses only PIT fundamentals; no raw OHLCV required
    "produces": [
        "osap_investment_ratio",       # raw capex_ttm / revenue_ttm
        "osap_investment_scaled",      # ratio / 36m rolling mean  (the core signal)
        "osap_investment_zscore",      # z-score vs 36m window (alternative normalisation)
    ],
    "tags": ["investment", "fundamentals", "accounting", "capex", "overinvestment"],
    "version": "1.0.0",
    "author": "Titman, Wei & Xie (2004); OpenSourceAP / Chen-Zimmermann; block by Claude",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 36 months ≈ 756 trading days; we work on a calendar-day merge so we use a
# window expressed in *observations* after the merge — each row is one trading
# day so 756 rows ≈ 3 years.
_ROLLING_WINDOW = 756        # trading-day rows ≈ 36 months
_MIN_PERIODS    = 126        # require ≥ 6 months of data before emitting a value
_REVENUE_MIN    = 10_000_000  # $10M revenue exclusion per spec


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add osap_investment_* columns to a single-ticker OHLCV DataFrame."""

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["capex_ttm", "revenue_ttm"])

    capex   = df["fund_capex_ttm"].copy()
    revenue = df["fund_revenue_ttm"].copy()

    # ------------------------------------------------------------------
    # 2. Raw ratio (cap-ex / revenue) — negative capex is a cash outflow
    #    convention in some filings; take abs so the ratio is always positive.
    # ------------------------------------------------------------------
    # Guard: revenue < $10M → exclude per spec
    revenue_valid = revenue.where(revenue.abs() >= _REVENUE_MIN)

    # capex is typically reported as negative in cash-flow statements
    capex_abs = capex.abs()

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(
            revenue_valid.isna() | (revenue_valid == 0),
            np.nan,
            capex_abs / revenue_valid.abs(),
        )
    raw_ratio = pd.Series(raw_ratio, index=df.index)

    # Replace inf with NaN
    raw_ratio = raw_ratio.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 3. 36-month rolling mean of the ratio (the denominator in the spec)
    # ------------------------------------------------------------------
    rolling_mean = raw_ratio.rolling(window=_ROLLING_WINDOW, min_periods=_MIN_PERIODS).mean()
    rolling_std  = raw_ratio.rolling(window=_ROLLING_WINDOW, min_periods=_MIN_PERIODS).std()

    # ------------------------------------------------------------------
    # 4. Scaled ratio: current ratio / rolling mean  (core signal per spec)
    #    Values > 1 = overinvestment relative to own history.
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(
            rolling_mean.isna() | (rolling_mean == 0),
            np.nan,
            raw_ratio / rolling_mean,
        )
    scaled = pd.Series(scaled, index=df.index).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 5. Z-score variant: (ratio - mean) / std  (alternative normalisation)
    # ------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(
            rolling_std.isna() | (rolling_std == 0),
            np.nan,
            (raw_ratio - rolling_mean) / rolling_std,
        )
    zscore = pd.Series(zscore, index=df.index).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # 6. Assign produced columns
    # ------------------------------------------------------------------
    df["osap_investment_ratio"]  = raw_ratio.values
    df["osap_investment_scaled"] = scaled.values
    df["osap_investment_zscore"] = zscore.values

    # ------------------------------------------------------------------
    # 7. Drop scratch fundamentals columns (not listed in produces)
    # ------------------------------------------------------------------
    for col in ["fund_capex_ttm", "fund_revenue_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
