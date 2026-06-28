"""
Candidate block: osap_divyieldst
Predicted dividend yield (short-term / next-month forecast), discretized.

Per-ticker proxy for Litzenberger & Ramaswamy (1979) / Chen-Zimmermann OpenSourceAP
signal "DivYieldST".  The original uses CRSP distribution codes (distcd) to identify
dividend type and select the appropriate lagged dividend observation (2-, 5-, or
11-months ago) as a predictor of the upcoming dividend.  Without distcd we proxy the
expected dividend using the trailing-twelve-month dividends paid (from PIT fundamentals)
amortised over 12 months to get an expected monthly dividend, then compute the ratio
against current price.  We also produce a slope variant (rate-of-change in yield) and
the raw continuous yield for model flexibility.  Discretization follows the paper's
buckets: 0 = zero yield, 1 = (0, 0.005], 2 = (0.005, 0.010], 3 = >0.010 (monthly).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fundamentals helper (PIT / backward merge)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA: dict = {
    "name": "osap_divyieldst",
    "description": (
        "Per-ticker proxy for the Litzenberger & Ramaswamy (1979) predicted "
        "short-term dividend yield signal from OpenSourceAP (Chen-Zimmermann). "
        "Original method uses CRSP distcd to select 2-, 5-, or 11-month-lagged "
        "dividends as next-month forecast; without distcd we use PIT fundamentals "
        "dividends_paid_ttm / 12 as the expected monthly dividend, then divide by "
        "Close to obtain an expected monthly yield.  Discretized into 4 buckets: "
        "0=zero, 1=(0,0.5%], 2=(0.5%,1.0%], 3=>1.0%.  Also produces the raw "
        "continuous yield and a 63-day slope of the yield for momentum.  "
        "Cross-sectional ranking is NOT applied here; this is a pure per-ticker signal."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_divyieldst_disc",   # discretized yield bucket 0-3 (main signal)
        "osap_divyieldst_raw",    # continuous expected monthly dividend yield
        "osap_divyieldst_slope",  # 63-day rolling slope of raw yield (dynamic)
    ],
    "tags": ["valuation", "dividend", "fundamentals", "accounting"],
    "version": "1.0",
    "author": (
        "Litzenberger and Ramaswamy (1979); OpenSourceAP / Chen-Zimmermann. "
        "Per-ticker proxy implementation (no CRSP distcd available)."
    ),
}


# ---------------------------------------------------------------------------
# Helper: robust rolling OLS slope (no sklearn)
# ---------------------------------------------------------------------------
def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Vectorised rolling linear slope via (Σxy - n*mean_x*mean_y) / (Σx² - n*mean_x²)."""
    n = window
    x = np.arange(n, dtype=np.float64)
    mean_x = x.mean()
    denom = float(((x - mean_x) ** 2).sum())
    if denom == 0:
        return pd.Series(np.nan, index=series.index)

    def _slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y = arr.astype(np.float64)
        mean_y = y.mean()
        return float(((x - mean_x) * (y - mean_y)).sum() / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # --- Pull PIT fundamentals ------------------------------------------------
    df = _fundamentals.as_of(df, fields=["dividends_paid_ttm"])

    # dividends_paid_ttm is reported as negative in cash-flow statements for many
    # data providers (cash outflow).  Take abs() to get the positive annual figure.
    annual_div = df["fund_dividends_paid_ttm"].abs()

    # Expected MONTHLY dividend = TTM / 12
    expected_monthly_div = annual_div / 12.0

    # Close price guard
    price = df["Close"].replace(0, np.nan)

    # --- Raw expected monthly dividend yield ----------------------------------
    raw_yield = expected_monthly_div / price

    # Clamp negatives to NaN (data anomalies) and cap at reasonable ceiling
    raw_yield = raw_yield.where(raw_yield >= 0, np.nan)
    raw_yield = raw_yield.where(raw_yield <= 1.0, np.nan)  # >100%/month is noise

    # Replace inf
    raw_yield = raw_yield.replace([np.inf, -np.inf], np.nan)

    # --- Discretize (paper buckets, monthly scale) ----------------------------
    # 0 = zero yield (no dividend)
    # 1 = (0, 0.005]   i.e. 0–0.5% per month
    # 2 = (0.005, 0.010] i.e. 0.5–1.0% per month
    # 3 = > 0.010       i.e. >1% per month
    disc = pd.Series(np.nan, index=df.index, dtype="float64")
    disc = disc.where(raw_yield.isna(), 0.0)                               # start at 0
    disc = disc.where(~(raw_yield > 0), 1.0)                               # bucket 1
    disc = disc.where(~(raw_yield > 0.005), 2.0)                           # bucket 2
    disc = disc.where(~(raw_yield > 0.010), 3.0)                           # bucket 3
    # Where raw_yield is NaN, disc should also be NaN
    disc = disc.where(raw_yield.notna(), np.nan)

    # --- Slope of raw yield over ~1 quarter (63 trading days) -----------------
    slope = _rolling_slope(raw_yield, window=63)
    slope = slope.replace([np.inf, -np.inf], np.nan)

    # --- Assign produced columns ----------------------------------------------
    df["osap_divyieldst_disc"] = disc
    df["osap_divyieldst_raw"] = raw_yield
    df["osap_divyieldst_slope"] = slope

    # Drop scratch fundamentals columns not in produces
    df = df.drop(columns=["fund_dividends_paid_ttm"], errors="ignore")

    return df
