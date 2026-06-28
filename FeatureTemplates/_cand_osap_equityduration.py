"""
Equity Duration (Dechow, Sloan & Soliman 2004) — per-ticker proxy.

Equity Duration measures the cash-flow-weighted average time until a stock's
value is realised, analogous to bond duration. High-duration stocks have most
of their value in distant future cash flows (growth stocks); low-duration
stocks earn value in the near term (value stocks). Predicted sign = -1
(long low-duration / short high-duration).

Per-ticker proxy:
  The canonical cross-sectional implementation uses a multi-stage residual
  income model calibrated on the full panel. Here we implement a faithful
  per-ticker version using PIT fundamentals following the spirit of DSS 2004:

  Step 1: Estimate near-term "book yield" = book_value_per_share / Close.
          A high book yield implies the stock's intrinsic value is anchored
          in existing assets (short duration); a low book yield implies value
          must come from distant future earnings (long duration).

  Step 2: Estimate earnings yield = eps_basic_ttm / Close.
          Near-term profitability further shortens duration.

  Step 3: Duration proxy = 1 / (book_yield + earnings_yield) clamped > 0.
          This collapses to a harmonic mean of the two yield measures, which
          is the dominant identifiable component of DSS duration in per-ticker
          OHLCV+fundamentals data.

  Additionally we produce a rolling 252-day z-score (level vs own history,
  capturing regime shifts) and a 63-day slope (rate of change).

Cross-sectional note: True DSS duration requires ranking across many stocks
  to assign the "implied growth" component and uses a panel regression.
  The per-ticker proxy here captures the same economic intuition — near vs
  far-future cash-flow weight — but is NOT the panel rank statistic.
  Expect lower but non-zero IC vs the canonical.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT-fundamentals helper
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
    "name": "osap_equityduration",
    "description": (
        "Equity Duration per-ticker proxy (Dechow, Sloan & Soliman 2004 / "
        "OpenSourceAP Chen-Zimmermann). "
        "duration_level = 1 / (book_yield + earnings_yield) using PIT "
        "book_value_per_share and eps_basic_ttm. High value = growth stock "
        "(long duration); low value = value stock (short duration). "
        "Predicted sign cross-sectionally = -1 (long low-duration). "
        "Cross-sectional panel-regression component omitted (per-ticker proxy)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_equityduration_level",    # raw duration proxy (lower = value)
        "osap_equityduration_zscore",   # 252-day rolling z-score vs own history
        "osap_equityduration_slope",    # 63-day rate of change (momentum in duration)
    ],
    "tags": ["valuation", "fundamentals", "duration", "accounting", "osap"],
    "version": "1.0",
    "author": "Dechow, Sloan & Soliman (2004); OpenSourceAP (Chen-Zimmermann); per-ticker proxy implementation",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Equity Duration per-ticker proxy using PIT fundamentals."""

    # ---- Pull PIT fundamentals (backward-safe merge_asof) -----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["book_value_per_share", "eps_basic_ttm"])

    price = df["Close"].replace(0, np.nan)

    # ---- Book yield = B/P ---------------------------------------------------
    bvps = df.get("fund_book_value_per_share", pd.Series(np.nan, index=df.index))
    # Book value per share must be positive to be meaningful
    bvps_pos = bvps.where(bvps > 0, np.nan)
    book_yield = bvps_pos / price          # B/P; high -> short duration

    # ---- Earnings yield = E/P (TTM) ----------------------------------------
    eps_ttm = df.get("fund_eps_basic_ttm", pd.Series(np.nan, index=df.index))
    # Use only positive earnings (negative E/P would invert the measure)
    eps_pos = eps_ttm.where(eps_ttm > 0, np.nan)
    earn_yield = eps_pos / price           # E/P; high -> short duration

    # ---- Duration level proxy ----------------------------------------------
    # sum of yields: high sum -> near-term cashflows -> short duration (small value)
    # Clamp sum to avoid division by near-zero
    yield_sum = book_yield.add(earn_yield, fill_value=0.0)
    yield_sum = yield_sum.where(yield_sum > 1e-8, np.nan)
    duration_level = 1.0 / yield_sum      # larger = longer duration = growth stock

    df["osap_equityduration_level"] = duration_level

    # ---- Rolling 252-day z-score (duration relative to own history) --------
    roll_mean = duration_level.rolling(252, min_periods=63).mean()
    roll_std  = duration_level.rolling(252, min_periods=63).std()
    roll_std  = roll_std.where(roll_std > 1e-12, np.nan)
    df["osap_equityduration_zscore"] = (duration_level - roll_mean) / roll_std

    # ---- 63-day slope (rate of change in duration) -------------------------
    lag63 = duration_level.shift(63)
    lag63 = lag63.where(lag63.abs() > 1e-12, np.nan)
    df["osap_equityduration_slope"] = (duration_level - lag63) / lag63.abs()

    # Clamp slope to avoid extreme values from near-zero denominators
    df["osap_equityduration_slope"] = df["osap_equityduration_slope"].clip(-10, 10)

    # ---- Drop scratch fund_ columns not in produces -----------------------
    for col in ["fund_book_value_per_share", "fund_eps_basic_ttm"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
