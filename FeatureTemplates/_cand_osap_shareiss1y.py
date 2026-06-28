"""
Share issuance (1 year) — per-ticker proxy.

Economic signal: Share issuance is a negative predictor of returns (dilution
signals overvaluation / equity-financing at high prices). The canonical method
(Pontiff & Woodgate 2008) measures growth in split-adjusted shares between
t-18 and t-6 months in the cross-section.

Proxy implementation: We use PIT fundamentals (shares_outstanding, filed on
the date the number was publicly known) and compute:
  1. osap_shareiss1y_level  : ln(shares_t) − ln(shares_t-12m), measured as of
                              the most-recent filing vs the filing ~12 months
                              prior, updated whenever a new filing lands.
                              Positive = dilution (bad for returns, sign = -1).
  2. osap_shareiss1y_lag6   : Same but lagged by ~6 months so the window
                              approximates the canonical t-18 to t-6 gap
                              (avoids the most recent months where some
                              issuance may not yet be reflected in prices,
                              matching the spirit of Pontiff-Woodgate).
  3. osap_shareiss1y_accel  : Change in the 1yr issuance rate over the past 2
                              quarters — positive = accelerating dilution.

Cross-section note: ranking across stocks is NOT done here (per-ticker only).
The raw log-growth is already a meaningful per-ticker signal.

Source: Pontiff and Woodgate (2008) "Share Issuance and Cross-Sectional
Returns", Journal of Finance 63(2):921-945.  Collected in Chen-Zimmermann
OpenSourceAP.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_shareiss1y",
    "description": (
        "Share issuance (1-year) per-ticker proxy. "
        "Measures log-growth in shares outstanding using PIT SEC filings. "
        "Predicted sign: -1 (more issuance -> lower future returns). "
        "Cross-section ranking not applied; raw log-growth used as signal. "
        "Source: Pontiff & Woodgate 2008 via Chen-Zimmermann OpenSourceAP."
    ),
    "requires": ["Close"],        # Close needed so the merge has a price anchor
    "produces": [
        "osap_shareiss1y_level",  # ln(shares_t / shares_t-12m)
        "osap_shareiss1y_lag6",   # same but lagged ~6 months (t-18 to t-6 approx)
        "osap_shareiss1y_accel",  # acceleration: change in 1yr issuance rate
    ],
    "tags": ["fundamentals", "issuance", "dilution", "external_financing", "pontiff_woodgate"],
    "version": "1.0",
    "author": "Pontiff and Woodgate (2008) via Chen-Zimmermann OpenSourceAP; proxy by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker share issuance features from PIT fundamentals."""

    # Pull PIT shares_outstanding (filed_date is the public knowledge date)
    df = _fundamentals.as_of(df, fields=["shares_outstanding"])

    shares = df["fund_shares_outstanding"]

    # Guard: replace zeros/negatives with NaN so log is safe
    shares = shares.where(shares > 0, other=np.nan)
    log_shares = np.log(shares)

    # ------------------------------------------------------------------
    # Feature 1: 1-year log growth in shares (most recent 12m of filings)
    # We approximate 12 months as a 252-trading-day trailing window.
    # Since fundamentals come at quarterly cadence, shift(252) picks up
    # ~4 quarters ago, which is ~12 months.
    # ------------------------------------------------------------------
    log_shares_1y_ago = log_shares.shift(252)
    osap_level = log_shares - log_shares_1y_ago

    # ------------------------------------------------------------------
    # Feature 2: Lagged version — growth ending ~6 months ago (shift 126)
    # This approximates the canonical Pontiff-Woodgate t-18 to t-6 window:
    #   t-18 = log_shares_1y_ago.shift(126) = log_shares.shift(378)
    #   t-6  = log_shares.shift(126)
    # ------------------------------------------------------------------
    log_shares_6m_ago = log_shares.shift(126)
    log_shares_18m_ago = log_shares.shift(378)
    osap_lag6 = log_shares_6m_ago - log_shares_18m_ago

    # ------------------------------------------------------------------
    # Feature 3: Acceleration — current 1yr rate minus 1yr rate from 2q ago
    # 2 quarters ~ 126 trading days
    # ------------------------------------------------------------------
    osap_level_2q_ago = log_shares.shift(126) - log_shares.shift(126 + 252)
    osap_accel = osap_level - osap_level_2q_ago

    # Guard against inf (shouldn't occur after the >0 guard, but be safe)
    df["osap_shareiss1y_level"] = osap_level.replace([np.inf, -np.inf], np.nan)
    df["osap_shareiss1y_lag6"]  = osap_lag6.replace([np.inf, -np.inf], np.nan)
    df["osap_shareiss1y_accel"] = osap_accel.replace([np.inf, -np.inf], np.nan)

    # Drop scratch fundamentals column — not in produces
    df = df.drop(columns=["fund_shares_outstanding"], errors="ignore")

    return df
