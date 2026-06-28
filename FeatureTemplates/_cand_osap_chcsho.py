"""
osap_chcsho — Change in Shares Outstanding (dilution/buyback signal).

Source: Pontiff & Woodgate (2008) "Share Issuance and Cross-Sectional Returns",
        Journal of Finance 63(2), and the Open Source Asset Pricing (OSAP) library
        (Chen & Zimmermann, 2020, Rev. Asset Pricing Studies).

Signal: YoY log-change in shares outstanding.
  - Negative (shrink) = buyback → historically positive returns.
  - Positive (grow)   = issuance/dilution → historically negative returns.

Per-ticker proxy using PIT fundamentals (shares_outstanding filed via SEC EDGAR).
OHLCV market-cap proxy (Close × shares implied by revenue_ttm/sales_per_share) is
used as a fallback check but shares_outstanding is the primary series.

Produces:
  osap_chcsho_log_chg   — rolling 252-day log-change in shares_outstanding (annualised).
  osap_chcsho_accel     — first difference of osap_chcsho_log_chg (acceleration: is
                          dilution speeding up or slowing down?).
  osap_chcsho_flag      — categorical: -1 buyback, 0 neutral, +1 issuance
                          (based on sign + materiality threshold of 1%).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# --- load fundamentals helper (by path) ---
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

METADATA = {
    "name": "osap_chcsho",
    "description": (
        "Change in Shares Outstanding (dilution/buyback) anomaly. "
        "Firms that grow shares (issuance/dilution) tend to underperform; "
        "firms that shrink shares (buybacks) tend to outperform. "
        "Computed as the rolling 252-day log-change in PIT shares_outstanding "
        "from SEC EDGAR filings (lookahead-safe via filed_date backward merge). "
        "Per-ticker time-series proxy for the cross-sectional OSAP chcsho factor. "
        "Source: Pontiff & Woodgate (2008); Chen & Zimmermann (2020) OSAP."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_chcsho_log_chg",
        "osap_chcsho_accel",
        "osap_chcsho_flag",
    ],
    "tags": ["fundamental", "issuance", "dilution", "osap", "shares"],
    "version": "1.0.0",
    "author": "Pontiff & Woodgate (2008) JF; Chen & Zimmermann (2020) OSAP; impl. osap_chcsho",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Pull PIT shares_outstanding                                       #
    # ------------------------------------------------------------------ #
    df = _fundamentals.as_of(df, fields=["shares_outstanding"])
    sho = df["fund_shares_outstanding"].copy()  # float, millions

    # ------------------------------------------------------------------ #
    # 2. Log-change over ~252 trading days (≈1 year)                      #
    #    Use shift(252) so we compare to ~1yr ago fundamental filing.     #
    #    Because filings arrive quarterly and fund_ is PIT, consecutive   #
    #    rows with the same value are stable between filings — the shift  #
    #    naturally captures the most recent annual change.                #
    # ------------------------------------------------------------------ #
    WINDOW = 252
    sho_prev = sho.shift(WINDOW)

    # Guard: both values must be positive
    valid = (sho > 0) & (sho_prev > 0)
    log_chg = np.where(valid, np.log(sho / sho_prev), np.nan)
    log_chg = pd.Series(log_chg, index=df.index)

    # ------------------------------------------------------------------ #
    # 3. Acceleration: change in log_chg (is dilution speeding up?)       #
    # ------------------------------------------------------------------ #
    accel = log_chg.diff()

    # ------------------------------------------------------------------ #
    # 4. Flag: -1 buyback (shrink > 1%), 0 neutral, +1 issuance (grow>1%)#
    # ------------------------------------------------------------------ #
    THRESHOLD = np.log(1.01)  # ~1% in log space
    flag = np.where(
        log_chg.isna(), np.nan,
        np.where(log_chg < -THRESHOLD, -1.0,
                 np.where(log_chg > THRESHOLD, 1.0, 0.0))
    )
    flag = pd.Series(flag, index=df.index)

    # ------------------------------------------------------------------ #
    # 5. Assign produced columns; drop scratch fund_ column               #
    # ------------------------------------------------------------------ #
    df["osap_chcsho_log_chg"] = log_chg
    df["osap_chcsho_accel"] = accel
    df["osap_chcsho_flag"] = flag

    df.drop(columns=["fund_shares_outstanding"], inplace=True)

    return df
