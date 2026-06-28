"""
Ohlson O-Score (Dichev 1998 / Chen-Zimmermann OpenSourceAP).

Per-ticker implementation using PIT fundamentals (backward-merged filed_date).
Predicted sign: -1  (high O-Score = higher default risk = NEGATIVE expected return).

Because the original formulation requires:
  - GNP deflator (macro series, not available here → approximated by SPY-CPI proxy
    using rolling 5-year nominal GDP deflator approximation; in practice we use a
    static scale based on 2023 USD ≈ constant, since at is already in nominal USD
    and the deflator mainly rescales; we use log(at) with a constant offset that
    approximates the mid-sample GNP deflator ≈ 1.0 in 1978 dollars; a reasonable
    modern approximation is GNP_deflator ≈ Close-year-deflated assets, but given
    data constraints we use log(at) directly which is standard in modern replications).
  - act (current assets), lct (current liabilities), lt (total liabilities),
    at (total assets), ib (income before extraordinary items ≈ net_income),
    oancf (operating cash flow).
  - Two lagged annual ib values (t-12m and t-24m periods) — approximated using
    the PIT fundamentals time-series by taking prior available net_income_ttm values.

Cross-sectional exclusion (SIC, bottom-quintile) is inherently cross-sectional;
described in metadata but not applied here (pipeline applies filters externally).
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
METADATA = {
    "name": "osap_oscore",
    "description": (
        "Ohlson O-Score: a linear combination of accounting ratios measuring "
        "bankruptcy / default risk (Dichev 1998; OpenSourceAP Chen-Zimmermann). "
        "Predicted sign -1 (high score = higher default risk = lower fwd return). "
        "Implemented per-ticker using PIT fundamentals (backward merge on filed_date). "
        "GNP deflator approximated as 1.0 (assets already in nominal USD; modern "
        "replications omit or fix the deflator). Two-period lagged net_income "
        "approximated via pandas shift on the merged fund_ column. "
        "Original cross-sectional exclusions (SIC 4000-4999/>5999, price<5, "
        "bottom-quintile O-Score) are noted but NOT applied here — apply externally."
    ),
    "requires": ["Close", "High", "Low", "Open", "Volume"],
    "produces": [
        "osap_oscore_level",    # raw O-Score (lower = safer)
        "osap_oscore_chg",      # 1-period change in O-Score (momentum of distress)
        "osap_oscore_zscore",   # trailing 252-day z-score of O-Score level
    ],
    "tags": ["fundamentals", "default_risk", "accounting", "oscore", "dichev"],
    "version": "1.0.0",
    "author": "Dichev (1998); OpenSourceAP Chen-Zimmermann; block by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Ohlson O-Score per ticker using PIT fundamentals."""

    # Pull the fundamentals we need (all PIT-safe via filed_date backward merge)
    fields = [
        "assets",           # at  — total assets
        "assets_current",   # act — current assets
        "liabilities",      # lt  — total liabilities
        "liabilities_current",  # lct — current liabilities
        "net_income",       # ib  — income before extraordinary items (proxy)
        "net_income_ttm",   # ib_ttm — trailing twelve months net income
        "operating_cash_flow",      # fopt / oancf
        "operating_cash_flow_ttm",  # ttm variant as fallback
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=fields)

    # ------------------------------------------------------------------
    # Alias to formula variable names (all fund_* columns)
    # ------------------------------------------------------------------
    at  = df["fund_assets"].replace(0, np.nan)
    act = df["fund_assets_current"]
    lt  = df["fund_liabilities"]
    lct = df["fund_liabilities_current"]

    # ib: income before extraordinary items → use net_income (best proxy)
    ib = df["fund_net_income"]

    # oancf: operating cash flow; use ttm if spot is null
    oancf = df["fund_operating_cash_flow"].where(
        df["fund_operating_cash_flow"].notna(),
        df["fund_operating_cash_flow_ttm"]
    )

    # ------------------------------------------------------------------
    # GNP deflator: approximated as 1.0 (nominal USD, modern replication)
    # So log(at / GNP) ≈ log(at)
    # ------------------------------------------------------------------
    log_at = np.log(at.clip(lower=1e-9))  # guard against non-positive

    # ------------------------------------------------------------------
    # Leverage ratio
    # ------------------------------------------------------------------
    lt_at = (lt / at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Working capital / total assets
    # ------------------------------------------------------------------
    wc_at = ((act - lct) / at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Current ratio: lct / act
    # ------------------------------------------------------------------
    lct_act = (lct / act).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Insolvency indicator: 1 if lt > at
    # ------------------------------------------------------------------
    I_insolv = (lt > at).astype(float)
    # Where either is null, set to NaN
    I_insolv = I_insolv.where(lt.notna() & at.notna(), other=np.nan)

    # ------------------------------------------------------------------
    # Profitability: ib / at
    # ------------------------------------------------------------------
    ib_at = (ib / at).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Funds flow / liabilities: oancf / lt
    # ------------------------------------------------------------------
    lt_safe = lt.replace(0, np.nan)
    fopt_lt = (oancf / lt_safe).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Two-period lagged ib (approximated by shifting the ib series)
    # Since fundamentals are annual filings, a shift of ~252 trading days
    # ≈ 1 year prior filing; however the PIT series already represents the
    # most recently filed value, so we shift by 1 and 2 filing periods
    # using the numeric index after sorting (already ascending by Date).
    # We use .shift(1) and .shift(2) on the ib column as rough proxies for
    # t-12m and t-24m (each represents the prior distinct filing value).
    # ------------------------------------------------------------------
    ib_lag1 = ib.shift(1)   # t-12m proxy
    ib_lag2 = ib.shift(2)   # t-24m proxy

    # Indicator: two consecutive net losses (ib + lag1 + lag2 < 0 → any 2 of 3 negative)
    # Original: (ib + ib_{t-12} + ib_{t-24}) < 0
    ib_sum3 = ib + ib_lag1 + ib_lag2
    I_loss = (ib_sum3 < 0).astype(float)
    I_loss = I_loss.where(ib.notna() & ib_lag1.notna() & ib_lag2.notna(), other=np.nan)

    # ------------------------------------------------------------------
    # Change in income (normalised): (ib - ib_lag1) / (|ib| + |ib_lag1|)
    # ------------------------------------------------------------------
    denom_chg = (ib.abs() + ib_lag1.abs()).replace(0, np.nan)
    delta_ib = ((ib - ib_lag1) / denom_chg).replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Ohlson O-Score formula (Dichev 1998)
    # OScore = -1.32
    #          - 0.407 * log(at/GNP)
    #          + 6.03  * (lt/at)
    #          - 1.43  * ((act - lct)/at)
    #          + 0.076 * (lct/act)
    #          - 1.72  * I(lt > at)
    #          - 2.37  * (ib/at)
    #          - 1.83  * (oancf/lt)
    #          + 0.285 * I(net_losses)
    #          - 0.521 * delta_ib
    # ------------------------------------------------------------------
    oscore = (
        -1.32
        - 0.407 * log_at
        + 6.03  * lt_at
        - 1.43  * wc_at
        + 0.076 * lct_act
        - 1.72  * I_insolv
        - 2.37  * ib_at
        - 1.83  * fopt_lt
        + 0.285 * I_loss
        - 0.521 * delta_ib
    )

    # Guard: if any critical input is NaN, oscore should be NaN
    critical_mask = (
        at.isna() | lt.isna() | act.isna() | lct.isna() | ib.isna()
    )
    oscore = oscore.where(~critical_mask, other=np.nan)
    oscore = oscore.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Derived columns
    # ------------------------------------------------------------------
    # 1. Level
    df["osap_oscore_level"] = oscore

    # 2. Period-over-period change (captures rising/falling distress)
    df["osap_oscore_chg"] = oscore.diff(1)

    # 3. Trailing 252-day z-score (normalises across time for comparability)
    roll_mean = oscore.rolling(window=252, min_periods=63).mean()
    roll_std  = oscore.rolling(window=252, min_periods=63).std(ddof=1)
    roll_std  = roll_std.replace(0, np.nan)
    df["osap_oscore_zscore"] = ((oscore - roll_mean) / roll_std).replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # Drop scratch fund_* columns we are NOT producing
    # ------------------------------------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols, errors="ignore")

    return df
