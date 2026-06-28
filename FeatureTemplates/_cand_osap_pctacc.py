"""
osap_pctacc — Percent Accruals (Hafzalla, Lundholm & Van Winkle 2011)

Scales the classic Sloan (1996) accruals by |net income| rather than total
assets, which amplifies the accrual signal for high-earnings firms and
attenuates it for low-earnings ones. Predicted sign: -1 (high pct-accruals
is bearish — earnings are less backed by cash flow as a fraction of income).

Per-ticker proxy: accruals are approximated from PIT fundamentals as
  acc = net_income_ttm - operating_cash_flow_ttm
scaled by |net_income_ttm| (denominator floored to avoid blow-up near zero).
A rolling slope column captures whether the percent-accrual is improving or
deteriorating.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

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
    "name": "osap_pctacc",
    "description": (
        "Percent Accruals: (net_income_ttm - operating_cash_flow_ttm) / "
        "|net_income_ttm|, per Hafzalla, Lundholm & Van Winkle (2011). "
        "Predicted sign: -1 (high percent-accruals is bearish). "
        "Per-ticker proxy using PIT TTM fundamentals — cross-sectional "
        "quintile ranking not performed (inherently XS); level and rolling "
        "4-quarter slope are produced."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_pctacc_level",    # (NI_ttm - OCF_ttm) / |NI_ttm|
        "osap_pctacc_slope",    # 63-day rolling OLS slope of the level
        "osap_pctacc_accruals", # raw accruals / avg_assets proxy
    ],
    "tags": ["accruals", "accounting", "quality", "fundamentals", "osap"],
    "version": "1.0",
    "author": "OpenSourceAP (Chen-Zimmermann); Hafzalla, Lundholm & Van Winkle 2011",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add percent-accrual features using PIT TTM fundamentals."""

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals (backward-safe merge_asof on filed_date)
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(
        df,
        fields=[
            "net_income_ttm",
            "operating_cash_flow_ttm",
            "assets",
        ],
    )

    ni = df["fund_net_income_ttm"]
    ocf = df["fund_operating_cash_flow_ttm"]
    assets = df["fund_assets"]

    # ------------------------------------------------------------------
    # 2. Raw accruals (income statement minus cash flow approach)
    # ------------------------------------------------------------------
    raw_acc = ni - ocf  # positive = more accrual-heavy

    # ------------------------------------------------------------------
    # 3. Percent Accruals: scale by |net income|
    #    Guard: when |NI| < 1e-6 (near-zero), set denominator to NaN
    #    to avoid +/-inf.  Also emit NaN if NI itself is NaN.
    # ------------------------------------------------------------------
    abs_ni = ni.abs()
    denom_ni = abs_ni.where(abs_ni > 1e-6, other=np.nan)
    pct_acc = raw_acc / denom_ni

    # Winsorise extreme values at ±10 (1000%) to avoid outlier damage
    pct_acc = pct_acc.clip(-10.0, 10.0)
    df["osap_pctacc_level"] = pct_acc

    # ------------------------------------------------------------------
    # 4. Classic accruals / avg_assets for reference (Sloan 1996 proxy)
    # ------------------------------------------------------------------
    avg_assets = assets.rolling(window=2, min_periods=1).mean()
    avg_assets_safe = avg_assets.where(avg_assets.abs() > 1e-6, other=np.nan)
    acc_ta = raw_acc / avg_assets_safe
    acc_ta = acc_ta.clip(-5.0, 5.0)
    df["osap_pctacc_accruals"] = acc_ta

    # ------------------------------------------------------------------
    # 5. 63-day (≈1 quarter) rolling OLS slope of pct_acc level
    #    Captures trend: deteriorating (rising) vs improving (falling)
    #    Uses a vectorised rolling window approach.
    # ------------------------------------------------------------------
    WIN = 63

    def _roll_slope(series: pd.Series, window: int) -> pd.Series:
        """Rolling OLS slope (no lookahead)."""
        n = len(series)
        result = np.full(n, np.nan, dtype=np.float64)
        arr = series.to_numpy(dtype=np.float64)
        x = np.arange(window, dtype=np.float64)
        x -= x.mean()  # centre for numerical stability
        x2 = (x * x).sum()
        if x2 < 1e-12:
            return series  # degenerate, leave NaN
        for i in range(window - 1, n):
            y = arr[i - window + 1 : i + 1]
            if np.isnan(y).any():
                continue
            y_dm = y - y.mean()
            result[i] = (x * y_dm).sum() / x2
        return pd.Series(result, index=series.index)

    df["osap_pctacc_slope"] = _roll_slope(df["osap_pctacc_level"], WIN)

    # ------------------------------------------------------------------
    # 6. Drop scratch fund_* columns not in produces
    # ------------------------------------------------------------------
    for col in ["fund_net_income_ttm", "fund_operating_cash_flow_ttm", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
