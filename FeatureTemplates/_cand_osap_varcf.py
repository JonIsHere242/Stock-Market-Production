"""
osap_varcf — Cash-flow-to-price variance (Haugen & Baker 1996, OpenSourceAP)

Rolling variance of (ib + dp) / mve_c over the past ~60 months (min 24 months).
High variance → negative expected return (predicted sign: −1).

Per-ticker proxy: uses PIT fundamentals (net_income_ttm as ib proxy,
operating_cash_flow_ttm − net_income_ttm as dp proxy, shares_outstanding × Close
as mve_c).  Cross-sectional ranking is inherently not possible per-ticker; the
raw variance level and its trend are the signals here.
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
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_varcf",
    "description": (
        "Rolling variance of cash-flow-to-price ratio (ib+dp)/mve_c over the "
        "trailing ~60 months (min 24 months).  ib ≈ net_income_ttm; dp ≈ "
        "operating_cash_flow_ttm − net_income_ttm (accrual component); "
        "mve_c = Close × shares_outstanding.  Per-ticker proxy for the "
        "cross-sectional Haugen & Baker 1996 factor from OpenSourceAP "
        "(Chen-Zimmermann).  Predicted sign: −1 (high variance → low return).  "
        "Fundamentals coverage ~84%; ETFs/foreign rows will be NaN."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_varcf_var60m",    # rolling 60-month variance of cf/price ratio
        "osap_varcf_var24m",    # shorter 24-month window for recency
        "osap_varcf_trend",     # slope of the ratio itself (sign of recent drift)
    ],
    "tags": ["accounting", "cash_flow", "risk", "variance", "value"],
    "version": "1.0",
    "author": "Haugen and Baker 1996; OpenSourceAP (Chen-Zimmermann)",
}

# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling variance of cash-flow-to-price and a trend variant.

    Steps
    -----
    1. Attach PIT fundamentals (net_income_ttm, operating_cash_flow_ttm,
       shares_outstanding).
    2. Build cf_price = (net_income_ttm + max(ocf_ttm − net_income_ttm, 0))
                        / (Close × shares_outstanding).
       Clamped depreciation proxy to ≥ 0 (OCF − NI can be negative for
       heavily accrual-heavy firms; we take the non-negative part as the
       pure cash add-back).
    3. Resample monthly (last observation per month) to match the
       "60 months" spec cadence, then roll variance, then merge back to
       daily df on a backward-fill.
    4. Produce a 24-month variance column and a 12-month OLS trend of the
       ratio.
    """
    # -- attach fundamentals -------------------------------------------------
    df = _fundamentals.as_of(
        df,
        fields=["net_income_ttm", "operating_cash_flow_ttm", "shares_outstanding"],
    )

    fund_cols = ["fund_net_income_ttm", "fund_operating_cash_flow_ttm",
                 "fund_shares_outstanding"]

    # -- build cf/price ratio on daily df ------------------------------------
    ni   = df["fund_net_income_ttm"]
    ocf  = df["fund_operating_cash_flow_ttm"]
    shr  = df["fund_shares_outstanding"]
    px   = df["Close"]

    # depreciation proxy: OCF − NI, floored at 0
    dp_proxy = (ocf - ni).clip(lower=0)
    cashflow = ni + dp_proxy          # ib + dp  (both in currency units)

    mve_c = px * shr                  # market value (same currency units)
    # guard division
    mve_c_safe = mve_c.where(mve_c.abs() > 0, other=np.nan)
    cf_price = cashflow / mve_c_safe

    # replace inf/-inf that can sneak through
    cf_price = cf_price.replace([np.inf, -np.inf], np.nan)

    # -- monthly resampling (end-of-month observation) -----------------------
    # Work on a temporary frame indexed by Date
    tmp = pd.DataFrame({"cf_price": cf_price.values}, index=pd.to_datetime(df["Date"]))
    monthly = tmp.resample("ME").last()   # last daily obs per calendar month

    # -- rolling variance (min_periods enforces data-quality gate) -----------
    var60 = monthly["cf_price"].rolling(window=60, min_periods=24).var()
    var24 = monthly["cf_price"].rolling(window=24, min_periods=12).var()

    # -- rolling OLS slope over 12 months ------------------------------------
    def _rolling_slope(s: pd.Series, window: int = 12, min_p: int = 6) -> pd.Series:
        """Efficient rolling OLS slope via numpy strided approach."""
        vals = s.values.astype(float)
        n    = len(vals)
        out  = np.full(n, np.nan)
        x    = np.arange(window, dtype=float)
        xm   = x.mean()
        xvar = ((x - xm) ** 2).sum()
        if xvar == 0:
            return pd.Series(out, index=s.index)
        for i in range(window - 1, n):
            y = vals[i - window + 1 : i + 1]
            valid = ~np.isnan(y)
            if valid.sum() < min_p:
                continue
            xv = x[valid]
            yv = y[valid]
            xv_m = xv.mean()
            yv_m = yv.mean()
            denom = ((xv - xv_m) ** 2).sum()
            if denom == 0:
                continue
            out[i] = ((xv - xv_m) * (yv - yv_m)).sum() / denom
        return pd.Series(out, index=s.index)

    slope12 = _rolling_slope(monthly["cf_price"], window=12, min_p=6)

    # -- build monthly result frame ------------------------------------------
    monthly_result = pd.DataFrame({
        "osap_varcf_var60m": var60.values,
        "osap_varcf_var24m": var24.values,
        "osap_varcf_trend":  slope12.values,
    }, index=monthly.index)

    # -- merge back to daily (backward fill — last available month end) -------
    daily_idx = pd.to_datetime(df["Date"])
    daily_result = pd.merge_asof(
        pd.DataFrame({"Date": daily_idx}).sort_values("Date"),
        monthly_result.reset_index().rename(columns={"index": "Date"}),
        on="Date",
        direction="backward",
    ).set_index(pd.RangeIndex(len(df)))

    # Align to original df index order (df is already ascending by date)
    df["osap_varcf_var60m"] = daily_result["osap_varcf_var60m"].values
    df["osap_varcf_var24m"] = daily_result["osap_varcf_var24m"].values
    df["osap_varcf_trend"]  = daily_result["osap_varcf_trend"].values

    # -- drop scratch fundamentals columns ------------------------------------
    df.drop(columns=[c for c in fund_cols if c in df.columns], inplace=True)

    return df
