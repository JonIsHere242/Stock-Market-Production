"""
osap_predictedfe — Predicted Analyst Forecast Error proxy (Frankel & Lee 1998)

Per-ticker proxy for the OpenSourceAP "Predicted Analyst Forecast Error" factor.
Original method: fitted value from cross-sectional regressions of analyst earnings
forecast errors on cross-sectional rankings of 5-year sales growth, book-to-market,
AOP (accruals / operating profitability), and analyst long-term growth.

Because (a) analyst forecasts are not in the OHLCV+PIT-fundamentals data and
(b) the regression is inherently cross-sectional, this block implements a
faithful per-ticker proxy:

  predicted_fe ~ alpha * sales_growth_proxy
               + beta  * (1 / book_to_market)   # growth tilt → optimism
               + gamma * aop_proxy               # accruals optimism
               + delta * ltg_proxy               # LTG optimism

All inputs are from PIT fundamentals (backward-merged, no lookahead).
Predicted sign is -1 (stocks where analysts are predicted to be most optimistic
underperform; long LOW predicted-forecast-error, short HIGH).

Produces three columns:
  osap_predictedfe_score  — composite z-score proxy (higher = more analyst optimism
                             = worse expected return; sign -1)
  osap_predictedfe_slope  — 63-day rolling trend in the composite (momentum of
                             optimism buildup)
  osap_predictedfe_aop    — accruals-to-operating-profit component alone (AOP proxy)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path as _P
import importlib.util as _ilu

METADATA = {
    "name": "osap_predictedfe",
    "description": (
        "Per-ticker proxy for Predicted Analyst Forecast Error (Frankel & Lee 1998 / "
        "OpenSourceAP Chen-Zimmermann). Original: cross-sectional regression of analyst "
        "forecast errors on 5-yr sales growth, book-to-market, AOP, and analyst LTG ranks. "
        "Here: PIT-fundamentals-based composite capturing the same analyst-optimism channels: "
        "revenue growth tilt, growth-stock (low BtM) bias, accruals-based optimism (AOP), and "
        "earnings momentum as LTG proxy. Inherently cross-sectional in origin; this is a "
        "per-ticker faithful proxy. Predicted sign -1: high score = more predicted optimism = "
        "underperformance expected."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_predictedfe_score",
        "osap_predictedfe_slope",
        "osap_predictedfe_aop",
    ],
    "tags": ["earnings", "forecast_error", "fundamentals", "analyst_optimism", "accounting"],
    "version": "1.0",
    "author": "Frankel and Lee 1998; OpenSourceAP (Chen-Zimmermann); per-ticker proxy impl.",
}

# ── load fundamentals helper ──────────────────────────────────────────────────
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)


def _safe_zscore_series(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score (past-only) with a minimum of 20 observations."""
    roll = s.rolling(window, min_periods=20)
    mu = roll.mean()
    sd = roll.std(ddof=1)
    sd = sd.replace(0, np.nan)
    return (s - mu) / sd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ── pull PIT fundamentals ────────────────────────────────────────────────
    fields = [
        "revenue_ttm",       # for sales-growth proxy
        "book_value_per_share",  # for BtM
        "net_income_ttm",    # for AOP / earnings
        "operating_income_ttm",  # for AOP
        "operating_cash_flow_ttm",  # for accruals: operating_income - op_cf
        "eps_basic_ttm",     # LTG proxy
        "assets",            # normalisation
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # ── 1. Sales-growth proxy (trailing revenue acceleration) ─────────────────
    # 5-year sales growth → approx with rolling % change over 252*5 bars on
    # TTM revenue. We use a shorter window (252) to stay stable, then rank
    # within ticker via rolling z-score.
    rev = df["fund_revenue_ttm"].replace(0, np.nan)
    # Year-on-year change (shift 252 trading days ≈ 1 year)
    rev_growth_1y = (rev - rev.shift(252)) / rev.shift(252).abs().replace(0, np.nan)
    # 5-year cumulative growth (shift 1260 ≈ 5 years, fall back to 1y if sparse)
    rev_growth_5y = (rev - rev.shift(1260)) / rev.shift(1260).abs().replace(0, np.nan)
    # Use 5y if available, otherwise 1y
    sales_growth = rev_growth_5y.where(rev_growth_5y.notna(), rev_growth_1y)
    sg_z = _safe_zscore_series(sales_growth)

    # ── 2. Inverse book-to-market (growth tilt) ──────────────────────────────
    # BtM = book_value_per_share / Close; 1/BtM = price-to-book
    # High P/B (growth stocks) → analysts tend to be overly optimistic
    bvps = df["fund_book_value_per_share"].replace(0, np.nan)
    close = df["Close"].replace(0, np.nan)
    btm = bvps / close
    btm = btm.where(btm > 0, np.nan)          # negative book = drop
    inv_btm = 1.0 / btm                        # P/B; high → growth optimism
    inv_btm_z = _safe_zscore_series(inv_btm)

    # ── 3. AOP proxy (accruals-based operating optimism) ─────────────────────
    # AOP ≈ (operating_income - operating_cash_flow) / assets
    # Positive accruals → earnings above cash → analyst optimism inflated
    op_inc = df["fund_operating_income_ttm"]
    op_cf = df["fund_operating_cash_flow_ttm"]
    assets = df["fund_assets"].replace(0, np.nan)
    accruals = (op_inc - op_cf) / assets
    aop_z = _safe_zscore_series(accruals)

    # Store the raw AOP z-score as its own produced column
    df["osap_predictedfe_aop"] = aop_z

    # ── 4. LTG proxy (analyst long-term growth → EPS acceleration) ───────────
    # Proxy: rolling slope of EPS_ttm (earnings trend); fast-growing EPS →
    # analysts embed high LTG → over-optimism
    eps = df["fund_eps_basic_ttm"]
    eps_chg_1y = (eps - eps.shift(252)) / eps.shift(252).abs().replace(0, np.nan)
    # Clip extreme values
    eps_chg_1y = eps_chg_1y.clip(-5, 5)
    ltg_z = _safe_zscore_series(eps_chg_1y)

    # ── 5. Composite predicted-FE score ──────────────────────────────────────
    # Weight roughly equal (no cross-sectional regression available per-ticker).
    # Signs aligned with optimism direction:
    #   sales_growth  +1 (more growth → more optimism)
    #   inv_btm       +1 (higher P/B → more optimism)
    #   aop           +1 (higher accruals → more optimism)
    #   ltg           +1 (higher EPS growth → analyst LTG inflated)
    components = pd.DataFrame(
        {
            "sg": sg_z,
            "ptb": inv_btm_z,
            "aop": aop_z,
            "ltg": ltg_z,
        }
    )
    # Equal-weight average of non-NaN components (need ≥2 to produce a score)
    n_valid = components.notna().sum(axis=1)
    composite = components.sum(axis=1, min_count=2) / n_valid.where(n_valid >= 2, np.nan)

    df["osap_predictedfe_score"] = composite

    # ── 6. Slope: 63-day rolling trend in composite score ────────────────────
    # Measures whether analyst optimism is building (positive slope = worsening)
    def rolling_slope(s: pd.Series, w: int = 63) -> pd.Series:
        """OLS slope of s over rolling window w using vectorised formula."""
        x = np.arange(w, dtype=float)
        x_dm = x - x.mean()
        ss_x = float((x_dm ** 2).sum())

        def _slope(arr: np.ndarray) -> float:
            if np.isnan(arr).sum() > w // 3:
                return np.nan
            arr_filled = pd.Series(arr).ffill().values
            y_dm = arr_filled - arr_filled.mean()
            return float(np.dot(x_dm, y_dm) / ss_x)

        return s.rolling(w, min_periods=max(w // 2, 10)).apply(_slope, raw=True)

    df["osap_predictedfe_slope"] = rolling_slope(composite, w=63)

    # ── 7. Drop scratch fund_* columns not in produces ────────────────────────
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
