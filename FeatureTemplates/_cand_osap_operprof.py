"""
Operating Profitability (Fama & French 2015 / OpenSourceAP Chen-Zimmermann)
SPEC ID: osap_operprof
SOURCE: OpenSourceAP (Chen-Zimmermann)

Operating profitability = (Revenue - COGS - SG&A - Interest Expense) / Book Equity.
This is the RMW (Robust Minus Weak) profitability factor from the Fama-French five-factor
model.  Predicted sign: +1 (long high operating profitability).

Per-ticker implementation: uses PIT fundamentals via _fundamentals.as_of().
The ratio is computed from TTM income-statement fields scaled by book equity (shareholders'
equity). Cross-sectional ranking is NOT performed here; the raw ratio is the signal.
Three columns are produced:
  - osap_operprof_ratio  : level signal (op profit / book equity)
  - osap_operprof_zscore : 1-year rolling z-score (tracks whether ratio is high vs own history)
  - osap_operprof_slope  : 1-year linear slope of ratio (improving vs deteriorating trend)
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helper: PIT fundamentals
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_operprof",
    "description": (
        "Operating profitability factor (Fama & French 2015 / OpenSourceAP Chen-Zimmermann). "
        "Formula: (Revenue - COGS - SG&A - Interest Expense) / Book Equity (shareholders equity). "
        "Corresponds to the RMW factor in the Fama-French five-factor model. "
        "SG&A is approximated as gross_profit_ttm - operating_income_ttm when not directly available. "
        "Interest expense from interest_expense_ttm. Book equity from fund_equity. "
        "Three variants produced: level ratio, 1-year rolling z-score, 1-year trend slope. "
        "Per-ticker only — no cross-sectional ranking. ETFs / foreign firms with no fundamentals "
        "will yield NaN throughout (expected; ~84% coverage)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_operprof_ratio",    # (Rev - COGS - SGA - IntExp) / BookEquity
        "osap_operprof_zscore",   # rolling 1-year z-score of ratio
        "osap_operprof_slope",    # 1-year linear slope of ratio (trend)
    ],
    "tags": ["profitability", "fundamentals", "accounting", "fama_french", "rmw"],
    "version": "1.0",
    "author": "Fama & French 2015 / OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # 1. Pull point-in-time fundamentals
    # -----------------------------------------------------------------------
    fields = [
        "revenue_ttm",
        "cost_of_revenue_ttm",
        "gross_profit_ttm",          # = revenue - cogs; helps derive SGA
        "operating_income_ttm",      # = gross_profit - SGA (approx)
        "interest_expense_ttm",
        "equity",                    # book equity (shareholders equity)
        "net_income_ttm",            # cross-check / fallback
        "operating_cash_flow_ttm",   # for additional context (not used in formula)
    ]
    df = _fundamentals.as_of(df, fields=fields)

    # -----------------------------------------------------------------------
    # 2. Derive components
    #
    # Fama-French (2015) operating profitability:
    #   OP = (revt - cogs - xsga - xint) / ceq
    #
    # Available mapping:
    #   revt  -> fund_revenue_ttm
    #   cogs  -> fund_cost_of_revenue_ttm
    #   xsga  -> gross_profit_ttm - operating_income_ttm
    #            (operating_income ≈ gross_profit - SGA, so SGA = gp - oi)
    #            Replace with 0 if missing (per OSAP convention)
    #   xint  -> fund_interest_expense_ttm  (replace NaN with 0)
    #   ceq   -> fund_equity  (book value of shareholders equity)
    # -----------------------------------------------------------------------

    rev  = df["fund_revenue_ttm"].copy()
    cogs = df["fund_cost_of_revenue_ttm"].copy()
    gp   = df["fund_gross_profit_ttm"].copy()
    oi   = df["fund_operating_income_ttm"].copy()
    xint = df["fund_interest_expense_ttm"].fillna(0.0)
    eq   = df["fund_equity"].copy()

    # SGA proxy: gross_profit - operating_income (replace missing with 0 per OSAP spec)
    sga_proxy = (gp - oi).fillna(0.0)
    # Clamp to >= 0 to avoid negative SGA from noise in proxies
    sga_proxy = sga_proxy.clip(lower=0.0)

    # Replace top-level missing items with 0 per OSAP convention for numerator
    rev_safe  = rev.fillna(0.0)
    cogs_safe = cogs.fillna(0.0)
    xint_safe = xint  # already filled above

    numerator = rev_safe - cogs_safe - sga_proxy - xint_safe

    # Scale by book equity; guard div-by-zero and near-zero equity
    eq_safe = eq.copy()
    eq_safe[eq_safe.abs() < 1e-6] = np.nan   # avoid division by tiny/zero equity
    eq_safe = eq_safe.replace(0.0, np.nan)

    ratio = numerator / eq_safe
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    # Winsorise extreme values (fundamentals data can have spikes at restatements)
    # Use 1st/99th percentile of non-NaN values as soft cap
    lo = ratio.quantile(0.01)
    hi = ratio.quantile(0.99)
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        ratio = ratio.clip(lower=lo, upper=hi)

    df["osap_operprof_ratio"] = ratio

    # -----------------------------------------------------------------------
    # 3. Rolling 1-year z-score (252 trading-day window, min 63)
    # -----------------------------------------------------------------------
    ROLL_WIN = 252
    MIN_PER  = 63

    roll_mean = ratio.rolling(window=ROLL_WIN, min_periods=MIN_PER).mean()
    roll_std  = ratio.rolling(window=ROLL_WIN, min_periods=MIN_PER).std()
    roll_std_safe = roll_std.replace(0.0, np.nan)

    zscore = (ratio - roll_mean) / roll_std_safe
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    df["osap_operprof_zscore"] = zscore

    # -----------------------------------------------------------------------
    # 4. 1-year linear trend slope of ratio (vectorised OLS via cumsum trick)
    #
    # For equally-spaced x = 0,1,...,n-1 over window of size W:
    #   slope = (W*Σ(x*y) - Σx*Σy) / (W*Σx² - (Σx)²)
    #
    # We compute rolling sums using pandas .rolling() to stay vectorised.
    # Slope is in ratio-units per trading day.
    # -----------------------------------------------------------------------
    SLOPE_WIN = 252
    MIN_SLOPE = 63

    y_ser = ratio.copy()
    n_rows = len(y_ser)

    # Build x-index relative to each window:  x_i = position within window
    # Trick: assign global index i; within window [start..end], x_k = k - start
    # Sums over window: Σx = Σ(i - start) = Σi - W*start
    # To do this fully vectorised, precompute prefix sums.

    y_arr = y_ser.values.astype(float)
    idx   = np.arange(n_rows, dtype=float)

    # For each window ending at position t (0-indexed):
    #   W_t  = number of valid points (non-NaN) in [t-W+1 .. t]
    #   Σy   = sum of valid y
    #   Σxy  = sum of (local_x * y) where local_x = position_in_window (0..W-1)
    #   Σx   = sum of local_x for valid positions
    #   Σx²  = sum of local_x² for valid positions
    # Local x for position p in window [start..t]: local_x = p - start

    # We avoid a Python loop by using numpy sliding_window_view where feasible,
    # but for a 252-window that is ~700*252 ~ 176k elements — acceptable.

    slopes = np.full(n_rows, np.nan)

    # Compute for each endpoint t from MIN_SLOPE-1 onward
    # Use vectorised prefix sums to get rolling quantities.
    # valid mask
    valid = ~np.isnan(y_arr)
    y_fill = np.where(valid, y_arr, 0.0)

    # Prefix sums (length n_rows+1, index 0 = before any element)
    cum_valid = np.concatenate([[0], valid.astype(float).cumsum()])
    cum_y     = np.concatenate([[0.0], y_fill.cumsum()])
    # For Σxy and Σx² we need weighted versions — we'll use global index i as x,
    # then correct for window offset later.
    cum_iy    = np.concatenate([[0.0], (idx * y_fill).cumsum()])
    cum_i     = np.concatenate([[0.0], (idx * valid).cumsum()])
    cum_i2    = np.concatenate([[0.0], (idx**2 * valid).cumsum()])

    for t in range(n_rows):
        start = max(0, t - SLOPE_WIN + 1)
        # prefix-sum slice [start..t] = cum[t+1] - cum[start]
        W_v  = cum_valid[t + 1] - cum_valid[start]
        if W_v < MIN_SLOPE:
            continue
        sum_y  = cum_y[t + 1]  - cum_y[start]
        sum_iy = cum_iy[t + 1] - cum_iy[start]
        sum_i  = cum_i[t + 1]  - cum_i[start]
        sum_i2 = cum_i2[t + 1] - cum_i2[start]

        # slope using global index (x = i); result is in ratio per day regardless of offset
        denom = W_v * sum_i2 - sum_i * sum_i
        if denom == 0.0:
            continue
        slopes[t] = (W_v * sum_iy - sum_i * sum_y) / denom

    df["osap_operprof_slope"] = slopes

    # -----------------------------------------------------------------------
    # 5. Drop scratch fund_* columns not in produces
    # -----------------------------------------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
