"""
Real Dirty Surplus (osap_rds) — per-ticker proxy block.

Economic signal (Landsman et al. 2011, via Chen-Zimmermann OpenSourceAP):
  Companies with large "real dirty surplus" — equity changes that bypass the
  income statement — tend to have *lower* future returns (predicted sign +1 means
  long high, but the construct measures accounting opacity; implementations often
  find the raw value negatively predicts returns when large).

Full definition:
  dirty_surplus   = Δmsa + Δrecta + 0.65 * Δmin(pcupsu - paddml, 0)
  real_dirty_surplus = Δceq - dirty_surplus - (ni - dvp) + divamt
                      - prcc_f * Δcsho

Proxy rationale:
  Compustat-specific items msa, recta, pcupsu, paddml, dvp, divamt are NOT
  available in the fundamentals panel.  We approximate:
    - dirty_surplus_proxy   ≈ Δequity - (net_income_ttm - dividends_paid_ttm_est)
      i.e. the "unexplained" change in book equity relative to clean retained
      earnings.  This captures the same OCI / pension / MSA reclassification
      noise that Landsman et al. target.
    - share_dilution_adj    = Close * Δshares_outstanding  (real-value
      component that dilutes existing holders, as in the full formula).
    - real_dirty_surplus_proxy = dirty_surplus_proxy - share_dilution_adj

  All values are normalised by lagged equity to make them comparable across
  firm sizes (and more robust to outliers).  TTM income / paid dividends are
  used as the closest PIT-safe annual equivalent.

  Cross-sectional rank (across stocks on a given day) is the correct use of
  this factor; the per-ticker time-series captures the same LEVEL signal and
  allows the downstream model to rank implicitly.
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
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_rds",
    "description": (
        "Per-ticker proxy for Real Dirty Surplus (Landsman et al. 2011, "
        "OpenSourceAP / Chen-Zimmermann). Measures equity changes that bypass "
        "the income statement (OCI, pension adjustments, hidden dilution). "
        "Full formula requires Compustat msa/recta/pcupsu/paddml/dvp/divamt; "
        "proxy uses available PIT fundamentals: Δequity vs clean retained "
        "earnings (net_income_ttm + dividends_paid_ttm), adjusted for share "
        "dilution cost (Close * Δshares_outstanding), normalised by lagged "
        "equity. Predicted sign: +1 (long high dirty surplus stocks). "
        "Cross-sectional by nature; per-ticker time-series captures level signal."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_rds_raw",       # raw real dirty surplus proxy / lagged equity
        "osap_rds_ma4",       # 4-quarter rolling mean of raw (smoothed)
        "osap_rds_slope",     # 4-period slope of raw (trend in accounting noise)
    ],
    "tags": ["accounting", "fundamentals", "dirty_surplus", "oci", "osap"],
    "version": "1.0",
    "author": "Landsman et al. 2011 via Chen-Zimmermann OpenSourceAP; proxy by Claude",
}


# ---------------------------------------------------------------------------
# Helper: robust slope over a short window (no lookahead, fully vectorised)
# ---------------------------------------------------------------------------
def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of series ~ [0,1,...,w-1] over rolling window, vectorised."""
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if window < 2 or n < window:
        return pd.Series(result, index=series.index)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        result[i] = cov / x_var if x_var > 0 else np.nan

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals we need
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=[
                "equity",             # book equity (ceq proxy)
                "net_income_ttm",     # TTM net income (ni proxy)
                "dividends_paid_ttm", # TTM dividends paid (divamt proxy, sign may be negative)
                "shares_outstanding", # common shares outstanding (csho)
            ],
        )

    eq   = df["fund_equity"].to_numpy(dtype=float)
    ni   = df["fund_net_income_ttm"].to_numpy(dtype=float)
    div  = df["fund_dividends_paid_ttm"].to_numpy(dtype=float)   # often negative in Compustat convention
    shr  = df["fund_shares_outstanding"].to_numpy(dtype=float)
    px   = df["Close"].to_numpy(dtype=float)
    n    = len(df)

    raw = np.full(n, np.nan)

    for i in range(1, n):
        eq_now  = eq[i]
        eq_prev = eq[i - 1]
        ni_now  = ni[i]
        div_now = div[i]
        shr_now = shr[i]
        shr_prv = shr[i - 1]
        px_now  = px[i]

        # Skip if any critical fundamental is missing
        if any(np.isnan(v) for v in [eq_now, eq_prev, ni_now, shr_now, shr_prv, px_now]):
            continue

        lagged_eq = eq_prev
        if lagged_eq == 0.0 or np.isnan(lagged_eq):
            continue

        # Clean retained earnings = net_income - dividends
        # dividends_paid_ttm is typically negative (cash outflow); abs to get magnitude
        div_val = abs(div_now) if not np.isnan(div_now) else 0.0

        clean_re = ni_now - div_val  # clean earnings retained

        # Δbook equity
        delta_eq = eq_now - eq_prev

        # Dirty surplus proxy = unexplained change in book equity
        dirty_surplus_proxy = delta_eq - clean_re

        # Share dilution cost: real value transferred to/from new shareholders
        delta_shr = shr_now - shr_prv
        share_dilution = px_now * delta_shr   # in same units as equity if equity is per-share?
        # Note: equity in fundamentals is total (millions); shares_outstanding in millions too.
        # px * delta_shr gives ($/share * million_shares) = $ millions -> consistent.

        rds = dirty_surplus_proxy - share_dilution

        # Normalise by lagged equity for cross-sectional comparability
        raw[i] = rds / abs(lagged_eq)

    raw_series = pd.Series(raw, index=df.index)

    # Guard against inf/-inf (division edge cases)
    raw_series = raw_series.replace([np.inf, -np.inf], np.nan)

    # 4-period rolling mean (smooths quarterly filing lumpiness)
    ma4 = raw_series.rolling(window=4, min_periods=2).mean()

    # 4-period slope (trend in accounting noise)
    slope4 = _rolling_slope(raw_series, window=4)
    slope4 = pd.Series(slope4.values, index=df.index).replace([np.inf, -np.inf], np.nan)

    df["osap_rds_raw"]   = raw_series.values
    df["osap_rds_ma4"]   = ma4.values
    df["osap_rds_slope"] = slope4.values

    # Drop scratch fund_ columns not in produces
    for col in ["fund_equity", "fund_net_income_ttm", "fund_dividends_paid_ttm", "fund_shares_outstanding"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
