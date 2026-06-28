"""
Feature block: osap_orderbacklogchg
Change in order backlog (Baik and Ahn 2007), via OpenSourceAP (Chen-Zimmermann).

True order backlog (OB) is a separate Compustat field not available in our
fundamentals panel. The closest PIT-safe proxy available is RECEIVABLES, which
captures pending-revenue claims (i.e., goods/services delivered but not yet
collected), a widely-used approximation for backlog in empirical accounting
research when the dedicated backlog item is missing.

Proxy definition (per-ticker, all values PIT-safe via filed_date):
  normalized_backlog(t) = receivables(t) / mean(assets(t), assets(t-1))
  osap_orderbacklogchg_lvl   = normalized_backlog (current level)
  osap_orderbacklogchg_chg   = normalized_backlog(t) - normalized_backlog(t-1yr)
  osap_orderbacklogchg_slope = 4-quarter rolling slope of normalized_backlog
                               (captures trend direction)

Excludes (sets NaN) when proxy receivables <= 0, matching the spec's
"exclude if order backlog is 0" criterion.

Cross-sectional note: original signal is ranked cross-sectionally; here we
produce the per-ticker level and YoY change, which preserve the same economic
ordering within each stock's time series and can be combined with XS-ranking
at aggregation time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

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
    "name": "osap_orderbacklogchg",
    "description": (
        "Change in normalized order backlog (Baik & Ahn 2007 / OpenSourceAP Chen-Zimmermann). "
        "True order-backlog field unavailable; proxy = receivables / avg(assets_t, assets_t-1). "
        "Produces level, YoY change, and 4Q rolling slope. "
        "Rows where receivables <= 0 are set to NaN per spec exclusion rule. "
        "Per-ticker proxy -- combine with cross-sectional rank at aggregation."
    ),
    "requires": [],   # uses only PIT fundamentals, no OHLCV
    "produces": [
        "osap_orderbacklogchg_lvl",
        "osap_orderbacklogchg_chg",
        "osap_orderbacklogchg_slope",
    ],
    "tags": ["fundamentals", "accruals", "accounting", "backlog", "receivables"],
    "version": "1.0",
    "author": "Baik and Ahn (2007) via OpenSourceAP (Chen-Zimmermann); proxy impl.",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute order-backlog change features using PIT fundamentals."""

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals (backward merge_asof on filed_date)
    # ------------------------------------------------------------------
    df = _fundamentals.as_of(df, fields=["receivables", "assets"])

    rec = df["fund_receivables"]
    ast = df["fund_assets"]

    # ------------------------------------------------------------------
    # 2. Normalized backlog = receivables / avg(assets_t, assets_t-1)
    #    Using ffill (carries last reported value forward) then shifting
    #    one period for avg-assets denominator.
    # ------------------------------------------------------------------
    ast_lag1 = ast.shift(1)
    avg_assets = (ast + ast_lag1) / 2.0

    # Guard zero/negative/missing denominator
    avg_assets_safe = avg_assets.where(avg_assets > 0, other=np.nan)

    norm_backlog = rec / avg_assets_safe

    # Exclude rows where receivables <= 0 (spec: exclude if backlog == 0)
    norm_backlog = norm_backlog.where(rec > 0, other=np.nan)

    # ------------------------------------------------------------------
    # 3. Level feature
    # ------------------------------------------------------------------
    df["osap_orderbacklogchg_lvl"] = norm_backlog

    # ------------------------------------------------------------------
    # 4. YoY change: shift by ~252 trading days ≈ 1 fiscal year
    #    We use 252 rows (row-shift, not calendar), which is standard for
    #    annual-frequency fundamentals carried forward daily.
    # ------------------------------------------------------------------
    norm_backlog_lag_yr = norm_backlog.shift(252)
    df["osap_orderbacklogchg_chg"] = norm_backlog - norm_backlog_lag_yr

    # ------------------------------------------------------------------
    # 5. Rolling 4-quarter (~63 trading days per quarter) slope
    #    Fit OLS slope = cov(x,y)/var(x) over last 4Q window = 252 rows
    #    Vectorised via rolling window on index ranks.
    # ------------------------------------------------------------------
    window = 252
    n = len(norm_backlog)

    if n >= window:
        vals = norm_backlog.to_numpy(dtype=float)
        slopes = np.full(n, np.nan)

        # Precompute centred x for a window: x = [0..w-1] - mean([0..w-1])
        x = np.arange(window, dtype=float)
        x_c = x - x.mean()
        ss_x = (x_c ** 2).sum()

        for i in range(window - 1, n):
            y = vals[i - window + 1: i + 1]
            valid = ~np.isnan(y)
            if valid.sum() < window // 2:
                continue
            # Use only valid positions to avoid NaN contamination
            y_c = y[valid] - np.nanmean(y)
            x_sub = x_c[valid]
            ss_x_sub = (x_sub ** 2).sum()
            if ss_x_sub == 0:
                continue
            slopes[i] = (x_sub * y_c).sum() / ss_x_sub

        df["osap_orderbacklogchg_slope"] = slopes
    else:
        df["osap_orderbacklogchg_slope"] = np.nan

    # ------------------------------------------------------------------
    # 6. Drop scratch fund_* columns not in produces
    # ------------------------------------------------------------------
    df.drop(columns=["fund_receivables", "fund_assets"], inplace=True)

    return df
