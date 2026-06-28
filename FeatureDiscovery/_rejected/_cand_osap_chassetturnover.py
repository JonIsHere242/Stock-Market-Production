"""
Change in Asset Turnover (Soliman 2008, via Chen-Zimmermann OpenSourceAP).

Per-ticker implementation using PIT SEC fundamentals (revenue_ttm / assets).
Cross-sectional ranking from the original paper is replaced by a per-ticker
time-series level + annual change signal; economic direction is preserved
(higher / rising asset turnover is bullish per METADATA sign = +1).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional helper: PIT fundamentals
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
    "name": "osap_chassetturnover",
    "description": (
        "Per-ticker annual change in asset turnover (revenue_ttm / assets), "
        "based on Soliman (2008) via the Chen-Zimmermann OpenSourceAP factor library. "
        "Original factor is cross-sectional (long high changers, short low); here we "
        "implement the per-ticker time-series analogue: level of asset turnover and "
        "its rolling 252-day (≈1yr) change, masked to NaN when Close < 5 (penny-stock "
        "exclusion per the spec). Coverage limited to ~84% of universe (ETFs/foreign "
        "lack fundamentals). Predicted sign: +1 (rising turnover is bullish)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_chassetturnover_level",   # current asset turnover (revenue_ttm / assets)
        "osap_chassetturnover_chg",     # annual change in asset turnover (primary signal)
        "osap_chassetturnover_slope",   # 63-day (≈1qtr) rolling slope of asset turnover
    ],
    "tags": ["fundamentals", "accounting", "sales_growth", "asset_turnover", "osap"],
    "version": "1.0",
    "author": "Soliman (2008); factor definition from Chen & Zimmermann OpenSourceAP",
}

# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns Date, Ticker, Open, High, Low, Close, Volume
         sorted ascending by Date (one ticker at a time).

    Returns
    -------
    df with three new columns added (in-place is fine; we always return df).
    """
    n = len(df)

    # Initialise output columns to NaN
    df["osap_chassetturnover_level"] = np.nan
    df["osap_chassetturnover_chg"] = np.nan
    df["osap_chassetturnover_slope"] = np.nan

    if n < 2:
        return df

    # ------------------------------------------------------------------
    # 1. Pull PIT fundamentals: revenue_ttm (numerator) and assets (denom)
    # ------------------------------------------------------------------
    try:
        df = _fundamentals.as_of(df, fields=["revenue_ttm", "assets"])
    except Exception:
        # Fundamentals unavailable → leave columns NaN
        return df

    # ------------------------------------------------------------------
    # 2. Asset turnover level  =  revenue_ttm / assets
    # ------------------------------------------------------------------
    rev = df["fund_revenue_ttm"].to_numpy(dtype=float)
    assets = df["fund_assets"].to_numpy(dtype=float)

    # Guard: assets == 0 → NaN; assets < 0 → NaN (balance-sheet noise)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = np.where((assets > 0) & np.isfinite(assets) & np.isfinite(rev),
                      rev / assets,
                      np.nan)

    # ------------------------------------------------------------------
    # 3. Annual change  (252 trading-day shift on the computed series)
    # ------------------------------------------------------------------
    lag_252 = np.full(n, np.nan)
    if n > 252:
        lag_252[252:] = at[:-252]

    with np.errstate(invalid="ignore"):
        chg = at - lag_252          # NaN if either side is NaN

    # ------------------------------------------------------------------
    # 4. Rolling slope of asset turnover over ~1 quarter (63 days)
    #    Use a simple OLS slope via covariance trick on a rolling window.
    # ------------------------------------------------------------------
    window = 63
    at_series = pd.Series(at, index=df.index)
    # rolling mean of at and of t (0..62), then cov / var(t)
    # t within each window: 0,1,...,window-1 → mean = (window-1)/2
    t_mean = (window - 1) / 2.0
    t_var = sum((i - t_mean) ** 2 for i in range(window))  # constant = 682.0 for w=63

    def _slope(arr):
        # arr is a numpy slice of length `window` from rolling
        valid = np.isfinite(arr)
        if valid.sum() < window // 2:
            return np.nan
        t_arr = np.arange(window, dtype=float)
        # use only valid positions
        t_v = t_arr[valid]
        a_v = arr[valid]
        t_vm = t_v.mean()
        a_vm = a_v.mean()
        denom = ((t_v - t_vm) ** 2).sum()
        if denom < 1e-12:
            return np.nan
        return ((t_v - t_vm) * (a_v - a_vm)).sum() / denom

    slope_vals = at_series.rolling(window, min_periods=window // 2).apply(
        _slope, raw=True
    ).to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 5. Penny-stock exclusion: mask all outputs where Close < 5
    # ------------------------------------------------------------------
    close = df["Close"].to_numpy(dtype=float)
    penny = close < 5.0  # True where we must mask

    at[penny] = np.nan
    chg[penny] = np.nan
    slope_vals[penny] = np.nan

    # ------------------------------------------------------------------
    # 6. Write outputs; clean up scratch fund_* columns
    # ------------------------------------------------------------------
    df["osap_chassetturnover_level"] = at
    df["osap_chassetturnover_chg"] = chg
    df["osap_chassetturnover_slope"] = slope_vals

    # Drop scratch fundamental columns (not in produces)
    for col in ["fund_revenue_ttm", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
