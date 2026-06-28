"""
Inventory Growth (osap_chinv)
Thomas and Zhang (2002) via OpenSourceAP (Chen-Zimmermann).

12-month change in inventory divided by average total assets.
Negative predicted sign: high inventory growth -> lower future returns
(over-investment / demand disappointment signal).

PIT fundamentals are used via _fundamentals.as_of() which does a backward
merge on filed_date, so there is no lookahead. Because fundamentals are
reported quarterly, the "12-month change" is approximated as the difference
between the most recently filed inventory and the inventory filed ~12 months
prior (using a 252-trading-day lookback on the PIT-merged series).
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _fundamentals helper
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
    "name": "osap_chinv",
    "description": (
        "Inventory Growth (Thomas & Zhang 2002, OpenSourceAP Chen-Zimmermann). "
        "12-month change in inventory divided by average total assets. "
        "Predicted sign: -1 (high inventory growth -> lower future returns). "
        "Per-ticker PIT fundamentals: inventory and assets are sourced via "
        "_fundamentals.as_of() (backward merge on filed_date, lookahead-safe). "
        "The 12-month delta is approximated on the PIT-merged daily series using "
        "a 252-trading-day shift of the filed value. A slope variant (63-day "
        "rolling linear trend of the level) is also produced."
    ),
    "requires": [],  # no OHLCV needed; uses fundamentals only
    "produces": [
        "osap_chinv_level",   # raw PIT inventory-growth ratio
        "osap_chinv_slope",   # 63-day rolling slope of the level (momentum)
    ],
    "tags": ["fundamentals", "investment", "inventory", "accounting", "osap"],
    "version": "1.0",
    "author": "Thomas and Zhang (2002) via OpenSourceAP (Chen-Zimmermann). Block impl: Claude.",
}


# ---------------------------------------------------------------------------
# Helper: rolling OLS slope (vectorised via cumsum trick)
# ---------------------------------------------------------------------------
def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of y ~ t over a rolling window, vectorised."""
    arr = series.to_numpy(dtype=np.float64, na_value=np.nan)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=series.index)

    # pre-compute t values centred on window (0..window-1)
    t = np.arange(window, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0:
        return pd.Series(out, index=series.index)

    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        mask = ~np.isnan(y)
        if mask.sum() < max(4, window // 2):
            continue
        y_m = np.where(mask, y, np.nan)
        # use only non-nan; recompute t subset if nans present
        if mask.all():
            y_mean = y_m.mean()
            out[i] = ((t - t_mean) * (y_m - y_mean)).sum() / t_var
        else:
            t_sub = t[mask]
            y_sub = y[mask]
            t_sub_mean = t_sub.mean()
            t_sub_var = ((t_sub - t_sub_mean) ** 2).sum()
            if t_sub_var == 0:
                continue
            y_sub_mean = y_sub.mean()
            out[i] = ((t_sub - t_sub_mean) * (y_sub - y_sub_mean)).sum() / t_sub_var

    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pull PIT fundamentals: inventory and total assets
    df = _fundamentals.as_of(df, fields=["inventory", "assets"])

    inv_col = "fund_inventory"
    ast_col = "fund_assets"

    # -----------------------------------------------------------------------
    # Build inventory-growth level
    # -----------------------------------------------------------------------
    # 252-trading-day lagged value (approx 1 year on PIT-merged daily series)
    LOOKBACK = 252

    inv = df[inv_col].copy()
    ast = df[ast_col].copy()

    inv_lag = inv.shift(LOOKBACK)
    ast_lag = ast.shift(LOOKBACK)

    # Average total assets over the 12-month period
    avg_assets = (ast + ast_lag) / 2.0

    # Inventory change scaled by average assets
    delta_inv = inv - inv_lag

    with np.errstate(divide="ignore", invalid="ignore"):
        chinv_level = np.where(
            (avg_assets.notna()) & (avg_assets != 0.0),
            delta_inv / avg_assets,
            np.nan,
        )

    chinv_level_series = pd.Series(chinv_level, index=df.index)

    # Winsorise at 1%/99% to reduce outlier blow-up (fundamentals can have
    # one-off huge swings; we cap not fill so NaN structure is preserved)
    lo = chinv_level_series.quantile(0.01)
    hi = chinv_level_series.quantile(0.99)
    chinv_level_series = chinv_level_series.clip(lower=lo, upper=hi)

    df["osap_chinv_level"] = chinv_level_series

    # -----------------------------------------------------------------------
    # Slope variant: 63-trading-day (≈1 quarter) rolling OLS trend of level
    # Captures acceleration / deceleration of inventory build-up
    # -----------------------------------------------------------------------
    df["osap_chinv_slope"] = _rolling_slope(chinv_level_series, window=63)

    # Drop scratch fundamentals columns we are not publishing
    df.drop(columns=[c for c in [inv_col, ast_col] if c in df.columns], inplace=True)

    return df
