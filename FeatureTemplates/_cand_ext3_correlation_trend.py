"""
ext3_correlation_trend — Market-correlation regime trend
Rolling correlation of stock returns with SPY, its trend, and fast-vs-slow gap.
Spec: Round-4 expansion (xdom2_downside_beta)
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext3_correlation_trend",
    "description": (
        "Per-ticker rolling correlation with SPY returns (60-day window) as a "
        "measure of market co-movement regime. Produces: (1) the correlation "
        "level itself, (2) its 126-day linear slope (rising = increasing "
        "integration, falling = growing diversification), and (3) the gap "
        "between a fast (20d) and slow (120d) rolling correlation — a "
        "momentum-style signal on the correlation itself. All computed per "
        "ticker against SPY daily log-returns; purely causal via backward "
        "merge_asof. Falls back to NaN when SPY data is missing."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_correlation_trend_60d",    # rolling 60d corr(stock, SPY)
        "ext3_correlation_trend_slope",  # 126d linear slope of the 60d corr
        "ext3_correlation_trend_gap",    # fast (20d) minus slow (120d) corr
    ],
    "tags": ["market", "correlation", "regime", "beta", "diversification"],
    "version": "1.0.0",
    "author": "Spec: Round-4 expansion (xdom2_downside_beta)",
}

# ---------------------------------------------------------------------------
# Helper: rolling linear slope without scipy
# ---------------------------------------------------------------------------
def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """OLS slope of the last `window` values of s (vectorised via strided ops)."""
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return pd.Series(out, index=s.index)
    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope = ((x - x_mean) * (y - y.mean())).sum() / x_var
        out[i] = slope
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns to NaN
    df["ext3_correlation_trend_60d"] = np.nan
    df["ext3_correlation_trend_slope"] = np.nan
    df["ext3_correlation_trend_gap"] = np.nan

    if len(df) < 21:
        return df

    # --- stock log-returns ---
    stock_ret = np.log(df["Close"] / df["Close"].shift(1))  # NaN at row 0; fine

    # --- SPY log-returns merged backward-safe ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spy_close = _indexes.index_close("SPY")
        if spy_close is None or len(spy_close) == 0:
            return df
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        spy_df["spy_ret"] = np.log(spy_df["spy_close"] / spy_df["spy_close"].shift(1))
    except Exception:
        return df

    # Merge backward (no lookahead)
    merged = pd.merge_asof(
        df[["Date"]].assign(Date=pd.to_datetime(df["Date"])).sort_values("Date"),
        spy_df[["Date", "spy_ret"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Realign to original df index
    merged = merged.set_index(df.index)

    spy_ret = merged["spy_ret"]

    # --- rolling correlations ---
    def _roll_corr(window: int) -> pd.Series:
        # Align series and compute rolling corr
        combined = pd.concat(
            [stock_ret.rename("s"), spy_ret.rename("m")], axis=1
        )
        return combined["s"].rolling(window, min_periods=max(5, window // 2)).corr(
            combined["m"]
        )

    corr_20 = _roll_corr(20)
    corr_60 = _roll_corr(60)
    corr_120 = _roll_corr(120)

    # Replace inf with NaN (guard)
    corr_20 = corr_20.replace([np.inf, -np.inf], np.nan)
    corr_60 = corr_60.replace([np.inf, -np.inf], np.nan)
    corr_120 = corr_120.replace([np.inf, -np.inf], np.nan)

    df["ext3_correlation_trend_60d"] = corr_60.values

    # --- 126d slope of the 60d correlation ---
    slope = _rolling_slope(corr_60, window=126)
    df["ext3_correlation_trend_slope"] = slope.values

    # --- fast-minus-slow gap ---
    gap = corr_20 - corr_120
    gap = gap.replace([np.inf, -np.inf], np.nan)
    df["ext3_correlation_trend_gap"] = gap.values

    return df
