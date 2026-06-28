"""
ext4_dual_beta_spread — QQQ-vs-IWM rolling beta spread.

Produces the rolling 120-day beta of the stock to QQQ minus its rolling
120-day beta to IWM (net large-cap-growth vs small-cap exposure), plus the
60-day change in that spread.  This is a factor-exposure spread distinct from
plain market beta and orthogonal to most OHLCV-derived signals.

Per-ticker computation; no lookahead; all index alignment is merge_asof
backward on Date.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path — never `from FeatureTemplates import`)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_indexes)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "ext4_dual_beta_spread",
    "description": (
        "Rolling 120-day beta to QQQ minus rolling 120-day beta to IWM "
        "(large-growth vs small-cap factor-exposure spread), plus the 60-day "
        "change in that spread.  A per-ticker proxy for relative style tilt; "
        "higher values = more large-growth-loaded.  Index series aligned via "
        "merge_asof backward (lookahead-safe).  Author note: cross-sectional "
        "ranking of the spread is the richer signal, but this per-ticker "
        "time-series captures the same economic axis."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_dual_beta_spread_level",   # beta_QQQ - beta_IWM (120d)
        "ext4_dual_beta_spread_chg60",   # 60-day change in the spread
    ],
    "tags": ["beta", "factor", "index", "style", "growth", "smallcap"],
    "version": "1.0.0",
    "author": "Round-5 expansion (NEW: multi-index factor)",
}


# ---------------------------------------------------------------------------
# Rolling OLS beta helper (vectorised, no python loop over rows)
# ---------------------------------------------------------------------------
def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS beta of y on x using window bars.

    beta = Cov(y, x) / Var(x), computed via rolling mean/var.
    Divisions guarded; returns NaN where Var(x) == 0.
    """
    # Align on common index (both should already be aligned when passed in)
    y = y.astype(float)
    x = x.astype(float)

    roll_cov = (
        (y - y.rolling(window, min_periods=window).mean())
        .mul(x - x.rolling(window, min_periods=window).mean())
        .rolling(window, min_periods=window)
        .mean()
    )
    roll_var = x.rolling(window, min_periods=window).var(ddof=0)

    beta = roll_cov / roll_var.replace(0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ext4_dual_beta_spread_level and ext4_dual_beta_spread_chg60 to df.
    """
    WINDOW = 120
    CHG_WINDOW = 60

    # ------------------------------------------------------------------
    # 1. Fetch QQQ and IWM close series; degrade gracefully if missing
    # ------------------------------------------------------------------
    qqq_close: pd.Series | None = None
    iwm_close: pd.Series | None = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            qqq_close = _indexes.index_close("QQQ")
        except Exception:
            pass
        try:
            iwm_close = _indexes.index_close("IWM")
        except Exception:
            pass

    # If either index is unavailable, fill produced columns with NaN
    if qqq_close is None or iwm_close is None:
        df["ext4_dual_beta_spread_level"] = np.nan
        df["ext4_dual_beta_spread_chg60"] = np.nan
        return df

    # ------------------------------------------------------------------
    # 2. Align index series to df via merge_asof backward on Date
    # ------------------------------------------------------------------
    df_dates = df[["Date"]].copy()
    df_dates["_orig_order"] = np.arange(len(df_dates))

    def _align_index(series: pd.Series, col_name: str) -> pd.DataFrame:
        idx_df = series.reset_index()
        idx_df.columns = ["Date", col_name]
        idx_df["Date"] = pd.to_datetime(idx_df["Date"])
        return idx_df

    qqq_df = _align_index(qqq_close, "_qqq_close")
    iwm_df = _align_index(iwm_close, "_iwm_close")

    df_sorted = df_dates.copy()
    df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
    df_sorted = df_sorted.sort_values("Date")

    qqq_df = qqq_df.sort_values("Date")
    iwm_df = iwm_df.sort_values("Date")

    merged = pd.merge_asof(df_sorted, qqq_df, on="Date", direction="backward")
    merged = pd.merge_asof(merged, iwm_df, on="Date", direction="backward")

    # Restore original row order
    merged = merged.sort_values("_orig_order").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Compute log returns for the stock and both indexes
    # ------------------------------------------------------------------
    stock_ret = np.log(
        df["Close"].astype(float) / df["Close"].astype(float).shift(1)
    )
    qqq_ret = np.log(
        merged["_qqq_close"].astype(float)
        / merged["_qqq_close"].astype(float).shift(1)
    )
    iwm_ret = np.log(
        merged["_iwm_close"].astype(float)
        / merged["_iwm_close"].astype(float).shift(1)
    )

    # ------------------------------------------------------------------
    # 4. Rolling 120-day beta to QQQ and to IWM
    # ------------------------------------------------------------------
    beta_qqq = _rolling_beta(stock_ret, qqq_ret, WINDOW)
    beta_iwm = _rolling_beta(stock_ret, iwm_ret, WINDOW)

    # ------------------------------------------------------------------
    # 5. Spread = beta_QQQ - beta_IWM; 60-day change in spread
    # ------------------------------------------------------------------
    spread = beta_qqq - beta_iwm
    spread_chg = spread - spread.shift(CHG_WINDOW)

    # Guard infinities (should not arise but belt-and-suspenders)
    spread = spread.replace([np.inf, -np.inf], np.nan)
    spread_chg = spread_chg.replace([np.inf, -np.inf], np.nan)

    df["ext4_dual_beta_spread_level"] = spread.values
    df["ext4_dual_beta_spread_chg60"] = spread_chg.values

    return df
