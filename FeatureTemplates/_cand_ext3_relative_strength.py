"""
Candidate feature block: ext3_relative_strength
Multi-horizon relative strength vs SPY market benchmark.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (by file path -- never import from FeatureTemplates)
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
    "name": "ext3_relative_strength",
    "description": (
        "Multi-horizon relative strength vs SPY at 21d, 63d, and 126d horizons, "
        "plus a slope measure (linear fit across the three horizon RS values) that "
        "captures whether relative strength is accelerating or decelerating. "
        "All values are computed per-ticker from OHLCV Close and the SPY index close; "
        "no cross-sectional information is used (each stock is processed independently). "
        "Proxy note: slope is estimated via a simple OLS across the three horizon points "
        "rather than a full rolling regression."
    ),
    "requires": ["Close"],
    "produces": [
        "ext3_relative_strength_21d",   # stock cum-ret minus SPY cum-ret over 21d
        "ext3_relative_strength_63d",   # stock cum-ret minus SPY cum-ret over 63d
        "ext3_relative_strength_126d",  # stock cum-ret minus SPY cum-ret over 126d
        "ext3_relative_strength_slope", # OLS slope across the three RS horizon values
    ],
    "tags": ["momentum", "relative_strength", "market_neutral", "multi_horizon"],
    "version": "1.0.0",
    "author": "Round-4 expansion (osap_idiovolaht); spec: ext3_relative_strength",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cum_ret(series: pd.Series, window: int) -> pd.Series:
    """Backward-looking cumulative return over `window` bars (no lookahead)."""
    shifted = series.shift(window)
    cr = series / shifted.replace(0, np.nan) - 1.0
    return cr


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multi-horizon relative strength columns to df (one stock at a time).

    Parameters
    ----------
    df : pd.DataFrame
        Per-ticker OHLCV with columns [Date, Ticker, Open, High, Low, Close, Volume],
        ascending by Date.

    Returns
    -------
    df : pd.DataFrame
        Original df with four new columns appended.
    """
    horizons = [21, 63, 126]

    # -----------------------------------------------------------------------
    # Fetch SPY close aligned to this stock's dates via merge_asof
    # -----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spy_close = _indexes.index_close("SPY")  # pd.Series indexed by DatetimeIndex
        except Exception:
            spy_close = None

    stock_close = df["Close"].values.astype(np.float64)

    # Build date array (ensure datetime for merge)
    dates = pd.to_datetime(df["Date"])

    rs_cols: dict[str, np.ndarray] = {}

    if spy_close is not None and len(spy_close) > 0:
        spy_df = spy_close.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        spy_df = spy_df.sort_values("Date").reset_index(drop=True)

        # Align SPY to stock dates (backward -- no lookahead)
        tmp = pd.DataFrame({"Date": dates})
        tmp = pd.merge_asof(
            tmp.sort_values("Date"),
            spy_df,
            on="Date",
            direction="backward",
        )
        # Restore original order
        tmp = tmp.set_index(df.index)
        spy_aligned = tmp["spy_close"].values.astype(np.float64)
    else:
        spy_aligned = np.full(len(df), np.nan)

    # -----------------------------------------------------------------------
    # Compute per-horizon relative strength
    # -----------------------------------------------------------------------
    stock_s = pd.Series(stock_close, index=df.index)
    spy_s = pd.Series(spy_aligned, index=df.index)

    horizon_rs = {}
    for h in horizons:
        stk_cr = _cum_ret(stock_s, h)
        spy_cr = _cum_ret(spy_s, h)
        rs = stk_cr - spy_cr
        col = f"ext3_relative_strength_{h}d"
        rs_cols[col] = rs.values
        horizon_rs[h] = rs.values

    # -----------------------------------------------------------------------
    # Slope across the three RS horizon points (per row)
    # OLS of rs ~ [21, 63, 126] (x values), one regression per time step.
    # Vectorised: x is fixed, only y varies.
    # slope = (n * sum(xi*yi) - sum(xi)*sum(yi)) / (n * sum(xi^2) - (sum(xi))^2)
    # -----------------------------------------------------------------------
    x = np.array([21.0, 63.0, 126.0], dtype=np.float64)
    n = len(x)
    sum_x = x.sum()         # 210
    sum_x2 = (x ** 2).sum() # 21^2 + 63^2 + 126^2 = 441 + 3969 + 15876 = 20286
    denom = n * sum_x2 - sum_x ** 2  # 3 * 20286 - 210^2 = 60858 - 44100 = 16758

    # Stack y values: shape (3, n_rows)
    y_stack = np.vstack([
        horizon_rs[21],
        horizon_rs[63],
        horizon_rs[126],
    ]).astype(np.float64)  # (3, n_rows)

    sum_y = y_stack.sum(axis=0)               # (n_rows,)
    sum_xy = (x[:, None] * y_stack).sum(axis=0)  # (n_rows,)

    if denom == 0.0:
        slope = np.full(len(df), np.nan)
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom

    # Where any of the three horizon values is NaN, slope should also be NaN
    any_nan = (
        np.isnan(horizon_rs[21]) |
        np.isnan(horizon_rs[63]) |
        np.isnan(horizon_rs[126])
    )
    slope = np.where(any_nan, np.nan, slope)

    # -----------------------------------------------------------------------
    # Attach columns
    # -----------------------------------------------------------------------
    for col, vals in rs_cols.items():
        df[col] = vals

    df["ext3_relative_strength_slope"] = slope

    # Guard: replace inf/-inf with NaN
    for col in METADATA["produces"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df
