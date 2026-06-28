"""
ext4_vrp_vix — Variance Risk Premium proxy vs VIX

Per-ticker 21-day realized vol (annualised) minus the contemporaneous VIX level
(divided by 100 to convert to decimal). Positive values mean the stock is
realising more vol than the broad-market implied vol (VIX), negative values
mean it is quieter. Also emits the 63-day rolling percentile rank of that spread
as a dynamics/slope variant.

Note: True VRP requires implied vol from options; here we use VIX as the
market-wide IV proxy and compare it to realised vol — a per-ticker
stock-vs-market VRP approximation faithful to the spec.
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
METADATA = {
    "name": "ext4_vrp_vix",
    "description": (
        "Stock-vs-market variance risk premium proxy. "
        "21-day annualised realized volatility of the stock minus the "
        "contemporaneous VIX level / 100 (VIX as broad-market IV proxy). "
        "Positive = stock realising more vol than market-implied; "
        "negative = stock quieter than implied. "
        "Also produces the 63-day rolling percentile rank of the spread. "
        "Per-ticker OHLCV proxy for true VRP (true VRP needs per-stock options IV)."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_vrp_vix_spread",       # rvol_21d - vix/100  (decimal vol units)
        "ext4_vrp_vix_pct_rank",     # 63d rolling percentile of spread
        "ext4_vrp_vix_rvol21",       # 21d annualised realized vol (standalone)
    ],
    "tags": ["volatility", "vrp", "vix", "realized_vol", "risk_premium"],
    "version": "1.0.0",
    "author": "Round-5 expansion (osap_betatailrisk); spec ext4_vrp_vix",
}

# ---------------------------------------------------------------------------
_TRADING_DAYS = 252
_RVOL_WIN = 21        # realized vol window (trading days)
_PCT_WIN   = 63       # rolling pct-rank window


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Add ext4_vrp_vix_* columns to df (in-place then returned)."""

    # ---- 1. 21-day annualised realized volatility -------------------------
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    rvol21 = log_ret.rolling(_RVOL_WIN, min_periods=_RVOL_WIN).std() * np.sqrt(_TRADING_DAYS)

    # ---- 2. VIX (backward-safe merge) -------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vix_df = _indexes.vix_daily_close()          # [["Date","vix_close"]]
            vix_df["Date"] = pd.to_datetime(vix_df["Date"])
            tmp = df[["Date"]].copy()
            tmp["Date"] = pd.to_datetime(tmp["Date"])
            tmp = pd.merge_asof(
                tmp.sort_values("Date"),
                vix_df.sort_values("Date"),
                on="Date",
                direction="backward",
            )
            # Restore original row order
            tmp.index = df.index
            vix_series = tmp["vix_close"]
        except Exception:
            vix_series = pd.Series(np.nan, index=df.index)

    vix_decimal = vix_series / 100.0   # VIX is in percent, convert to decimal

    # ---- 3. Spread = rvol - vix_implied ----------------------------------
    spread = rvol21 - vix_decimal

    # ---- 4. 63-day rolling percentile rank of spread ---------------------
    def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
        """Vectorised rolling percentile rank using numpy sliding_window_view."""
        arr = s.to_numpy(dtype=float)
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            window_vals = arr[i - window + 1: i + 1]
            cur = arr[i]
            if np.isnan(cur):
                continue
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < 2:
                continue
            result[i] = float(np.sum(valid <= cur) - 1) / float(len(valid) - 1)
        return pd.Series(result, index=s.index)

    pct_rank = _rolling_pct_rank(spread, _PCT_WIN)

    # ---- 5. Guard: replace inf/-inf with nan ------------------------------
    spread = spread.replace([np.inf, -np.inf], np.nan)
    rvol21 = rvol21.replace([np.inf, -np.inf], np.nan)
    pct_rank = pct_rank.replace([np.inf, -np.inf], np.nan)

    # ---- 6. Assign -------------------------------------------------------
    df["ext4_vrp_vix_rvol21"]   = rvol21
    df["ext4_vrp_vix_spread"]   = spread
    df["ext4_vrp_vix_pct_rank"] = pct_rank

    return df
