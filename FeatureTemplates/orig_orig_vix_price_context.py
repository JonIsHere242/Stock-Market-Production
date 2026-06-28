"""
orig_orig_vix_price_context.py  --  Faithful AlphaSensitivity port: VIX-adjusted
price-context + market-regime base features.

Ported verbatim (math-preserving) from
_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py :: calculate_vix_features
  - L1212-1218  Log_Return, Realized_Vol_10d, Realized_Vol_21d
  - L1307       percent_change_Close
  - L1325-1344  SMA50, Trend_Direction, Market_Regime

ORIGINAL column names are kept verbatim (NOT prefixed). METADATA name == file stem.

INDEX DEPENDENCE
----------------
Market_Regime branches on VIX_Close < 20, so it inherits index-coupling. The VIX
series is loaded through the shared FeatureTemplates/_indexes.py helper, whose
vix_daily_close() mirrors the monolith merge exactly:
    vix['Close'].resample('D').last().ffill()  ->  backward merge_asof on Date  ->  ffill.
Everything else here is pure OHLCV (Close), so it lands at the float32 floor.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the underscore-prefixed shared index helper by path (auto-discovery
# skips it, so import it explicitly the way vix_features.py does).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location("_indexes", _Path(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


METADATA = {
    "name":        "orig_orig_vix_price_context",
    "description": "AlphaSensitivity VIX-adjusted price-context + market-regime base "
                   "features (Log_Return, realized vol, SMA50 trend, VIX regime).",
    "requires":    ["Date", "Close"],
    "produces":    [
        "Log_Return",
        "Realized_Vol_10d",
        "Realized_Vol_21d",
        "percent_change_Close",
        "SMA50",
        "Trend_Direction",
        "Market_Regime",
    ],
    "tags":        ["market_regime", "volatility"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    new = {}

    # --- VIX-ADJUSTED PRICE INDICATORS (L1212-1218) ------------------------
    # Log return (today vs yesterday)
    new["Log_Return"] = np.log(close / close.shift(1))

    # Realized volatility over rolling windows, annualized by sqrt(252)
    for window in [10, 21]:
        new[f"Realized_Vol_{window}d"] = new["Log_Return"].rolling(
            window, min_periods=window
        ).std() * np.sqrt(252)

    # --- RISK ADJUSTMENT FEATURES (L1307) ----------------------------------
    new["percent_change_Close"] = close.pct_change(fill_method=None)

    # --- MARKET REGIME CLASSIFICATION (L1325-1344) -------------------------
    # 50-day SMA
    new["SMA50"] = close.rolling(50, min_periods=50).mean()

    # Trend direction: +1 if Close > SMA50 else -1; NaN where SMA50 is NaN
    new["Trend_Direction"] = pd.Series(
        np.where(
            new["SMA50"].notna(),
            np.where(close > new["SMA50"], 1, -1),
            np.nan,
        ),
        index=df.index,
    )

    # VIX_Close (index-coupled) -- merged the same way as the monolith.
    vix_daily = _indexes.vix_daily_close()  # ['Date','vix_close'], ascending
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    vc = pd.Series(merged["vix_close"].ffill().to_numpy(), index=df.index)

    # Market regime:
    # 0: Low vol + Uptrend, 1: Low vol + Downtrend,
    # 2: High vol + Uptrend, 3: High vol + Downtrend
    new["Market_Regime"] = pd.Series(
        np.where(
            new["Trend_Direction"].notna(),
            np.where(
                vc < 20,
                np.where(new["Trend_Direction"] > 0, 0, 1),  # Low vol regimes
                np.where(new["Trend_Direction"] > 0, 2, 3),  # High vol regimes
            ),
            np.nan,
        ),
        index=df.index,
    )

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
