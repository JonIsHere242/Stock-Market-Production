"""
orig_orig_canbuy_momentum.py  --  Faithful AlphaSensitivity port of the can_buy /
momentum signal triplet.

Ported verbatim (math-preserving) from
_backups/_old_versions/3__AlphaSensitivity_pre_tsz_20260530.py :
  - calculate_price_momentum_features  (L3114-3148)
      rapid_price_change_5d   L3130   close / close.shift(5) - 1
      trend_consistency_5d    L3140-3146  pct_change rolling-share-positive
  - calculate_vix_regime_features      (L2971-3006)
      vix_momentum_5d         L2996   VIX_Close.pct_change(5)

ORIGINAL column names kept verbatim (NOT prefixed). METADATA name == file stem.

INDEX DEPENDENCE
----------------
vix_momentum_5d = VIX_Close.pct_change(5). VIX_Close is NOT in the per-ticker OHLCV
frame; it is reconstructed via the shared FeatureTemplates/_indexes.py helper exactly
the way orig_orig_vix_price_context.py does (vix_daily_close() ->
backward merge_asof on Date -> ffill). rapid_price_change_5d and trend_consistency_5d
are pure OHLCV (Close) and land at the float32 floor.
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
    "name":        "orig_orig_canbuy_momentum",
    "description": "AlphaSensitivity can_buy/momentum triplet: rapid 5d price change, "
                   "5d trend consistency (share of up days), and 5d VIX momentum.",
    "requires":    ["Date", "Close"],
    "produces":    [
        "rapid_price_change_5d",
        "trend_consistency_5d",
        "vix_momentum_5d",
    ],
    "tags":        ["momentum", "can_buy", "vix"],
    "version":     "1.0",
    "author":      "alphasens port",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    new = {}

    # --- Rapid price movement detection (L3130) ----------------------------
    new["rapid_price_change_5d"] = close / close.shift(5) - 1

    # --- Trend consistency 5d (L3140-3146) ---------------------------------
    # share of the last 5 daily returns that were positive; min_periods=5//2=2
    daily_returns = close.pct_change()
    new["trend_consistency_5d"] = daily_returns.rolling(
        5, min_periods=5 // 2
    ).apply(lambda x: (x > 0).mean(), raw=True)

    # --- VIX momentum 5d (L2996) -------------------------------------------
    # VIX_Close (index-coupled) -- merged the same way as the monolith.
    vix_daily = _indexes.vix_daily_close()  # ['Date','vix_close'], ascending
    dates = pd.to_datetime(df["Date"])
    tmp = pd.DataFrame({"Date": dates.values})
    merged = pd.merge_asof(tmp, vix_daily, on="Date", direction="backward")
    vix_close = pd.Series(merged["vix_close"].ffill().to_numpy(), index=df.index)
    new["vix_momentum_5d"] = vix_close.pct_change(5)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
