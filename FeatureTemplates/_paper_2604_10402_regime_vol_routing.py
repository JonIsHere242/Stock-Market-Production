"""
Risk-sensitive specialist routing for volatility forecasting — per-ticker proxy.

Based on arXiv 2604.10402: "Risk-Sensitive Specialist Routing for Volatility Forecasting".

The paper uses multiple specialist forecasters each tuned to a distinct market
regime (calm / stressed), combined via state-dependent gating based on online
risk-sensitive evaluation. Their key finding: the best volatility model is
regime-dependent, not stable across all states.

Per-ticker proxy: we implement the structural insight without cross-sectional
routing. We compute:
  1. Realised volatility in two regimes gated by VIX level (calm vs stressed).
  2. A "routing signal" = exponentially-smoothed ratio of stressed-vol to
     calm-vol (regime-relative vol spread).
  3. VIX-gated rolling volatility — the "specialist" forecast for each regime.
  4. A vol-forecast error signal based on whether realised vol exceeded the
     regime-conditioned expectation (proxy for the paper's QLIKE/underprediction
     loss reduction).

All computed purely from per-ticker OHLCV + VIX index (backward-merged, no
lookahead). Requires the _indexes helper.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Load _indexes helper by file path (no package import needed)
# --------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# --------------------------------------------------------------------------
METADATA = {
    "name": "paper_2604_10402_regime_vol_routing",
    "description": (
        "Regime-gated volatility forecasts (calm vs stressed regimes via VIX) "
        "inspired by risk-sensitive specialist routing (arXiv 2604.10402); "
        "proxy: per-ticker OHLCV + VIX — no cross-sectional routing."
    ),
    "requires": ["Close", "High", "Low"],
    "produces": [
        "rsrv_calm_vol_20d",
        "rsrv_stressed_vol_20d",
        "rsrv_regime_flag",
        "rsrv_vol_spread_ratio",
        "rsrv_vol_surprise",
        "rsrv_gated_vol_forecast",
    ],
    "tags": ["volatility", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2604.10402",
}


# VIX threshold separating calm from stressed regimes (historically ~20 is neutral)
_VIX_STRESSED_THRESHOLD = 20.0
_SHORT_WINDOW = 5   # days for intra-regime short vol
_LONG_WINDOW = 20   # days for base vol


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Merge VIX (backward-safe)                                        #
    # ------------------------------------------------------------------ #
    try:
        vix_df = _indexes.vix_daily_close()  # columns: Date, vix_close
    except Exception:
        vix_df = pd.DataFrame(columns=["Date", "vix_close"])

    # Ensure Date column (not index) in df
    if "Date" in df.columns:
        date_col = pd.to_datetime(df["Date"])
    else:
        date_col = pd.to_datetime(df.index)

    if len(vix_df) > 0:
        vix_df = vix_df.copy()
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        tmp = df.copy()
        tmp["_date_sort"] = date_col.values
        tmp = tmp.sort_values("_date_sort")
        merged = pd.merge_asof(
            tmp[["_date_sort"]],
            vix_df.rename(columns={"Date": "_date_sort"}),
            on="_date_sort",
            direction="backward",
        )
        vix_series = merged["vix_close"].values
        # Reindex back to original order
        order = tmp.index
        vix_s = pd.Series(vix_series, index=order).reindex(df.index)
    else:
        vix_s = pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------ #
    # 2. Log returns + True Range for realised vol                        #
    # ------------------------------------------------------------------ #
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    hl_range = np.log(df["High"] / df["Low"])  # simple range proxy for realised vol

    # ------------------------------------------------------------------ #
    # 3. Regime flag: 1 = stressed (VIX >= threshold), 0 = calm          #
    # ------------------------------------------------------------------ #
    regime_flag = (vix_s >= _VIX_STRESSED_THRESHOLD).astype(float)
    regime_flag = regime_flag.where(vix_s.notna(), other=np.nan)
    df["rsrv_regime_flag"] = regime_flag

    # ------------------------------------------------------------------ #
    # 4. Calm-only and stressed-only volatility: rolling realised vol     #
    #    gated to the respective regime windows.                          #
    # ------------------------------------------------------------------ #
    # We compute rolling std of log returns in each regime by setting the
    # other regime's returns to NaN, then using rolling with min_periods.
    calm_ret = log_ret.where(regime_flag == 0)
    stressed_ret = log_ret.where(regime_flag == 1)

    # Rolling std — window applies over ALL calendar bars (not just regime bars),
    # consistent with a real walk-forward approach. min_periods=3 to allow early.
    calm_vol = calm_ret.rolling(_LONG_WINDOW, min_periods=3).std() * np.sqrt(252)
    stressed_vol = stressed_ret.rolling(_LONG_WINDOW, min_periods=3).std() * np.sqrt(252)

    # Forward-fill within each series so the last observed value persists
    calm_vol_ffill = calm_vol.ffill()
    stressed_vol_ffill = stressed_vol.ffill()

    df["rsrv_calm_vol_20d"] = calm_vol_ffill
    df["rsrv_stressed_vol_20d"] = stressed_vol_ffill

    # ------------------------------------------------------------------ #
    # 5. Routing signal: ratio of stressed-vol to calm-vol               #
    # ------------------------------------------------------------------ #
    # High ratio → stressed regime is much more volatile than calm periods.
    ratio_raw = stressed_vol_ffill / calm_vol_ffill.replace(0.0, np.nan)
    # EWM-smooth the ratio (proxy for online risk-sensitive evaluation)
    ratio_smooth = ratio_raw.ewm(span=10, min_periods=3, adjust=False).mean()
    df["rsrv_vol_spread_ratio"] = ratio_smooth

    # ------------------------------------------------------------------ #
    # 6. Gated vol forecast: use the regime-appropriate specialist        #
    # ------------------------------------------------------------------ #
    gated = np.where(
        regime_flag == 1,
        stressed_vol_ffill,
        calm_vol_ffill,
    )
    gated = pd.Series(gated, index=df.index)
    # Fall back to overall vol where regime unknown
    overall_vol = log_ret.rolling(_LONG_WINDOW, min_periods=3).std() * np.sqrt(252)
    gated = gated.where(regime_flag.notna(), other=overall_vol)
    df["rsrv_gated_vol_forecast"] = gated

    # ------------------------------------------------------------------ #
    # 7. Vol surprise: actual short-window realised vol vs gated forecast #
    # ------------------------------------------------------------------ #
    actual_short_vol = log_ret.rolling(_SHORT_WINDOW, min_periods=2).std() * np.sqrt(252)
    # Shift forecast forward by 1 (forecast made yesterday for today)
    lagged_forecast = gated.shift(1)
    surprise = actual_short_vol - lagged_forecast
    df["rsrv_vol_surprise"] = surprise

    return df
