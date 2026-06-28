"""
Growth-defensive style timing features derived from:
  "Continuous Timing Signals for Growth-Defensive Style Allocation:
   Factor Attribution, Risk Matching, and Out-of-Sample Evidence" (arXiv 2605.20636).

The paper builds a smooth allocation score from:
  - Rate relief (rate falling below recent avg) — not available, omitted
  - SPY drawdown depth (depth of market pullback from peak)
  - High-VIX stress relief (VIX falling from elevated levels)
  - Growth-crowding penalty (VIX × beta proxy; growth names punished when VIX rises)

Interaction terms are smoothed via softplus; final score through tanh.

Per-ticker proxy:
  - Market-level features (SPY drawdown, VIX level/momentum) from _indexes.py
  - Growth crowding: per-ticker price momentum interacted with VIX
  - Smooth score assembled per the paper's formula (softplus + tanh)
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import pandas as pd
import numpy as np

# Load _indexes.py by file path (same pattern as vix_features.py)
_spec = _ilu.spec_from_file_location(
    "_indexes",
    _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "paper_2605_20636_growth_defensive_timing",
    "description": (
        "Continuous growth-defensive style timing score from SPY drawdown depth, "
        "VIX stress relief, and growth-crowding penalty (softplus + tanh); "
        "proxy per-ticker implementation of arXiv 2605.20636."
    ),
    "requires":    ["Close"],
    "produces": [
        "gdt_spy_drawdown",
        "gdt_spy_drawdown_depth",
        "gdt_vix_stress_relief",
        "gdt_vix_stress_relief_raw",
        "gdt_crowding_penalty",
        "gdt_smooth_score",
        "gdt_ewma_score",
    ],
    "tags":        ["market_regime", "momentum", "volatility", "experimental"],
    "version":     "1.0",
    "author":      "paper:2605.20636",
}


def _softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Softplus: log(1 + exp(beta*x)) / beta — smooth approximation of ReLU."""
    return np.log1p(np.exp(np.clip(beta * x, -50, 50))) / beta


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Load SPY and VIX index data
    # ------------------------------------------------------------------ #
    try:
        spy_close = _indexes.index_close("SPY")   # DatetimeIndex Series
        vix_daily = _indexes.vix_daily_close()     # DataFrame: Date, vix_close
    except Exception:
        spy_close = pd.Series(dtype=float)
        vix_daily = pd.DataFrame(columns=["Date", "vix_close"])

    # Ensure Date column is present for merge
    df_work = df.copy() if "Date" not in df.columns else df
    if "Date" not in df_work.columns:
        df_work = df_work.reset_index()

    date_col = pd.to_datetime(df_work["Date"])

    # ------------------------------------------------------------------ #
    # 2. SPY drawdown depth
    # ------------------------------------------------------------------ #
    if len(spy_close) > 0:
        spy_s = spy_close.copy()
        spy_s.index = pd.to_datetime(spy_s.index)
        spy_aligned = spy_s.reindex(date_col.values, method="ffill")
        spy_arr = spy_aligned.values.astype(np.float64)
    else:
        spy_arr = np.full(len(df), np.nan)

    # Rolling 252-day peak → drawdown from peak
    spy_series = pd.Series(spy_arr, index=df.index)
    spy_peak   = spy_series.rolling(252, min_periods=20).max()
    spy_dd     = (spy_series - spy_peak) / spy_peak.replace(0, np.nan)  # <= 0
    spy_dd_depth = spy_dd.clip(-1, 0)  # negative depth, e.g. -0.20 = 20% down

    df["gdt_spy_drawdown"]       = spy_dd
    df["gdt_spy_drawdown_depth"] = spy_dd_depth

    # ------------------------------------------------------------------ #
    # 3. VIX stress relief signal
    # ------------------------------------------------------------------ #
    if len(vix_daily) > 0:
        vix_merged = pd.merge_asof(
            pd.DataFrame({"Date": pd.to_datetime(date_col.values)}),
            vix_daily.rename(columns={"Date": "Date"}),
            on="Date", direction="backward"
        )
        vix_arr = vix_merged["vix_close"].values.astype(np.float64)
    else:
        vix_arr = np.full(len(df), np.nan)

    vix_s   = pd.Series(vix_arr, index=df.index)
    # VIX stress relief: VIX falling from recent peak (positive = stress easing)
    vix_peak_20 = vix_s.rolling(20, min_periods=5).max()
    vix_relief  = (vix_peak_20 - vix_s) / vix_peak_20.replace(0, np.nan)  # >= 0 when VIX falling
    # Also raw VIX z-score (inverted: high VIX = stress)
    vix_roll_mean = vix_s.rolling(63, min_periods=20).mean()
    vix_roll_std  = vix_s.rolling(63, min_periods=20).std()
    vix_raw_z     = -(vix_s - vix_roll_mean) / vix_roll_std.replace(0, np.nan)  # positive = low VIX

    df["gdt_vix_stress_relief"]     = vix_relief
    df["gdt_vix_stress_relief_raw"] = vix_raw_z

    # ------------------------------------------------------------------ #
    # 4. Growth-crowding penalty: per-ticker momentum × VIX regime
    #    When VIX is high, high-momentum (crowded growth) names get penalised.
    # ------------------------------------------------------------------ #
    ret = df["Close"].pct_change()
    mom_20 = ret.rolling(20, min_periods=10).mean() / ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    # Crowding penalty: negative when vix high AND momentum high (growth crowded)
    crowding = -mom_20 * vix_s.rolling(20, min_periods=5).mean() / 20.0  # normalise VIX to ~1 range
    df["gdt_crowding_penalty"] = crowding.clip(-5, 5)

    # ------------------------------------------------------------------ #
    # 5. Smooth allocation score (per paper's formula)
    #    score = tanh(
    #        softplus(spy_relief) + softplus(vix_relief) - softplus(crowding_penalty)
    #    )
    # ------------------------------------------------------------------ #
    spy_relief_arr  = (-spy_dd_depth.fillna(0)).values   # positive = drawdown (buy signal)
    vix_rel_arr     = vix_relief.fillna(0).values
    crowd_pen_arr   = crowding.fillna(0).values

    raw_score = (
        _softplus( spy_relief_arr * 3.0)    # drawdown → growth risk
        + _softplus(vix_rel_arr * 2.0)       # VIX easing → risk-on
        - _softplus(crowd_pen_arr * 2.0)     # crowding → avoid growth
    )
    # tanh maps raw_score to (-1, 1); subtract a fixed offset (log(2)/1=0.693)
    # which is softplus(0) — the baseline when all inputs are zero.
    # No full-series statistics used here: purely element-wise.
    smooth_score = np.tanh(raw_score - np.log(2.0))
    df["gdt_smooth_score"] = smooth_score

    # EWMA of the smooth score (paper smooths final weights with EWMA)
    df["gdt_ewma_score"] = pd.Series(smooth_score, index=df.index).ewm(span=10, min_periods=5).mean()

    return df
