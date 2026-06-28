"""
Global-persistence / local-residual AR decomposition — per-ticker proxy.

Based on arXiv 2604.09821: "Global Persistence, Local Residual Structure:
Forecasting Heterogeneous Investment Panels".

The paper fits a two-stage architecture on a macro+firm panel:
  Stage 1: global pooled AR(1) captures shared persistence across all actors.
  Stage 2: actor-specific local models capture residual idiosyncratic dynamics.

The key insight: factoring out global persistence before fitting local structure
improves OOS R² from 0.630 → 0.677 across heterogeneous panels.

Per-ticker proxy: we treat the broad market (SPY) as the "global pool".
  - Global persistence: SPY log-return AR(1) coefficient (rolling).
  - Global-persistence-adjusted return: subtract the SPY-AR(1) contribution.
  - Local AR(1): fit AR(1) to the global-adjusted return residuals (per-ticker).
  - Local autocorrelation of residuals: measures residual idiosyncratic structure.

This captures the paper's central finding — idiosyncratic dynamics beyond
shared market persistence are the actionable signal.
"""

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Load _indexes helper by file path
# --------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

# --------------------------------------------------------------------------
METADATA = {
    "name": "paper_2604_09821_global_local_ar_residual",
    "description": (
        "Two-stage AR decomposition: global SPY-persistence subtracted first, "
        "then local AR(1) fitted to idiosyncratic residuals; proxy for the "
        "global-pooled AR + local-residual panel architecture in arXiv 2604.09821."
    ),
    "requires": ["Close"],
    "produces": [
        "glar_spy_ar1_phi_60d",
        "glar_global_adj_ret",
        "glar_local_ar1_phi_60d",
        "glar_local_resid_ac1_60d",
        "glar_idio_momentum_20d",
        "glar_two_stage_resid_zscore_60d",
    ],
    "tags": ["mean_reversion", "momentum", "market_regime", "experimental"],
    "version": "1.0",
    "author": "paper:2604.09821",
}

_WINDOW = 60
_MIN_OBS = 20
_SHORT = 20


def _rolling_ar1_phi(ret_arr: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """Rolling OLS AR(1) coefficient for a 1-D return array."""
    n = len(ret_arr)
    phi = np.full(n, np.nan)
    for i in range(1, n):
        start = max(0, i - window + 1)
        seg = ret_arr[start: i + 1]
        mask = ~np.isnan(seg)
        seg = seg[mask]
        k = len(seg)
        if k < min_obs:
            continue
        y = seg[1:]
        x = seg[:-1]
        if len(x) < 4:
            continue
        x_mean = x.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx < 1e-14:
            continue
        phi[i] = ((x - x_mean) * (y - y.mean())).sum() / ss_xx
    return phi


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Ticker log-returns
    # ------------------------------------------------------------------
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).values.astype(np.float64)

    # ------------------------------------------------------------------
    # 2. SPY log-returns (global pool proxy)
    # ------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        if "Date" in df.columns:
            date_col = pd.to_datetime(df["Date"])
        else:
            date_col = pd.to_datetime(df.index)
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "SPY_Close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        tmp = pd.DataFrame({"Date": date_col.values}, index=df.index).sort_values("Date")
        merged = pd.merge_asof(
            tmp[["Date"]],
            spy_df,
            on="Date",
            direction="backward",
        )
        spy_close_series = merged["SPY_Close"].values
        # Reindex back
        spy_close_aligned = pd.Series(spy_close_series, index=tmp.index).reindex(df.index)
    except Exception:
        spy_close_aligned = pd.Series(np.nan, index=df.index)

    spy_log_ret = np.log(spy_close_aligned / spy_close_aligned.shift(1)).values.astype(np.float64)

    # ------------------------------------------------------------------
    # 3. Rolling AR(1) phi for SPY (global persistence)
    # ------------------------------------------------------------------
    spy_phi = _rolling_ar1_phi(spy_log_ret, _WINDOW, _MIN_OBS)
    df["glar_spy_ar1_phi_60d"] = spy_phi

    # ------------------------------------------------------------------
    # 4. Global-persistence-adjusted return
    #    r_adj_t = r_ticker_t - spy_phi_{t-1} * r_SPY_t
    # ------------------------------------------------------------------
    spy_phi_lag = np.roll(spy_phi, 1)
    spy_phi_lag[0] = np.nan

    spy_contribution = spy_phi_lag * spy_log_ret
    global_adj_ret = log_ret - spy_contribution
    df["glar_global_adj_ret"] = global_adj_ret

    # ------------------------------------------------------------------
    # 5. Local AR(1) on global-adjusted residuals (idiosyncratic structure)
    # ------------------------------------------------------------------
    local_phi = _rolling_ar1_phi(global_adj_ret, _WINDOW, _MIN_OBS)
    df["glar_local_ar1_phi_60d"] = local_phi

    # ------------------------------------------------------------------
    # 6. Local residual autocorrelation (AC1 of the global-adj returns)
    # ------------------------------------------------------------------
    n = len(global_adj_ret)
    ac1_arr = np.full(n, np.nan)
    for i in range(_WINDOW, n):
        start = i - _WINDOW + 1
        seg = global_adj_ret[start: i + 1]
        mask = ~np.isnan(seg)
        seg = seg[mask]
        if len(seg) < _MIN_OBS:
            continue
        y = seg[1:]
        x = seg[:-1]
        x_m, y_m = x.mean(), y.mean()
        denom = np.sqrt(((x - x_m) ** 2).sum() * ((y - y_m) ** 2).sum())
        if denom < 1e-14:
            continue
        ac1_arr[i] = ((x - x_m) * (y - y_m)).sum() / denom
    df["glar_local_resid_ac1_60d"] = ac1_arr

    # ------------------------------------------------------------------
    # 7. Idiosyncratic momentum: cumulative global-adjusted return over
    #    the short window (pure idiosyncratic price momentum)
    # ------------------------------------------------------------------
    idio_mom = (
        pd.Series(global_adj_ret, index=df.index)
        .rolling(_SHORT, min_periods=5)
        .sum()
    )
    df["glar_idio_momentum_20d"] = idio_mom

    # ------------------------------------------------------------------
    # 8. Two-stage residual z-score: deviation of current global-adj
    #    return from its rolling mean, scaled by its rolling std.
    # ------------------------------------------------------------------
    gadj_s = pd.Series(global_adj_ret, index=df.index)
    roll_mean = gadj_s.rolling(_WINDOW, min_periods=_MIN_OBS).mean()
    roll_std = gadj_s.rolling(_WINDOW, min_periods=_MIN_OBS).std()
    zscore = (gadj_s - roll_mean) / roll_std.replace(0.0, np.nan)
    df["glar_two_stage_resid_zscore_60d"] = zscore

    return df
