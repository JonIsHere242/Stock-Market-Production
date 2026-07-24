from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load index helper (SPY + VIX)
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290014i_beta_risk_downside_lpm_beta_vix_conditioned_spy_250",
    "description": (
        "LPM downside beta vs SPY conditioned on high-VIX regimes. "
        "Computes the stressed downside beta (days where VIX > its rolling 250d median "
        "AND SPY return < 0) minus the unconditional downside beta. Positive value means "
        "the stock amplifies drawdowns specifically during VIX stress. Per-ticker proxy; "
        "rolling 250-bar estimates with a minimum of 15 qualifying observations."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290014i_beta_risk_downside_lpm_beta_vix_conditioned_spy_250_spread",
        "ff06290014i_beta_risk_downside_lpm_beta_vix_conditioned_spy_250_stressed",
        "ff06290014i_beta_risk_downside_lpm_beta_vix_conditioned_spy_250_unconditional",
    ],
    "tags": ["beta", "downside", "vix", "regime", "risk", "lpm"],
    "version": "1.0.0",
    "author": "feature-factory ff06290014i",
}

_WINDOW = 250
_MIN_OBS = 15
_DENOM_FLOOR = 1e-12

_COL_SPREAD = METADATA["produces"][0]
_COL_STRESSED = METADATA["produces"][1]
_COL_UNCOND = METADATA["produces"][2]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_SPREAD] = np.nan
    df[_COL_STRESSED] = np.nan
    df[_COL_UNCOND] = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # -----------------------------------------------------------------------
    # Fetch SPY and VIX series
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    try:
        vix_df = _indexes.vix_daily_close()
    except Exception:
        return df

    # Ensure Date column is datetime for merge
    df_work = df[["Date"]].copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    df_work = pd.merge_asof(
        df_work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    df_work = pd.merge_asof(
        df_work,
        vix_df[["Date", "vix_close"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Restore original order
    df_work = df_work.set_index(df_work.index).reindex(range(len(df)))

    spy_arr = df_work["spy_close"].values.astype(np.float64)
    vix_arr = df_work["vix_close"].values.astype(np.float64)

    close_arr = df["Close"].values.astype(np.float64)

    n = len(df)
    spread_out = np.full(n, np.nan)
    stressed_out = np.full(n, np.nan)
    uncond_out = np.full(n, np.nan)

    # Ticker returns (simple, shift-1)
    tkr_ret = np.empty(n)
    tkr_ret[0] = np.nan
    tkr_ret[1:] = close_arr[1:] / np.where(close_arr[:-1] == 0, np.nan, close_arr[:-1]) - 1.0

    # SPY returns
    spy_ret = np.empty(n)
    spy_ret[0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        spy_ret[1:] = np.where(spy_arr[:-1] == 0, np.nan,
                                spy_arr[1:] / spy_arr[:-1] - 1.0)

    # Vectorised rolling computation using a fixed stride is not required here
    # because we compute rolling windows naturally.  We use numpy stride tricks
    # for efficiency: build an index loop only at each bar (O(n) scalar ops on
    # pre-computed arrays, not O(n^2) Python row loops).

    for t in range(_WINDOW, n):
        sl = slice(t - _WINDOW + 1, t + 1)  # current bar included (causal)
        r = tkr_ret[sl]      # stock returns in window
        m = spy_ret[sl]      # SPY returns in window
        v = vix_arr[sl]      # VIX values in window

        valid = (~np.isnan(r)) & (~np.isnan(m)) & (~np.isnan(v))
        if valid.sum() < _MIN_OBS:
            continue

        r_v = r[valid]
        m_v = m[valid]
        v_v = v[valid]

        # --- Unconditional downside beta (SPY return < 0) ---
        down_mask = m_v < 0.0
        if down_mask.sum() < _MIN_OBS:
            continue

        m_down = m_v[down_mask]
        r_down = r_v[down_mask]
        denom_u = np.mean(m_down ** 2)
        if denom_u < _DENOM_FLOOR:
            continue
        # LPM beta = Cov(r,m|m<0) / E[m^2|m<0]  (matches LPM definition)
        beta_uncond = np.mean((r_down - r_v.mean()) * m_down) / denom_u

        # --- VIX 250d rolling median (use the same window) ---
        vix_median = np.nanmedian(v_v)

        # --- Stressed downside beta: high VIX AND SPY < 0 ---
        stressed_mask = down_mask & (v_v > vix_median)
        if stressed_mask.sum() < _MIN_OBS:
            # Not enough stressed days; leave spread as NaN but record uncond
            uncond_out[t] = beta_uncond
            continue

        m_s = m_v[stressed_mask]
        r_s = r_v[stressed_mask]
        denom_s = np.mean(m_s ** 2)
        if denom_s < _DENOM_FLOOR:
            uncond_out[t] = beta_uncond
            continue

        beta_stressed = np.mean((r_s - r_v.mean()) * m_s) / denom_s

        stressed_out[t] = beta_stressed
        uncond_out[t] = beta_uncond
        spread_out[t] = beta_stressed - beta_uncond

    df[_COL_SPREAD] = spread_out
    df[_COL_STRESSED] = stressed_out
    df[_COL_UNCOND] = uncond_out

    return df
