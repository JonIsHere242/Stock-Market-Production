from __future__ import annotations
import numpy as np
import pandas as pd
import importlib.util as _ilu
from pathlib import Path as _P

# -- load _indexes helper --
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ff06290000h_beta_risk_bawa_lindenberg_lpm_beta_spy_250",
    "description": (
        "Bawa-Lindenberg lower-partial-moment (LPM) beta vs SPY over a 250-day rolling window. "
        "LPM-beta = E[(r - mean_r) * (m - tau) | m < tau] / E[(m - tau)^2 | m < tau], where tau=0 "
        "and m=SPY daily return. Captures co-movement only in falling markets (downside semivariance). "
        "Requires >=20 downside days in the window; otherwise NaN. "
        "Also produces a 60-day fast LPM-beta for a short-horizon variant, and the ratio "
        "(fast/slow - 1) as a beta-regime shift signal. Per-ticker proxy; vectorised."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06290000h_lpm_beta_250",
        "ff06290000h_lpm_beta_60",
        "ff06290000h_lpm_beta_regime",
    ],
    "tags": ["beta", "downside_risk", "lpm", "bawa_lindenberg", "spy", "rolling"],
    "version": "1.0.0",
    "author": "feature-factory/ff06290000h",
}

_MIN_DOWN = 20   # minimum downside days required
_WIN_SLOW = 250
_WIN_FAST = 60
_TAU = 0.0       # threshold; SPY return < 0 => downside day


def _lpm_beta_series(ret: np.ndarray, spy_ret: np.ndarray, window: int, min_down: int) -> np.ndarray:
    """
    Rolling Bawa-Lindenberg LPM beta.
    ret      : per-stock log returns, length n
    spy_ret  : SPY log returns, length n (aligned)
    Returns array of length n with np.nan where coverage insufficient.
    """
    n = len(ret)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        r_win = ret[i - window + 1: i + 1]       # stock returns in window
        m_win = spy_ret[i - window + 1: i + 1]   # SPY returns in window

        # downside mask: days where SPY return < tau (0)
        down = m_win < _TAU

        n_down = int(np.sum(down))
        if n_down < min_down:
            continue

        m_down = m_win[down]           # SPY returns on downside days
        r_down = r_win[down]           # stock returns on downside days

        # valid (non-nan) rows only
        valid = ~(np.isnan(m_down) | np.isnan(r_down))
        if valid.sum() < min_down:
            continue

        m_d = m_down[valid]
        r_d = r_down[valid]

        # excess SPY below tau
        m_excess = m_d - _TAU           # m - tau  (negative values)

        denom = np.mean(m_excess ** 2)
        if denom < 1e-12:
            out[i] = 0.0
            continue

        # mean stock return over full window (not just downside), per spec
        mean_r = np.nanmean(r_win)
        numer = np.mean((r_d - mean_r) * m_excess)

        out[i] = numer / denom

    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # initialise outputs to NaN on every code path
    df["ff06290000h_lpm_beta_250"] = np.nan
    df["ff06290000h_lpm_beta_60"] = np.nan
    df["ff06290000h_lpm_beta_regime"] = np.nan

    if len(df) < _WIN_FAST + 5:
        return df

    # --- get SPY closes, merge backward-safe ---
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or len(spy_close) == 0:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    work = df[["Date"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # align back to original order
    work = work.set_index(work.index)  # keep positional index

    spy_vals = work["spy_close"].values
    close_vals = df["Close"].values

    # log returns (shift-forward = past relative to current bar; no lookahead)
    stock_ret = np.full(len(close_vals), np.nan)
    stock_ret[1:] = np.diff(np.log(np.where(close_vals > 0, close_vals, np.nan)))

    spy_ret = np.full(len(spy_vals), np.nan)
    spy_vals_safe = np.where(spy_vals > 0, spy_vals, np.nan)
    spy_ret[1:] = np.diff(np.log(spy_vals_safe))

    # compute slow (250) and fast (60) LPM betas
    beta_slow = _lpm_beta_series(stock_ret, spy_ret, _WIN_SLOW, _MIN_DOWN)
    beta_fast = _lpm_beta_series(stock_ret, spy_ret, _WIN_FAST, max(5, _MIN_DOWN // 4))

    df["ff06290000h_lpm_beta_250"] = beta_slow
    df["ff06290000h_lpm_beta_60"] = beta_fast

    # regime shift: fast/slow - 1 (positive = downside beta rising recently)
    with np.errstate(divide="ignore", invalid="ignore"):
        regime = np.where(
            np.abs(beta_slow) > 1e-9,
            beta_fast / beta_slow - 1.0,
            np.nan,
        )
    regime = np.where(np.isinf(regime), np.nan, regime)
    df["ff06290000h_lpm_beta_regime"] = regime

    return df
