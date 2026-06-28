"""
Composite tail-risk score (coskew + cokurt + downbeta) vs SPY.

Per-ticker proxy: combines three rolling tail-risk loadings against the SPY
market return, each standardised to a z-score over the trailing 252-day
window, then equal-weight averaged into a composite.  Also produces the
60-day change in the composite to capture momentum in tail-risk loading.

Source: Round-5 expansion (osap_betatailrisk)
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: _indexes (SPY Close)
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
    "name": "ext4_tail_risk_composite",
    "description": (
        "Composite tail-risk score vs SPY: equal-weight average of rolling "
        "252-day z-scored downside-beta, negative coskewness, and cokurtosis "
        "relative to SPY daily returns.  Downside beta uses only days when "
        "SPY return < 0.  Coskewness captures co-movement in the third "
        "moment; cokurtosis in the fourth.  All three components are "
        "z-scored before combining so no single moment dominates.  Also "
        "produces the 60-day change in the composite score.  Per-ticker "
        "time-series proxy for a cross-sectional tail-beta concept."
    ),
    "requires": ["Close"],
    "produces": [
        "ext4_tail_risk_composite_score",   # rolling z-score composite
        "ext4_tail_risk_composite_chg60",   # 60-day change in composite
        "ext4_tail_risk_composite_downbeta",# raw downside-beta component (z-scored)
    ],
    "tags": ["tail-risk", "coskewness", "cokurtosis", "downside-beta", "market-beta"],
    "version": "1.0.0",
    "author": "Round-5 expansion (osap_betatailrisk)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 252        # rolling estimation window (calendar-day aligned)
_ZWIN   = 252        # z-score window (same)
_CHG_LAG = 60        # change look-back


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite tail-risk score for a single ticker."""

    # ------------------------------------------------------------------ #
    # 1. Align SPY index close via merge_asof (backward, lookahead-safe)  #
    # ------------------------------------------------------------------ #
    try:
        spy_series = _indexes.index_close("SPY")  # Series indexed by DatetimeIndex
        spy_df = spy_series.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        spy_df = pd.DataFrame(columns=["Date", "spy_close"])

    work = df[["Date", "Close"]].copy()
    work["Date"] = pd.to_datetime(work["Date"])

    if len(spy_df) > 0:
        spy_df = spy_df.sort_values("Date").reset_index(drop=True)
        work = pd.merge_asof(
            work.sort_values("Date"),
            spy_df,
            on="Date",
            direction="backward",
        )
        # restore original order
        work = work.set_index(df.index)
    else:
        work["spy_close"] = np.nan

    # ------------------------------------------------------------------ #
    # 2. Compute daily returns                                            #
    # ------------------------------------------------------------------ #
    stock_ret = work["Close"].pct_change()           # r_i,t
    spy_ret   = work["spy_close"].pct_change()       # r_m,t

    n = len(df)

    # ------------------------------------------------------------------ #
    # 3. Rolling window estimations using sliding_window_view             #
    # ------------------------------------------------------------------ #
    # Pre-fill output arrays with NaN
    downbeta_arr  = np.full(n, np.nan)
    coskew_arr    = np.full(n, np.nan)
    cokurt_arr    = np.full(n, np.nan)

    sr = stock_ret.to_numpy(dtype=float)
    mr = spy_ret.to_numpy(dtype=float)

    for t in range(_WINDOW - 1, n):
        s_win = sr[t - _WINDOW + 1 : t + 1]   # shape (WINDOW,)
        m_win = mr[t - _WINDOW + 1 : t + 1]

        # drop any NaN pairs
        valid = np.isfinite(s_win) & np.isfinite(m_win)
        sv = s_win[valid]
        mv = m_win[valid]

        if len(sv) < 30:
            continue

        # demean
        sm = sv - np.nanmean(sv)
        mm = mv - np.nanmean(mv)

        # ---- downside beta: cov(r_i, r_m | r_m < 0) / var(r_m | r_m < 0)
        down_mask = mv < 0
        if down_mask.sum() >= 10:
            sm_d = sm[down_mask]
            mm_d = mm[down_mask]
            var_m_d = np.mean(mm_d ** 2)
            if var_m_d > 0:
                downbeta_arr[t] = np.mean(sm_d * mm_d) / var_m_d

        # ---- coskewness: E[r_i * r_m^2] / (std_i * std_m^2)
        std_s = np.std(sv, ddof=1)
        std_m = np.std(mv, ddof=1)
        if std_s > 0 and std_m > 0:
            coskew_arr[t] = (
                np.mean(sm * mm ** 2) / (std_s * std_m ** 2)
            )
            # ---- cokurtosis: E[r_i * r_m^3] / (std_i * std_m^3)
            cokurt_arr[t] = (
                np.mean(sm * mm ** 3) / (std_s * std_m ** 3)
            )

    # ------------------------------------------------------------------ #
    # 4. Z-score each component over a trailing _ZWIN window             #
    # ------------------------------------------------------------------ #
    def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for t in range(window - 1, len(arr)):
            seg = arr[t - window + 1 : t + 1]
            valid = seg[np.isfinite(seg)]
            if len(valid) < 10:
                continue
            mu  = np.mean(valid)
            sig = np.std(valid, ddof=1)
            if sig > 0 and np.isfinite(arr[t]):
                out[t] = (arr[t] - mu) / sig
        return out

    z_downbeta = _rolling_zscore(downbeta_arr, _ZWIN)
    z_coskew   = _rolling_zscore(-coskew_arr, _ZWIN)   # negate: high neg coskew = more tail risk
    z_cokurt   = _rolling_zscore(cokurt_arr,  _ZWIN)

    # ------------------------------------------------------------------ #
    # 5. Equal-weight composite (average of available z-scores)           #
    # ------------------------------------------------------------------ #
    components = np.stack([z_downbeta, z_coskew, z_cokurt], axis=1)  # (n, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        composite = np.nanmean(components, axis=1)

    # Guard: if all components NaN → NaN (nanmean returns nan already, but
    # set explicitly for rows where all three are NaN)
    all_nan = np.all(~np.isfinite(components), axis=1)
    composite[all_nan] = np.nan

    # Replace inf (guard)
    composite  = np.where(np.isfinite(composite),  composite,  np.nan)
    z_downbeta = np.where(np.isfinite(z_downbeta), z_downbeta, np.nan)

    # ------------------------------------------------------------------ #
    # 6. 60-day change                                                    #
    # ------------------------------------------------------------------ #
    chg60 = np.full(n, np.nan)
    for t in range(_CHG_LAG, n):
        if np.isfinite(composite[t]) and np.isfinite(composite[t - _CHG_LAG]):
            chg60[t] = composite[t] - composite[t - _CHG_LAG]

    # ------------------------------------------------------------------ #
    # 7. Assign back to df                                                #
    # ------------------------------------------------------------------ #
    df["ext4_tail_risk_composite_score"]    = composite
    df["ext4_tail_risk_composite_chg60"]    = chg60
    df["ext4_tail_risk_composite_downbeta"] = z_downbeta

    return df
