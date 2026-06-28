"""
Regime-conditional coskewness: coskewness of stock with SPY computed separately
in high-VIX and low-VIX regimes (median split over trailing 250d window), plus
the regime spread (high_vix_coskew - low_vix_coskew).

Orthogonal axis vs. osap_coskewacx: that block uses a single unconditional
coskewness estimate; this block partitions by volatility regime inside each window,
capturing whether the stock's crash-amplification worsens when fear is already elevated.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import pandas as pd
import numpy as np

# --- load _indexes helper ---
_s = _ilu.spec_from_file_location("_indexes", _P(__file__).resolve().parent / "_indexes.py")
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

METADATA = {
    "name": "ext_conditional_coskew",
    "description": (
        "Regime-conditional coskewness with SPY. Over a 250-day rolling window the "
        "daily VIX is split at its in-window median into high-VIX and low-VIX regimes. "
        "Coskewness E[r_stock * r_spy^2] / (std_stock * std_spy^2) is computed "
        "separately for each regime. Produces: high-VIX coskewness, low-VIX "
        "coskewness, and their difference (regime spread). This is orthogonal to "
        "unconditional coskewness because it isolates whether crash-co-movement "
        "concentrates in already-stressed market states. Per-ticker proxy — no "
        "cross-sectional ranks. Uses SPY returns and VIX via _indexes helpers."
    ),
    "requires": ["Close"],
    "produces": [
        "ext_conditional_coskew_hv",   # high-VIX regime coskewness
        "ext_conditional_coskew_lv",   # low-VIX regime coskewness
        "ext_conditional_coskew_spread",  # hv - lv (regime sensitivity)
    ],
    "tags": ["coskewness", "regime", "vix", "spy", "risk", "higher_moment"],
    "version": "1.0.0",
    "author": "Extension/exploration of osap_coskewacx gate-validated winner; spec by project team",
}

_WINDOW = 250
_MIN_OBS = 20  # minimum observations per regime to compute coskew


def _coskewness(r_stock: np.ndarray, r_spy: np.ndarray) -> float:
    """
    Standardized coskewness: E[r_i * r_m^2] / (sigma_i * sigma_m^2).
    Returns NaN if insufficient variance or too few observations.
    """
    n = len(r_stock)
    if n < _MIN_OBS:
        return np.nan
    si = np.std(r_stock, ddof=1)
    sm = np.std(r_spy, ddof=1)
    if si < 1e-12 or sm < 1e-12:
        return np.nan
    num = np.mean(r_stock * r_spy ** 2)
    denom = si * (sm ** 2)
    return num / denom


def compute(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    hv_out = np.full(n, np.nan)
    lv_out = np.full(n, np.nan)
    spread_out = np.full(n, np.nan)

    # --- stock log returns ---
    close = df["Close"].values.astype(np.float64)
    stock_ret = np.empty(n)
    stock_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret[1:] = np.where(
            (close[:-1] > 0) & np.isfinite(close[:-1]) & np.isfinite(close[1:]),
            np.log(close[1:] / close[:-1]),
            np.nan,
        )

    # --- SPY daily closes merged on Date (backward, lookahead-safe) ---
    try:
        spy_series = _indexes.index_close("SPY")  # DatetimeIndex -> Close
        spy_df = spy_series.rename("spy_close").reset_index()
        spy_df.columns = ["Date", "spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    except Exception:
        # If SPY unavailable, emit all-NaN columns
        df["ext_conditional_coskew_hv"] = np.nan
        df["ext_conditional_coskew_lv"] = np.nan
        df["ext_conditional_coskew_spread"] = np.nan
        return df

    # --- VIX daily close ---
    try:
        vix_df = _indexes.vix_daily_close()
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    except Exception:
        vix_df = None

    # Build a working frame with stock dates (sorted for merge_asof, re-sorted back later)
    dates = pd.to_datetime(df["Date"].values)
    work = pd.DataFrame({"Date": dates, "stock_ret": stock_ret})

    # Merge SPY returns (backward-safe merge_asof)
    spy_df_sorted = spy_df.sort_values("Date").reset_index(drop=True)
    spy_close_arr = spy_df_sorted["spy_close"].values.astype(np.float64)

    spy_ret_arr = np.empty(len(spy_df_sorted))
    spy_ret_arr[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        spy_ret_arr[1:] = np.where(
            (spy_close_arr[:-1] > 0),
            np.log(spy_close_arr[1:] / spy_close_arr[:-1]),
            np.nan,
        )
    spy_df_sorted["spy_ret"] = spy_ret_arr
    spy_ret_df = spy_df_sorted[["Date", "spy_ret"]]

    work = pd.merge_asof(
        work.sort_values("Date"),
        spy_ret_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    # Merge VIX
    if vix_df is not None:
        work = pd.merge_asof(
            work,
            vix_df.sort_values("Date"),
            on="Date",
            direction="backward",
        )
    else:
        work["vix_close"] = np.nan

    # Re-align merged (sorted) frame back to original df row order via a
    # positional join on Date. We attach the original position index, merge,
    # then sort by it.
    orig_pos = pd.DataFrame({"Date": dates, "_orig_pos": np.arange(n)})
    work = pd.merge_asof(
        orig_pos.sort_values("Date"),
        work.sort_values("Date")[["Date", "stock_ret", "spy_ret", "vix_close"]],
        on="Date",
        direction="backward",
    ).sort_values("_orig_pos").reset_index(drop=True)

    s_ret = work["stock_ret"].values.astype(np.float64)
    spy_ret = work["spy_ret"].values.astype(np.float64)
    vix_vals = work["vix_close"].values.astype(np.float64)

    # --- rolling window coskewness by VIX regime ---
    for t in range(_WINDOW - 1, n):
        s_w = s_ret[t - _WINDOW + 1: t + 1]
        m_w = spy_ret[t - _WINDOW + 1: t + 1]
        v_w = vix_vals[t - _WINDOW + 1: t + 1]

        valid = np.isfinite(s_w) & np.isfinite(m_w) & np.isfinite(v_w)
        s_v = s_w[valid]
        m_v = m_w[valid]
        v_v = v_w[valid]

        if len(s_v) < _MIN_OBS * 2:
            continue

        vix_med = np.median(v_v)
        hv_mask = v_v >= vix_med
        lv_mask = ~hv_mask

        hv_cs = _coskewness(s_v[hv_mask], m_v[hv_mask])
        lv_cs = _coskewness(s_v[lv_mask], m_v[lv_mask])

        hv_out[t] = hv_cs
        lv_out[t] = lv_cs
        if np.isfinite(hv_cs) and np.isfinite(lv_cs):
            spread_out[t] = hv_cs - lv_cs

    df["ext_conditional_coskew_hv"] = hv_out
    df["ext_conditional_coskew_lv"] = lv_out
    df["ext_conditional_coskew_spread"] = spread_out
    return df
