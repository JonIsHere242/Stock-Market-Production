"""
_p625_lottery_orthogonal_max.py -- Vol-orthogonalized lottery demand.

MAX5 lottery demand scaled by idiosyncratic vol (intensity), single-day variance-share
concentration, and the idiosyncratic MAX (firm-specific upside spike net of the market's
best day x rolling beta). All trailing / causal.

Refs: Bali, Cakici & Whitelaw (2011) JFE "Maxing out"; Bali, Brown, Murray & Tang (2017)
JFQA "A lottery-demand-based explanation of the beta anomaly"; Nguyen (2026) JBFA.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings as _warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared index helper (loaded by file path; auto-skipped by discovery)
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "_p625_lottery_orthogonal_max",
    "description": (
        "Vol-orthogonalized lottery demand: MAX5/idiovol intensity, single-day "
        "variance-share concentration, and idiosyncratic MAX net of market best-day x beta "
        "(Bali Cakici Whitelaw 2011 JFE; Bali Brown Murray Tang 2017 JFQA; Nguyen 2026 JBFA)."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "lmx_intensity_21",
        "lmx_maxshare_21",
        "lmx_idio_max_21",
        "lmx_min_intensity_21",
        "lmx_intensity_10",
    ],
    "tags":        ["lottery", "idiosyncratic", "behavioral", "experimental"],
    "version":     "1.0",
    "author":      "paper:Bali,Cakici&Whitelaw(2011)JFE; Bali,Brown,Murray&Tang(2017)JFQA; Nguyen(2026)JBFA",
}

_BETA_MIN = 20   # min trailing obs for a rolling beta estimate


def _roll_sum_largest_k(r: np.ndarray, w: int, k: int, largest: bool) -> np.ndarray:
    """
    Trailing mean of the k largest (or smallest) values in each length-w window.
    Returns array aligned to r (NaN for warmup rows < w). Causal: window ends at t.
    """
    n = r.size
    out = np.full(n, np.nan, dtype="float64")
    if n < w:
        return out
    # windows[i] corresponds to r[i : i+w], i.e. window ENDING at index i+w-1
    windows = np.lib.stride_tricks.sliding_window_view(r, w)  # (n-w+1, w)
    if largest:
        # k largest -> partition so last k are the largest
        part = np.partition(windows, w - k, axis=1)[:, w - k:]
    else:
        # k smallest -> partition so first k are the smallest
        part = np.partition(windows, k - 1, axis=1)[:, :k]
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        vals = np.nanmean(part, axis=1)
    out[w - 1:] = vals
    return out


def compute(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)

    produced = list(METADATA["produces"])

    # ---- stock simple returns, Date-indexed (temporary) --------------------
    dates = pd.to_datetime(df["Date"])
    stock_close = pd.Series(close.values, index=dates.values)
    stock_ret = stock_close.pct_change()

    # ---- market (SPY) simple returns, aligned by Date ----------------------
    df_dates = dates.values
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = pd.Series(dtype="float64")

    if spy_close is None or spy_close.empty:
        # No market data -> emit NaN for everything, contract-clean.
        for c in produced:
            df[c] = np.nan
        return df

    spy_ret = spy_close.pct_change()

    # Inner-join on shared trading dates (both are simple returns)
    r_al, m_al = stock_ret.align(spy_ret, join="inner")
    r_al = r_al.astype(float)
    m_al = m_al.astype(float)

    if len(r_al) < _BETA_MIN:
        for c in produced:
            df[c] = np.nan
        return df

    # ---- rolling beta per window (cov/var), causal -------------------------
    def _beta(w: int) -> pd.Series:
        cov = r_al.rolling(w, min_periods=min(_BETA_MIN, w)).cov(m_al)
        var = m_al.rolling(w, min_periods=min(_BETA_MIN, w)).var()
        var = var.where(var > 0, np.nan)        # guard 0/empty denominator
        b = cov / var
        return b

    beta21 = _beta(21)

    # ---- idiosyncratic residual e = r - beta*m (beta_t known at t) ---------
    # idiovol_W = trailing std of e over W
    def _idiovol(beta_w: pd.Series, w: int) -> pd.Series:
        e = r_al - beta_w * m_al
        iv = e.rolling(w, min_periods=min(_BETA_MIN, w)).std()
        return iv

    iv21 = _idiovol(beta21, 21)
    beta10 = _beta(10)
    iv10 = _idiovol(beta10, 10)

    # ---- ndarray views of aligned series for stride tricks -----------------
    r_arr = r_al.to_numpy(dtype="float64")
    m_arr = m_al.to_numpy(dtype="float64")

    # MAX5 / MIN5 (mean of 5 largest / smallest) and MAX1 (largest) over W=21
    max5_21 = _roll_sum_largest_k(r_arr, 21, 5, largest=True)
    min5_21 = _roll_sum_largest_k(r_arr, 21, 5, largest=False)
    max1_21 = _roll_sum_largest_k(r_arr, 21, 1, largest=True)        # single largest r
    mmax1_21 = _roll_sum_largest_k(m_arr, 21, 1, largest=True)       # market single largest

    # rv = sum(r^2) over W=21 (realized variance proxy)
    r2 = r_arr ** 2
    rv21 = np.full(r_arr.size, np.nan, dtype="float64")
    if r_arr.size >= 21:
        rv21[20:] = np.lib.stride_tricks.sliding_window_view(r2, 21).sum(axis=1)

    # MAX5 over W=10 (for lmx_intensity_10)
    max5_10 = _roll_sum_largest_k(r_arr, 10, 5, largest=True)

    iv21_arr = iv21.to_numpy(dtype="float64")
    iv10_arr = iv10.to_numpy(dtype="float64")
    beta21_arr = beta21.to_numpy(dtype="float64")

    # ---- assemble features on the ALIGNED index, all guarded ---------------
    eps = 1e-4
    intensity_21 = np.clip(max5_21 / (iv21_arr + eps), -50.0, 50.0)
    min_intensity_21 = min5_21 / (iv21_arr + eps)          # signed (MIN5 typically negative)
    intensity_10 = np.clip(max5_10 / (iv10_arr + eps), -50.0, 50.0)

    maxshare_21 = np.clip((max1_21 ** 2) / (rv21 + 1e-12), 0.0, 1.0)

    idio_max_21 = np.clip(max1_21 - beta21_arr * mmax1_21, -1.0, 1.0)

    feat = pd.DataFrame(
        {
            "lmx_intensity_21":     intensity_21,
            "lmx_maxshare_21":      maxshare_21,
            "lmx_idio_max_21":      idio_max_21,
            "lmx_min_intensity_21": min_intensity_21,
            "lmx_intensity_10":     intensity_10,
        },
        index=r_al.index,
    )

    # ---- reindex back onto the original df row order (no df reorder) --------
    feat = feat.reindex(df_dates)
    for c in produced:
        df[c] = feat[c].to_numpy(dtype="float64")

    # final inf guard (defensive; clips/guards above should prevent inf)
    for c in produced:
        col = df[c].to_numpy(dtype="float64")
        col[~np.isfinite(col)] = np.nan
        df[c] = col

    return df
