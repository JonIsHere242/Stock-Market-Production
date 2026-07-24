"""
Exceedance correlation asymmetry (Longin-Solnik 2001).
Captures the well-known phenomenon that stock-market correlations spike
in down-markets more than in up-markets -- a per-ticker tail-risk signal.
"""
from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# helper: market index
# ---------------------------------------------------------------------------
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ---------------------------------------------------------------------------
METADATA = {
    "name": "ff06282240_factor_exceedance_corr_gap",
    "description": (
        "Exceedance correlation asymmetry (Longin & Solnik 2001). "
        "Over a trailing 250-day window, returns for the stock and SPY are "
        "jointly standardised by their rolling mean/std; Pearson correlation "
        "is then computed restricted to days where BOTH standardised series "
        "are below -0.5σ (lower exceedance) and above +0.5σ (upper "
        "exceedance) respectively. The asymmetry exc_corr_asym = lower - upper "
        "is positive when correlations spike in down-markets (common for "
        "equities, a latent tail-risk proxy). Recomputed every 5 bars from "
        "series start (causal striding); forward-filled between strides."
    ),
    "requires": ["Close"],
    "produces": [
        "ff06282240_factor_exceedance_corr_gap_lower",
        "ff06282240_factor_exceedance_corr_gap_upper",
        "ff06282240_factor_exceedance_corr_gap_asym",
    ],
    "tags": ["factor", "correlation", "tail-risk", "asymmetry", "market"],
    "version": "1.0.0",
    "author": "feature-factory ff06282240",
}

# ---------------------------------------------------------------------------
_WINDOW = 250
_STRIDE = 5
_THRESHOLD = 0.5   # in standardised units
_MIN_OBS = 15      # min observations per leg
_COL_LOWER = "ff06282240_factor_exceedance_corr_gap_lower"
_COL_UPPER = "ff06282240_factor_exceedance_corr_gap_upper"
_COL_ASYM  = "ff06282240_factor_exceedance_corr_gap_asym"


def _exc_corr(s: np.ndarray, m: np.ndarray, min_obs: int, threshold: float):
    """
    Given two arrays s (stock std-ret) and m (SPY std-ret) of length W,
    return (lower_corr, upper_corr).
    """
    lower_mask = (s <= -threshold) & (m <= -threshold)
    upper_mask = (s >= threshold)  & (m >= threshold)

    def _pearson(a, b):
        if len(a) < min_obs:
            return np.nan
        ma, mb = a.mean(), b.mean()
        da, db = a - ma, b - mb
        denom = np.sqrt((da * da).sum() * (db * db).sum())
        if denom == 0.0:
            return np.nan
        return float((da * db).sum() / denom)

    lc = _pearson(s[lower_mask], m[lower_mask])
    uc = _pearson(s[upper_mask], m[upper_mask])
    return lc, uc


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise outputs to NaN on every code path
    df[_COL_LOWER] = np.nan
    df[_COL_UPPER] = np.nan
    df[_COL_ASYM]  = np.nan

    if len(df) < _WINDOW + 1:
        return df

    # --- fetch SPY closes and align with df dates (backward-safe) ----------
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        return df

    if spy_close is None or spy_close.empty:
        return df

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]

    # Ensure Date column is datetime for merge
    df_dates = df[["Date"]].copy()
    df_dates["Date"] = pd.to_datetime(df_dates["Date"])
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])

    merged = pd.merge_asof(
        df_dates.sort_values("Date"),
        spy_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )
    # Re-align to original df index order
    merged.index = df_dates.sort_values("Date").index
    spy_aligned = merged["spy_close"].reindex(df.index)

    # Stock log returns
    stock_ret = np.log(df["Close"] / df["Close"].shift(1))
    # SPY log returns (already aligned to df rows)
    spy_ret = np.log(spy_aligned / spy_aligned.shift(1))

    stock_arr = stock_ret.values.astype(float)
    spy_arr   = spy_ret.values.astype(float)
    n = len(df)

    lower_out = np.full(n, np.nan)
    upper_out = np.full(n, np.nan)

    # Causal striding: stride from series start, NOT end
    for i in range(_WINDOW, n):
        if (i % _STRIDE) != 0:
            continue
        # window [i-WINDOW, i) -- does NOT include bar i (causal)
        s_win = stock_arr[i - _WINDOW: i]
        m_win = spy_arr[i - _WINDOW: i]

        # mask out NaN pairs
        valid = np.isfinite(s_win) & np.isfinite(m_win)
        s_v = s_win[valid]
        m_v = m_win[valid]

        if len(s_v) < _WINDOW // 2:
            continue  # not enough valid data

        # Jointly standardise within window
        s_mu, s_sd = s_v.mean(), s_v.std(ddof=1)
        m_mu, m_sd = m_v.mean(), m_v.std(ddof=1)

        if s_sd == 0.0 or m_sd == 0.0:
            continue

        s_std = (s_v - s_mu) / s_sd
        m_std = (m_v - m_mu) / m_sd

        lc, uc = _exc_corr(s_std, m_std, _MIN_OBS, _THRESHOLD)
        lower_out[i] = lc
        upper_out[i] = uc

    # Forward-fill between strides (pd.Series.ffill)
    lower_series = pd.Series(lower_out, index=df.index).ffill()
    upper_series = pd.Series(upper_out, index=df.index).ffill()

    df[_COL_LOWER] = lower_series.values
    df[_COL_UPPER] = upper_series.values

    # asym = lower - upper; positive means down-market corr spike
    with np.errstate(invalid="ignore"):
        asym = np.where(
            np.isfinite(lower_series.values) & np.isfinite(upper_series.values),
            lower_series.values - upper_series.values,
            np.nan,
        )
    df[_COL_ASYM] = asym

    return df
