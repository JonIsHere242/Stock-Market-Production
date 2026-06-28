"""
Candidate feature block: ext_cokurtosis
Systematic cokurtosis with the market (SPY).

Rolling 120d standardized cokurtosis = E[e_i * e_m^3] / (sd_i * sd_m^3)
where e_i = stock return deviation from mean, e_m = SPY return deviation from mean.
Captures tail-event sensitivity beyond beta and coskewness.

Produces: level + 60d change (delta).
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ── Load _indexes helper ────────────────────────────────────────────────────
_s = _ilu.spec_from_file_location(
    "_indexes", _P(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_s)
_s.loader.exec_module(_indexes)

# ── Metadata ────────────────────────────────────────────────────────────────
METADATA = {
    "name": "ext_cokurtosis",
    "description": (
        "Rolling 120-day standardized cokurtosis of the stock with the market (SPY). "
        "Computed as E[e_i * e_m^3] / (sd_i * sd_m^3) where e_i and e_m are "
        "demeaned daily log-returns for the stock and SPY respectively. "
        "Captures the stock's sensitivity to market tail events (extreme SPY moves), "
        "orthogonal to beta (cov / var) and coskewness (E[e_i * e_m^2]). "
        "Also produces the 60-day rolling change in cokurtosis as a momentum signal. "
        "Per-ticker proxy -- cross-sectional ranking is done downstream. "
        "SPY index loaded via _indexes helper; degrades to NaN if unavailable."
    ),
    "requires": ["Close"],
    "produces": ["ext_cokurtosis_120d", "ext_cokurtosis_delta60d"],
    "tags": ["cokurtosis", "market", "tail-risk", "moment", "systematic"],
    "version": "1.0.0",
    "author": (
        "Spec: Extension/exploration of gate-validated winner xdom2_downside_beta. "
        "Implementation: Claude Sonnet 4.6 for Stock-Market pipeline."
    ),
}

# ── Constants ────────────────────────────────────────────────────────────────
_WINDOW = 120       # cokurtosis estimation window (days)
_DELTA_LAG = 60     # lag for computing the change in cokurtosis
_MIN_PERIODS = 60   # minimum observations for a valid estimate


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 120-day cokurtosis vs SPY for each stock row."""
    # ── Load SPY close series ────────────────────────────────────────────────
    try:
        spy_close = _indexes.index_close("SPY")
    except Exception:
        spy_close = None

    if spy_close is None or spy_close.empty:
        df["ext_cokurtosis_120d"] = np.nan
        df["ext_cokurtosis_delta60d"] = np.nan
        return df

    # ── Align SPY to our dates via merge_asof (backward = no lookahead) ─────
    df_sorted = df.sort_values("Date").copy()
    dates = pd.to_datetime(df_sorted["Date"])

    spy_df = spy_close.rename("spy_close").reset_index()
    spy_df.columns = ["Date", "spy_close"]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted[["Date"]].assign(Date=dates),
        spy_df,
        on="Date",
        direction="backward",
    )

    # ── Daily log returns ────────────────────────────────────────────────────
    stock_ret = np.log(df_sorted["Close"].values / np.where(
        df_sorted["Close"].shift(1).values > 0,
        df_sorted["Close"].shift(1).values,
        np.nan,
    ))
    spy_ret = np.log(merged["spy_close"].values / np.where(
        merged["spy_close"].shift(1).values > 0,
        merged["spy_close"].shift(1).values,
        np.nan,
    ))

    n = len(df_sorted)
    cokurt_vals = np.full(n, np.nan)

    # ── Rolling cokurtosis via sliding window ────────────────────────────────
    # cokurtosis = E[e_i * e_m^3] / (sd_i * sd_m^3)
    # We compute with a Python loop over windows (n ~ 700, window 120 => ~580 iters,
    # each vectorised over 120 elements -> total ~69k ops, fast enough).
    for t in range(_MIN_PERIODS - 1, n):
        lo = max(0, t - _WINDOW + 1)
        ri = t + 1        # exclusive

        si = stock_ret[lo:ri]
        sm = spy_ret[lo:ri]

        # Drop any NaN positions jointly
        mask = np.isfinite(si) & np.isfinite(sm)
        if mask.sum() < _MIN_PERIODS:
            continue

        si_valid = si[mask]
        sm_valid = sm[mask]

        e_i = si_valid - si_valid.mean()
        e_m = sm_valid - sm_valid.mean()

        sd_i = e_i.std(ddof=1)
        sd_m = e_m.std(ddof=1)

        denom = sd_i * (sd_m ** 3)
        if not np.isfinite(denom) or denom == 0.0:
            continue

        cokurt_vals[t] = np.mean(e_i * (e_m ** 3)) / denom

    # ── 60-day change in cokurtosis ──────────────────────────────────────────
    cokurt_series = pd.Series(cokurt_vals, index=df_sorted.index)
    delta_vals = cokurt_series - cokurt_series.shift(_DELTA_LAG)

    # ── Write back preserving original row order ─────────────────────────────
    df["ext_cokurtosis_120d"] = cokurt_series
    df["ext_cokurtosis_delta60d"] = delta_vals

    return df
