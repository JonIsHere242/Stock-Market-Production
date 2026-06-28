"""
ext2_tail_dependence
--------------------
Lower-tail crash co-exceedance with the market (per-ticker proxy).

Over a rolling 250-day window, computes the fraction of the stock's
worst-decile-return days that coincide with SPY worst-decile return days
(lower-tail co-occurrence / crash co-dependence), the upper-tail analog,
and the lower-minus-upper asymmetry.

This is a nonparametric tail-co-movement loading -- conceptually related to
tail-dependence coefficients (lambda_L / lambda_U) from extreme-value theory
but implemented fully in-sample on the rolling window (no parametric copula).

Per-ticker proxy: cross-sectional ranking is replaced by rolling within-series
quantile thresholds, which is the valid per-ticker causal equivalent.

SOURCE: Round-3 deep exploration of xdom2_downside_beta winner vein.
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load _indexes helper (SPY daily close)
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
    "name": "ext2_tail_dependence",
    "description": (
        "Rolling 250-day nonparametric tail co-dependence with SPY. "
        "ext2_tail_dependence_lower: fraction of stock's worst-decile-return days "
        "that coincide with SPY worst-decile days (lower tail). "
        "ext2_tail_dependence_upper: same for best-decile days (upper tail). "
        "ext2_tail_dependence_asym: lower minus upper asymmetry (positive = "
        "crash-prone co-movement dominates). Per-ticker proxy for the "
        "classical tail-dependence coefficient lambda_L/lambda_U."
    ),
    "requires": ["Close"],
    "produces": [
        "ext2_tail_dependence_lower",
        "ext2_tail_dependence_upper",
        "ext2_tail_dependence_asym",
    ],
    "tags": ["tail", "downside", "market", "comovement", "crash", "spy", "extreme"],
    "version": "1.0.0",
    "author": "Round-3 deep exploration of xdom2_downside_beta winner vein",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW = 250
_TAIL_Q = 0.10  # worst/best decile


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling lower-tail, upper-tail, and asymmetry co-exceedance
    with SPY.  One stock at a time, ascending by Date.
    """
    n = len(df)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    if n < _WINDOW:
        df["ext2_tail_dependence_lower"] = lower
        df["ext2_tail_dependence_upper"] = upper
        df["ext2_tail_dependence_asym"] = np.full(n, np.nan)
        return df

    # -----------------------------------------------------------------------
    # Fetch SPY daily returns; merge backward onto df dates
    # -----------------------------------------------------------------------
    try:
        spy_close = _indexes.index_close("SPY")
        spy_df = spy_close.reset_index()
        spy_df.columns = ["Date", "_spy_close"]
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        spy_df = spy_df.sort_values("Date").reset_index(drop=True)
        spy_df["_spy_ret"] = spy_df["_spy_close"].pct_change()

        work = df[["Date"]].copy()
        work["Date"] = pd.to_datetime(work["Date"])
        work = pd.merge_asof(
            work.sort_values("Date"),
            spy_df[["Date", "_spy_ret"]],
            on="Date",
            direction="backward",
        )
        # restore original order
        work = work.set_index(df.index)
        spy_ret = work["_spy_ret"].values
    except Exception:
        # SPY unavailable — degrade gracefully
        spy_ret = np.full(n, np.nan)

    # -----------------------------------------------------------------------
    # Stock log returns (pct_change is fine for daily)
    # -----------------------------------------------------------------------
    close = df["Close"].values.astype(float)
    stk_ret = np.empty(n)
    stk_ret[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev = close[:-1]
        curr = close[1:]
        stk_ret[1:] = np.where(prev == 0.0, np.nan, (curr - prev) / prev)

    # -----------------------------------------------------------------------
    # Rolling tail co-exceedance (vectorised with stride tricks)
    # -----------------------------------------------------------------------
    for t in range(_WINDOW - 1, n):
        s_win = stk_ret[t - _WINDOW + 1 : t + 1]   # shape (250,)
        m_win = spy_ret[t - _WINDOW + 1 : t + 1]

        # mask out any NaN rows jointly
        valid = np.isfinite(s_win) & np.isfinite(m_win)
        if valid.sum() < _WINDOW // 2:
            continue

        sv = s_win[valid]
        mv = m_win[valid]

        # Tail thresholds (within-window quantiles)
        s_lo_thr = np.quantile(sv, _TAIL_Q)
        s_hi_thr = np.quantile(sv, 1.0 - _TAIL_Q)
        m_lo_thr = np.quantile(mv, _TAIL_Q)
        m_hi_thr = np.quantile(mv, 1.0 - _TAIL_Q)

        s_lo = sv <= s_lo_thr
        m_lo = mv <= m_lo_thr
        s_hi = sv >= s_hi_thr
        m_hi = mv >= m_hi_thr

        n_s_lo = s_lo.sum()
        n_s_hi = s_hi.sum()

        # fraction of stock's tail days that coincide with market tail days
        lower[t] = float(np.logical_and(s_lo, m_lo).sum()) / n_s_lo if n_s_lo > 0 else np.nan
        upper[t] = float(np.logical_and(s_hi, m_hi).sum()) / n_s_hi if n_s_hi > 0 else np.nan

    df["ext2_tail_dependence_lower"] = lower
    df["ext2_tail_dependence_upper"] = upper
    df["ext2_tail_dependence_asym"] = lower - upper  # NaN propagates correctly

    return df
