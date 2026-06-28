"""
_paper_2605_30442_downside_beta.py  --  Regime-dependent downside (semi-)beta suite.

Proxy for arXiv 2605.30442 "When market boundaries weaken: Network reconfiguration and
regime-dependent cross-asset spillovers."  The paper shows that coupling to the broader
system rises in high-turbulence states; downside semi-beta is the canonical per-ticker
analogue of that regime-dependent amplification.

External SPY data is loaded through the shared _indexes.py helper and merged onto the
per-ticker frame with a backward merge_asof (past-only, look-ahead safe) WITHOUT
reordering df — df is already ascending by Date.

All betas are computed inside a trailing 120-day rolling window. The market return
subset used for each metric is chosen on PAST data only (no peeking at the current
day's market direction for the current day's beta).
"""

import importlib.util as _ilu
import math
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load the underscore-prefixed shared index helper by path (auto-discovery
# skips it, so we import it explicitly).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)


METADATA = {
    "name": "paper_2605_30442_downside_beta",
    "description": (
        "Regime-dependent semi-beta suite: downside beta, upside beta, asymmetry gap, "
        "downside correlation, tail co-movement, and full-window reference beta "
        "(all 120-day trailing, SPY market proxy)."
    ),
    "requires": ["Date", "Close"],
    "produces": [
        "dsb_beta_120",
        "dsb_down_beta_120",
        "dsb_up_beta_120",
        "dsb_beta_asym_120",
        "dsb_down_corr_120",
        "dsb_tail_comove_120",
    ],
    "tags": ["market_regime", "risk", "beta", "experimental"],
    "version": "1.0",
    "author": "paper arXiv 2605.30442 — downside-beta proxy",
}

# Minimum number of return observations in a conditional subset before we
# return NaN instead of an unreliable estimate.
_MIN_OBS = 10

# Rolling window length in trading days.
_WINDOW = 120


def _safe_beta(stock_r: np.ndarray, mkt_r: np.ndarray) -> float:
    """
    OLS beta = cov(stock, mkt) / var(mkt) from two aligned 1-D arrays.
    Returns NaN when fewer than _MIN_OBS valid pairs exist or var(mkt) == 0.
    """
    mask = np.isfinite(stock_r) & np.isfinite(mkt_r)
    n = mask.sum()
    if n < _MIN_OBS:
        return math.nan
    s = stock_r[mask]
    m = mkt_r[mask]
    var_m = np.var(m, ddof=1)
    if var_m == 0.0 or not np.isfinite(var_m):
        return math.nan
    cov_sm = np.cov(s, m, ddof=1)[0, 1]
    if not np.isfinite(cov_sm):
        return math.nan
    return float(cov_sm / var_m)


def _safe_corr(stock_r: np.ndarray, mkt_r: np.ndarray) -> float:
    """
    Pearson correlation from two aligned 1-D arrays.
    Returns NaN when fewer than _MIN_OBS valid pairs or zero std.
    """
    mask = np.isfinite(stock_r) & np.isfinite(mkt_r)
    n = mask.sum()
    if n < _MIN_OBS:
        return math.nan
    s = stock_r[mask]
    m = mkt_r[mask]
    std_s = np.std(s, ddof=1)
    std_m = np.std(m, ddof=1)
    if std_s == 0.0 or std_m == 0.0:
        return math.nan
    if not (np.isfinite(std_s) and np.isfinite(std_m)):
        return math.nan
    cov_sm = np.cov(s, m, ddof=1)[0, 1]
    if not np.isfinite(cov_sm):
        return math.nan
    return float(cov_sm / (std_s * std_m))


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # Merge SPY close onto df without reordering.  merge_asof backward = last
    # known SPY on/before each row's Date.
    # -----------------------------------------------------------------------
    spy_close = _indexes.index_close("SPY")   # pd.Series indexed by DatetimeIndex

    nan_cols = [
        "dsb_beta_120",
        "dsb_down_beta_120",
        "dsb_up_beta_120",
        "dsb_beta_asym_120",
        "dsb_down_corr_120",
        "dsb_tail_comove_120",
    ]

    if spy_close.empty:
        # SPY unavailable — degrade every produced column to NaN.
        new = {c: pd.Series(np.nan, index=df.index) for c in nan_cols}
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # Build a two-column helper frame: Date + spy_close, ascending.
    spy_df = (
        spy_close
        .reset_index()
        .rename(columns={"Close": "spy_close"})
    )
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)

    # Merge onto df rows (df already ascending, no sort needed).
    tmp = pd.DataFrame({"Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(tmp, spy_df, on="Date", direction="backward")
    spy_aligned = merged["spy_close"].to_numpy(dtype=np.float64)

    # Daily simple returns (aligned to df rows).
    # spy_ret[i] = (spy_aligned[i] / spy_aligned[i-1]) - 1
    spy_ret = np.empty(len(df), dtype=np.float64)
    spy_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        spy_ret[1:] = np.where(
            spy_aligned[:-1] > 0,
            spy_aligned[1:] / spy_aligned[:-1] - 1.0,
            np.nan,
        )

    # Stock daily simple returns.
    close_arr = df["Close"].to_numpy(dtype=np.float64)
    stk_ret = np.empty(len(df), dtype=np.float64)
    stk_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        stk_ret[1:] = np.where(
            close_arr[:-1] > 0,
            close_arr[1:] / close_arr[:-1] - 1.0,
            np.nan,
        )

    n = len(df)
    out_beta    = np.full(n, np.nan)
    out_dbeta   = np.full(n, np.nan)
    out_ubeta   = np.full(n, np.nan)
    out_asym    = np.full(n, np.nan)
    out_dcorr   = np.full(n, np.nan)
    out_tail    = np.full(n, np.nan)

    # Precompute 10th-percentile threshold of SPY returns for each window.
    # tail_thresh[i] = 10th percentile of mkt_ret[i-WINDOW+1 .. i] (past only).
    # We use this to identify the worst-decile market days in the window.

    for i in range(_WINDOW - 1, n):
        s_win = stk_ret[i - _WINDOW + 1 : i + 1]
        m_win = spy_ret[i - _WINDOW + 1 : i + 1]

        # Full-window beta.
        out_beta[i] = _safe_beta(s_win, m_win)

        # Down/up day masks.
        down_mask = m_win < 0.0
        up_mask   = m_win >= 0.0

        s_down = s_win[down_mask]
        m_down = m_win[down_mask]
        s_up   = s_win[up_mask]
        m_up   = m_win[up_mask]

        db = _safe_beta(s_down, m_down)
        ub = _safe_beta(s_up,   m_up)
        out_dbeta[i] = db
        out_ubeta[i] = ub

        if np.isfinite(db) and np.isfinite(ub):
            out_asym[i] = db - ub

        out_dcorr[i] = _safe_corr(s_down, m_down)

        # Tail co-movement: fraction of worst-decile market days where stock
        # ALSO had a negative return.
        valid_mask = np.isfinite(m_win) & np.isfinite(s_win)
        m_valid = m_win[valid_mask]
        s_valid = s_win[valid_mask]
        if len(m_valid) >= _MIN_OBS:
            thresh = np.percentile(m_valid, 10.0)
            tail_mkt_mask = m_valid <= thresh
            n_tail = tail_mkt_mask.sum()
            if n_tail > 0:
                n_tail_neg_stk = (s_valid[tail_mkt_mask] < 0.0).sum()
                out_tail[i] = float(n_tail_neg_stk) / float(n_tail)

    new = {
        "dsb_beta_120":      pd.Series(out_beta,  index=df.index),
        "dsb_down_beta_120": pd.Series(out_dbeta, index=df.index),
        "dsb_up_beta_120":   pd.Series(out_ubeta, index=df.index),
        "dsb_beta_asym_120": pd.Series(out_asym,  index=df.index),
        "dsb_down_corr_120": pd.Series(out_dcorr, index=df.index),
        "dsb_tail_comove_120": pd.Series(out_tail, index=df.index),
    }

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
