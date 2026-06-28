"""
R&D Ability — per-ticker proxy.

Cohen, Diether and Malloy (2013) regress annual log-sales-growth on 5 lags of
log(1 + xrd/sale) over rolling 8-year windows.  Their cross-sectional gate
(top tercile of xrd/sale within year) cannot be applied inside a single-ticker
block; we replace it with an absolute threshold (xrd_intensity > 0.01, i.e.
the firm spends at least 1% of revenue on R&D) and require ≥50% of the window
to have non-zero R&D — the same economic screen, applied per-ticker.

Outputs:
  osap_rdability_score   – mean OLS coefficient across the 5 lags (the core signal)
  osap_rdability_rdi     – current annual R&D intensity log(1 + xrd/sale)
  osap_rdability_slope   – 3-year rolling trend of xrd_intensity (momentum variant)
"""

from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fundamentals helper (PIT-safe backward merge)
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_rdability",
    "description": (
        "Per-ticker R&D Ability proxy (Cohen, Diether & Malloy 2013 / OpenSourceAP "
        "Chen-Zimmermann). Fits 5 bivariate OLS regressions of log-annual-sales-growth "
        "on lags 1-5 of log(1+rnd_intensity) over a rolling 8-year window (requires "
        "≥6 valid obs, ≥50% non-zero R&D rows). The cross-sectional top-tercile gate "
        "is replaced by an absolute threshold: xrd/sale > 0.01 at the evaluation bar "
        "(i.e. firm is an active R&D spender). Score = mean of the 5 lag coefficients "
        "(positive = historically better sales conversion from R&D). "
        "osap_rdability_rdi = current log(1+xrd/sale). "
        "osap_rdability_slope = 3yr rolling OLS slope of rdi (R&D intensity trend)."
    ),
    "requires": ["Close"],  # only Close needed to anchor dates; fundamentals via helper
    "produces": [
        "osap_rdability_score",
        "osap_rdability_rdi",
        "osap_rdability_slope",
    ],
    "tags": ["fundamentals", "rd", "accounting", "osap", "innovation"],
    "version": "1.0",
    "author": "Cohen, Diether & Malloy (2013) via OpenSourceAP (Chen-Zimmermann); per-ticker proxy impl.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Simple bivariate OLS slope β in y = α + β·x, no intercept optimised version."""
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = (xm * xm).sum()
    if denom == 0.0:
        return np.nan
    return float((xm * ym).sum() / denom)


def _rd_ability_from_annual(rdi_arr: np.ndarray, sg_arr: np.ndarray) -> float:
    """
    Given arrays of annual log(1+xrd/sale) and log(sale_growth), compute
    RD Ability = mean coefficient across 5 bivariate lag regressions.

    Window: last 8 observations (passed in already).
    Requires: >=6 non-NaN pairs per lag regression (with at least 6 annual obs
    total in the window), >=50% of rdi values non-zero.
    """
    n = len(rdi_arr)
    if n < 6:
        return np.nan

    # Require >=50% non-zero R&D
    nonzero = np.sum(np.isfinite(rdi_arr) & (rdi_arr > 0.0))
    if nonzero < (n / 2.0):
        return np.nan

    coefs = []
    for lag in range(1, 6):
        if lag >= n:
            continue
        # y = sales_growth[lag:], x = rdi[:-lag]
        y = sg_arr[lag:]
        x = rdi_arr[: n - lag]
        # need >=6 joint valid
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 6:
            continue
        beta = _ols_slope(y[mask], x[mask])
        if np.isfinite(beta):
            coefs.append(beta)

    if len(coefs) == 0:
        return np.nan
    return float(np.mean(coefs))


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Initialise output columns with NaN
    df["osap_rdability_score"] = np.nan
    df["osap_rdability_rdi"] = np.nan
    df["osap_rdability_slope"] = np.nan

    if df.shape[0] < 2:
        return df

    # Pull PIT fundamentals (revenue_ttm, rnd_expense_ttm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["revenue_ttm", "rnd_expense_ttm"])

    rev_col = "fund_revenue_ttm"
    rnd_col = "fund_rnd_expense_ttm"

    # Guard: if columns absent, return
    if rev_col not in df.columns or rnd_col not in df.columns:
        for c in ["fund_revenue_ttm", "fund_rnd_expense_ttm"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        return df

    rev = pd.to_numeric(df[rev_col], errors="coerce")
    rnd = pd.to_numeric(df[rnd_col], errors="coerce")

    # R&D intensity: log(1 + max(rnd,0) / rev) — guard zero/neg revenue
    rnd_safe = rnd.clip(lower=0.0)
    rev_safe = rev.where(rev > 0, other=np.nan)
    rdi_daily = np.log1p(rnd_safe / rev_safe)  # NaN where rev<=0 or rev NaN

    # Annual sales growth: log(rev_t / rev_{t-365d})
    # We work in daily space but fundamental values update at filing dates.
    # We sample approximately annually (every ~252 trading days) to avoid
    # look-ahead: use only strictly past values.
    df_sorted = df.reset_index(drop=True)
    dates = pd.to_datetime(df_sorted["Date"])

    # --- Annual sampling: for each row, record the fundamentals at that date ---
    # We build annual time series by resampling at ~1-year intervals from the END.
    rdi_series = rdi_daily.values.copy()  # aligned with df_sorted rows
    rev_series = rev.values.copy()

    n_rows = len(df_sorted)

    # Build annual snapshots: step back ~252 bars at a time
    # We produce a score at each row using data strictly up to and including that row.
    ANNUAL_STEP = 252
    MIN_WINDOW_YEARS = 6   # require at least 6 annual obs
    MAX_WINDOW_YEARS = 8

    scores = np.full(n_rows, np.nan)
    rdi_out = np.full(n_rows, np.nan)
    rdi_slope_out = np.full(n_rows, np.nan)

    for i in range(n_rows):
        rdi_val = rdi_series[i]
        rdi_out[i] = rdi_val if np.isfinite(rdi_val) else np.nan

        # Sample annual snapshots strictly up to row i
        # Collect one sample per ~252 rows looking backwards
        ann_rdi = []
        ann_rev = []
        step = ANNUAL_STEP
        idx = i
        while idx >= 0 and len(ann_rdi) < MAX_WINDOW_YEARS:
            r = rdi_series[idx]
            rv = rev_series[idx]
            ann_rdi.append(r if np.isfinite(r) else np.nan)
            ann_rev.append(rv if np.isfinite(rv) else np.nan)
            idx -= step

        # Reverse so oldest first
        ann_rdi_arr = np.array(ann_rdi[::-1], dtype=float)
        ann_rev_arr = np.array(ann_rev[::-1], dtype=float)

        n_ann = len(ann_rdi_arr)
        if n_ann < MIN_WINDOW_YEARS:
            continue

        # Sales growth: log(rev[t] / rev[t-1])  for t=1..n_ann-1
        rev_prev = ann_rev_arr[:-1]
        rev_curr = ann_rev_arr[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            sg = np.where(
                (rev_prev > 0) & (rev_curr > 0),
                np.log(rev_curr / rev_prev),
                np.nan,
            )

        # rdi for growth window is ann_rdi_arr[1:] aligned with sg
        rdi_w = ann_rdi_arr[1:]  # same length as sg

        score = _rd_ability_from_annual(rdi_w, sg)

        # Absolute R&D-spender gate (replace top-tercile XS gate):
        # current rdi must be > 0.01 (≈ 1% of revenue)
        if np.isfinite(rdi_val) and rdi_val > np.log1p(0.01):
            scores[i] = score
        # else leave NaN (firm not an active R&D spender this period)

        # R&D intensity slope (3 annual obs trend)
        if n_ann >= 3:
            rdi_last3 = ann_rdi_arr[-3:]
            if np.sum(np.isfinite(rdi_last3)) >= 2:
                x3 = np.arange(3, dtype=float)
                rdi_slope_out[i] = _ols_slope(rdi_last3, x3)

    df["osap_rdability_score"] = scores
    df["osap_rdability_rdi"] = rdi_out
    df["osap_rdability_slope"] = rdi_slope_out

    # Drop scratch fund_ columns
    drop_cols = [c for c in df.columns if c.startswith("fund_")]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    return df
