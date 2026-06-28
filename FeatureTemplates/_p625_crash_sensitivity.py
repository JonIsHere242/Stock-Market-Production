"""
_p625_crash_sensitivity.py -- Crash-sensitivity MAGNITUDE vs the market.

For each rolling window W the market lower-tail days are those where the SPY return m
sits at/below its own trailing 10th percentile.  We then measure how the stock behaves
on exactly those days:

    csen_cond_mean_W : mean stock return r conditional on a market lower-tail day
                       = sum(r * tail_mask) / sum(tail_mask)   (>= 6 tail days)
    csen_tail_lift_W : that conditional mean minus the stock's own rolling mean return
                       (the EXCESS over its drift -- how much worse it does on crash days)
    csen_htcr_63     : hybrid tail-covariance, sum((r-meanR)*(m-meanM)*mask)/sum(mask)*1e4

All statistics are trailing/causal (rolling 10% quantile + trailing masked sums over the
SAME window), so the value at row t uses only rows <= t.

Refs: Chabi-Yo, Ruenzi & Weigert (2018) JFQA "Crash Sensitivity and the Cross-Section of
Expected Stock Returns"; Kelly & Jiang (2014) RFS "Tail Risk and Asset Prices";
Agarwal, Ruenzi & Weigert hybrid tail-covariance risk (HTCR).
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared market-index helper (auto-skipped by framework discovery; import by path).
# ---------------------------------------------------------------------------
_spec = _ilu.spec_from_file_location(
    "_indexes", _Path(__file__).resolve().parent / "_indexes.py"
)
_indexes = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_indexes)

METADATA = {
    "name":        "_p625_crash_sensitivity",
    "description": (
        "Crash-sensitivity magnitude: stock return conditional on market lower-tail days, "
        "its excess over own drift, and hybrid tail-covariance vs SPY "
        "(Chabi-Yo/Ruenzi/Weigert 2018 JFQA; Kelly-Jiang 2014 RFS; ARW HTCR)."
    ),
    "requires":    ["Date", "Close"],
    "produces":    [
        "csen_cond_mean_63",
        "csen_tail_lift_63",
        "csen_htcr_63",
        "csen_cond_mean_126",
        "csen_tail_lift_126",
    ],
    "tags":        ["market_regime", "tail_risk", "crash", "experimental"],
    "version":     "1.0",
    "author":      "paper:Chabi-Yo Ruenzi Weigert 2018 JFQA; Kelly-Jiang 2014 RFS; ARW HTCR",
}

_TAIL_Q   = 0.10   # market lower-tail threshold
_MIN_TAIL = 6      # minimum tail days inside a window for a valid conditional stat


def _nan_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise num/den with den==0 (or non-finite) mapped to NaN; never inf."""
    den = den.where(den != 0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Pre-create columns so the contract holds even if SPY is unavailable for this run.
    for col in METADATA["produces"]:
        df[col] = np.nan

    close = df["Close"].astype(float)
    r = close.pct_change()  # trailing one-day stock return; uses rows <= t

    # ---- Align SPY return onto this ticker's own dates (backward merge_asof) ----
    spy_close = _indexes.index_close("SPY")
    if spy_close is None or spy_close.empty:
        return df

    spy = pd.DataFrame({
        "Date": pd.to_datetime(spy_close.index),
        "_spy_close": spy_close.to_numpy(dtype="float64"),
    }).sort_values("Date")
    spy["_m"] = spy["_spy_close"].pct_change()  # SPY trailing return on SPY's calendar
    spy = spy[["Date", "_m"]]

    left = pd.DataFrame({"_pos": np.arange(len(df)),
                         "Date": pd.to_datetime(df["Date"].values)})
    merged = pd.merge_asof(
        left.sort_values("Date"), spy, on="Date", direction="backward"
    ).sort_values("_pos")
    m = pd.Series(merged["_m"].to_numpy(dtype="float64"), index=df.index)

    if m.notna().sum() < _MIN_TAIL:
        return df

    for W in (63, 126):
        # Trailing market lower-tail threshold (rolling 10% quantile, causal).
        q = m.rolling(W, min_periods=_MIN_TAIL).quantile(_TAIL_Q)
        mask = (m <= q) & m.notna() & q.notna()
        mask_f = mask.astype(float)

        # Trailing masked sums over the SAME window (NaN-safe products).
        r_masked = (r * mask_f).fillna(0.0)
        n_tail   = mask_f.rolling(W, min_periods=_MIN_TAIL).sum()
        sum_r    = r_masked.rolling(W, min_periods=_MIN_TAIL).sum()

        valid = n_tail >= _MIN_TAIL
        cond_mean = _nan_div(sum_r, n_tail).where(valid)
        df[f"csen_cond_mean_{W}"] = cond_mean.to_numpy(dtype="float64")

        # Excess over the stock's own trailing mean return.
        mean_r = r.rolling(W, min_periods=_MIN_TAIL).mean()
        df[f"csen_tail_lift_{W}"] = (cond_mean - mean_r).to_numpy(dtype="float64")

        if W == 63:
            # Hybrid tail-covariance: demeaned (trailing means) product on tail days.
            mean_m = m.rolling(W, min_periods=_MIN_TAIL).mean()
            cross = ((r - mean_r) * (m - mean_m) * mask_f).fillna(0.0)
            sum_cross = cross.rolling(W, min_periods=_MIN_TAIL).sum()
            htcr = _nan_div(sum_cross, n_tail).where(valid) * 1e4
            df["csen_htcr_63"] = htcr.to_numpy(dtype="float64")

    return df
