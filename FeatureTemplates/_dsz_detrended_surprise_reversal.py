"""
_dsz_detrended_surprise_reversal.py  --  CANDIDATE block for the `dt` (de-trended reversal) arm.

Construction-matched to the dt LABEL (next-day return minus trailing 5d-mean of daily returns): emits
today's vol-standardized deviation from its OWN prior 5d drift, signed for fade, as a 1-day surprise,
a 5-day EWMA accumulation, and a 5-day cumulative-above-trend surprise. Aligns the displacement window
exactly with the target's 5d de-trend window.

Distinct from incumbents: ilr_restore uses a 40d log-price z-score and _p625_illiq_reversal a
vol-standardized 5d return displacement, but BOTH are illiquidity-percentile GATED and NEITHER
subtracts the own 5d mean-of-daily-returns; conditional_reversion_skew estimates conditional DRIFT
asymmetries, not the de-trended own-surprise residual. Vol-trap-safe (ratio to own 21d vol).

Refs: Blitz, Huij, Lansdorp & Verbeek (2013) JFM 16(3) short-term residual reversal; Da, Liu &
Schaumburg (2014) MgmtSci 60(3). CAVEAT: short-horizon own-history reversal is the family with the
project's documented 85-95% IS->OOS decay -- the multi-seed (>=4) gate is essential. Candidate.
"""

import numpy as np
import pandas as pd

METADATA = {
    "name":        "_dsz_detrended_surprise_reversal",
    "description": "Window-matched dt residual: today's return minus its own trailing 5d-mean of "
                   "daily returns, vol-standardized and signed for fade, as a 1d surprise, 5d EWMA "
                   "accumulation, and 5d cumulative-above-trend surprise. Blitz et al 2013.",
    "requires":    ["Close"],
    "produces":    ["dsz_resid_1d", "dsz_acc5", "dsz_cum5"],
    "tags":        ["dt", "mean_reversion", "candidate"],
    "version":     "0.1",
    "author":      "alt-target-feature-research 2026-06-26 (dt label-matched residual)",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    r = df["Close"].pct_change()
    sd21 = r.rolling(21, min_periods=10).std().replace(0, np.nan)
    trend5 = r.rolling(5, min_periods=5).mean()

    resid = r - trend5.shift(1)                       # deviation from PRIOR 5d drift (strictly trailing)
    ratio = -(resid / sd21)                           # positive surprise -> expect fade tomorrow
    df["dsz_resid_1d"] = ratio.clip(-6.0, 6.0)
    df["dsz_acc5"] = ratio.ewm(span=5, min_periods=3).mean().clip(-6.0, 6.0)

    cum5 = (df["Close"] / df["Close"].shift(5) - 1.0) - 5.0 * trend5.shift(5)
    df["dsz_cum5"] = (-(cum5 / (sd21 * np.sqrt(5.0)))).clip(-6.0, 6.0)
    return df
