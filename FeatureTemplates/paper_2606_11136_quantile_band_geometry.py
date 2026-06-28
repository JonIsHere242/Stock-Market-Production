"""
Quantile Band Geometry Features
Paper: "Conformal Prediction for Dyadic Regression Under Complex Missingness"
arXiv: 2606.11136

The paper is pure statistical theory for conformal prediction in dyadic (pairwise
relationship) regression settings under complex missingness, with no OHLCV-computable
method.  SKIPPED as-is.

INSTEAD: We implement a distinctive OHLCV feature family inspired by the paper's
core CONCEPT — measuring the GEOMETRY of prediction/quantile bands and how observations
sit relative to those bands.  The idea: treat rolling quantile envelopes as the
"prediction bands" and score each bar's OHLCV primitives by their position within,
at the edge of, or outside those bands.  This is a natural per-ticker version of
conformal distributional validity.

Feature family (7 columns prefixed "qbg_"):
  qbg_close_rank_20      Rank (0-1) of Close within 20d rolling quantile envelope
  qbg_close_rank_60      Same, 60d window
  qbg_vol_rank_20        Rank of Volume within 20d envelope
  qbg_hl_band_width_20   (High-Low) / 20d IQR of (High-Low) — normalised range
  qbg_band_breach_score  Fraction of last 5 bars where Close was outside [10th,90th] pctile
  qbg_close_tail_20      (Close - 80th pctile) / IQR  — positive when tail-high
  qbg_conformal_pvalue   Empirical p-value: fraction of 60d history with Close <= today's Close
"""

import numpy as np
import pandas as pd

METADATA = {
    "name": "paper_2606_11136_quantile_band_geometry",
    "description": (
        "Quantile band geometry features inspired by conformal prediction theory "
        "(arXiv:2606.11136 — conformal dyadic regression). Rolling quantile envelopes "
        "score OHLCV bars by position within empirical prediction bands: rank, tail "
        "excess, band-breach frequency, and conformal p-value."
    ),
    "requires": ["Close", "High", "Low", "Volume"],
    "produces": [
        "qbg_close_rank_20",
        "qbg_close_rank_60",
        "qbg_vol_rank_20",
        "qbg_hl_band_width_20",
        "qbg_band_breach_score",
        "qbg_close_tail_20",
        "qbg_conformal_pvalue",
    ],
    "tags": ["volatility", "mean_reversion", "experimental"],
    "version": "1.0",
    "author": "paper:2606.11136",
}


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """
    For each row t, compute the rank of s[t] within the past `window` values
    of s (including t itself). Returns a value in [0, 1].
    Causal: uses only current and past values.
    """
    def _rank_last(arr):
        if len(arr) == 0:
            return np.nan
        return float(np.sum(arr[:-1] <= arr[-1])) / max(len(arr) - 1, 1)

    return s.rolling(window, min_periods=max(5, window // 4)).apply(
        _rank_last, raw=True
    )


def _rolling_quantile_iqr(s: pd.Series, window: int, q_lo: float = 0.25,
                           q_hi: float = 0.75) -> pd.Series:
    lo = s.rolling(window, min_periods=max(5, window // 4)).quantile(q_lo)
    hi = s.rolling(window, min_periods=max(5, window // 4)).quantile(q_hi)
    return hi - lo


def compute(df: pd.DataFrame) -> pd.DataFrame:
    C = df["Close"].astype(float)
    V = df["Volume"].astype(float)
    H = df["High"].astype(float)
    L = df["Low"].astype(float)

    # ── Rolling rank of Close within past window ────────────────────────────────
    df["qbg_close_rank_20"] = _rolling_rank(C, 20)
    df["qbg_close_rank_60"] = _rolling_rank(C, 60)

    # ── Rolling rank of Volume ──────────────────────────────────────────────────
    df["qbg_vol_rank_20"] = _rolling_rank(V, 20)

    # ── HL-range normalized by rolling IQR of HL-range ─────────────────────────
    hl = (H - L).astype(float)
    hl_iqr_20 = _rolling_quantile_iqr(hl, 20)
    df["qbg_hl_band_width_20"] = hl / hl_iqr_20.replace(0, np.nan)

    # ── Band breach score: fraction of last 5 days outside [10th,90th] pctile ──
    lo10 = C.rolling(60, min_periods=20).quantile(0.10)
    hi90 = C.rolling(60, min_periods=20).quantile(0.90)
    outside = ((C < lo10) | (C > hi90)).astype(float)
    df["qbg_band_breach_score"] = outside.rolling(5, min_periods=2).mean()

    # ── Tail excess: (Close - 80th pctile) / IQR  — how far into upper tail ────
    p80 = C.rolling(20, min_periods=8).quantile(0.80)
    iqr20 = _rolling_quantile_iqr(C, 20)
    df["qbg_close_tail_20"] = (C - p80) / iqr20.replace(0, np.nan)

    # ── Conformal empirical p-value: fraction of 60d history <= today's Close ──
    # Exactly what conformal prediction uses: the rank of the new observation
    # in the calibration set.  Uses .apply over rolling window (causal, no leakage).
    def _empirical_pval(arr):
        if len(arr) < 2:
            return np.nan
        # arr[-1] is today; history is arr[:-1]
        return float(np.sum(arr[:-1] <= arr[-1])) / len(arr[:-1])

    df["qbg_conformal_pvalue"] = C.rolling(60, min_periods=20).apply(
        _empirical_pval, raw=True
    )

    return df
