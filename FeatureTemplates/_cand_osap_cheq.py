"""
osap_cheq — Growth in book equity (Lockwood & Prombutr 2010 / Chen-Zimmermann)

Per the OpenSourceAP (Chen-Zimmermann) replication spec:
  cheq = book_equity_t / book_equity_{t-1}

Predicted sign: -1 (high book-equity growth → lower future returns; investment anomaly).

Per-ticker proxy: we pull PIT book equity from _fundamentals.as_of() using the
`equity` field (common stockholders' equity / book equity, ~84% coverage).
We then compute:
  osap_cheq_ratio   — trailing 12-month book equity growth ratio (current / prior year);
                      NaN when either period is <= 0 (spec: include only if positive both years).
  osap_cheq_log     — log(cheq_ratio); linearises the distribution for tree models.
  osap_cheq_accel   — slope of the last 3 annual cheq_ratio observations (acceleration).

Cross-sectional note: the original signal is ranked cross-sectionally; here it is
computed per-ticker from PIT fundamentals as a time-series ratio. The RANK should be
applied downstream by the framework's --add_xs_features step.
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ── load PIT fundamentals helper ──────────────────────────────────────────────
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ─────────────────────────────────────────────────────────────────────────────
METADATA = {
    "name": "osap_cheq",
    "description": (
        "Growth in book equity (ceq_t / ceq_{t-1}), per Lockwood & Prombutr (2010) "
        "as catalogued in OpenSourceAP (Chen & Zimmermann). Computed per-ticker using "
        "PIT book equity from _fundamentals.as_of(). Ratio is NaN when either "
        "period's equity is <= 0 (per spec). Predicted cross-sectional sign: -1 "
        "(higher growth → lower future return; investment anomaly). Cross-sectional "
        "ranking is applied downstream; this block outputs the per-ticker ratio."
    ),
    "requires": [],  # no raw OHLCV needed beyond Date/Ticker for the merge
    "produces": [
        "osap_cheq_ratio",
        "osap_cheq_log",
        "osap_cheq_accel",
    ],
    "tags": ["fundamental", "investment", "book_equity", "accounting", "osap"],
    "version": "1.0.0",
    "author": "Lockwood & Prombutr (2010); OpenSourceAP / Chen-Zimmermann catalogue",
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute book-equity growth features.

    df: single-ticker DataFrame, ascending by Date,
        columns: Date, Ticker, Open, High, Low, Close, Volume.
    Returns df with three new columns appended.
    """
    # ── pull PIT equity ───────────────────────────────────────────────────────
    df = _fundamentals.as_of(df, fields=["equity"])
    # fund_equity is now a column; values reflect what was publicly filed as of each Date.

    eq = df["fund_equity"].copy()  # float series, NaN where no filing

    # ── rolling 1-year growth ratio ───────────────────────────────────────────
    # Fundamentals are updated ~quarterly with backward merge; the filed value
    # changes step-wise. We shift by 252 trading days (~1 fiscal year) to get
    # the "prior year" value in a lookahead-free way.
    YEAR_DAYS = 252

    eq_prior = eq.shift(YEAR_DAYS)

    # Only valid where BOTH current and prior equity are strictly positive (per spec)
    valid = (eq > 0) & (eq_prior > 0)

    ratio = np.where(valid, eq / eq_prior, np.nan)
    ratio = pd.Series(ratio, index=df.index, dtype="float32")

    log_ratio = np.where(valid & (ratio > 0), np.log(ratio), np.nan)
    log_ratio = pd.Series(log_ratio, index=df.index, dtype="float32")

    # ── acceleration: slope of the last ~3 annual snapshots ───────────────────
    # Sample ratio at t, t-252, t-504 and fit a linear slope (no lookahead).
    ratio_1y_ago = ratio.shift(YEAR_DAYS)
    ratio_2y_ago = ratio.shift(2 * YEAR_DAYS)

    # Slope of [t-504, t-252, t] over equal spacing (simplified OLS for 3 points):
    # slope = ((y2 - y0) / 2) using indices 0,1,2 → x = [-1, 0, 1], centred
    # centred: slope = (y[t] - y[t-504]) / 2
    accel_valid = valid & ratio_1y_ago.notna() & ratio_2y_ago.notna()
    accel = np.where(
        accel_valid,
        (ratio - ratio_2y_ago) / 2.0,
        np.nan,
    )
    accel = pd.Series(accel, index=df.index, dtype="float32")

    # ── assign produced columns ───────────────────────────────────────────────
    df["osap_cheq_ratio"] = ratio
    df["osap_cheq_log"] = log_ratio
    df["osap_cheq_accel"] = accel

    # ── drop scratch fundamental column ──────────────────────────────────────
    df.drop(columns=["fund_equity"], inplace=True)

    return df
