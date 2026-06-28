"""
Unexpected R&D increase (SurpriseRD) — per-ticker PIT proxy.

Based on:
  Eberhart, Maxwell and Siddique (2004) — "An Examination of Long-Term Abnormal
  Stock Returns and Operating Performance Following R&D Increases"
  via OpenSourceAP (Chen-Zimmermann).

Signal: Binary 1 if the firm shows an "unexpected" R&D increase:
  - R&D / Revenue > 0
  - R&D / Assets > 0
  - Annual growth in R&D > 5%
  - Annual growth in (R&D / Assets) > 5%
  Else 0.

Per-ticker proxy note: The original signal is computed once per annual filing.
Here we surface it as a step function tied to each filed_date so it updates
whenever a new 10-K/10-Q becomes public (PIT-safe via _fundamentals.as_of).
We also add a continuous intensity variant (RD-to-assets ratio) and an
annual growth-rate variant for richer signal.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_surpriserd",
    "description": (
        "Unexpected R&D increase indicator (SurpriseRD) per Eberhart, Maxwell & "
        "Siddique (2004). Binary flag (1) when: R&D/Revenue > 0, R&D/Assets > 0, "
        "annual R&D growth > 5%, and annual growth of R&D-to-Assets ratio > 5%. "
        "Per-ticker PIT proxy: updates at each new filed_date using _fundamentals "
        "helper (not a cross-sectional rank). Also emits continuous R&D intensity "
        "(R&D/Assets) and the annual R&D growth rate as companion columns."
    ),
    "requires": [],
    "produces": [
        "osap_surpriserd_flag",        # binary 1/0 SurpriseRD indicator
        "osap_surpriserd_rd_intensity", # R&D / Total Assets (continuous)
        "osap_surpriserd_rd_growth",    # annual growth rate of R&D expense
    ],
    "tags": ["fundamentals", "rd", "innovation", "accounting", "binary"],
    "version": "1.0",
    "author": "Eberhart, Maxwell & Siddique (2004); OpenSourceAP (Chen-Zimmermann); impl by Claude",
}


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: single-ticker, ascending Date, columns [Date, Ticker, Open, High, Low, Close, Volume].
    Adds three columns and returns df.
    """
    # Fetch PIT fundamentals — backward merge on filed_date (lookahead-safe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(
            df,
            fields=["rnd_expense", "revenue", "assets"],
        )

    rnd = df["fund_rnd_expense"]
    rev = df["fund_revenue"]
    assets = df["fund_assets"]

    # -----------------------------------------------------------------------
    # R&D intensity: R&D / Assets
    # -----------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        rd_intensity = np.where(
            (assets.notna()) & (assets != 0),
            rnd / assets,
            np.nan,
        )
    rd_intensity = pd.Series(rd_intensity, index=df.index)

    # -----------------------------------------------------------------------
    # R&D / Revenue ratio (used for positivity check)
    # -----------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        rd_over_rev = np.where(
            (rev.notna()) & (rev != 0),
            rnd / rev,
            np.nan,
        )
    rd_over_rev = pd.Series(rd_over_rev, index=df.index)

    # -----------------------------------------------------------------------
    # Annual R&D growth and annual R&D/Assets growth
    # Fundamentals update on filed_date cadence; to avoid double-counting we
    # use the *change in the PIT-merged value* across a ~252-trading-day
    # lookback (one fiscal year proxy).  Because values step-function on
    # filed_date, this naturally captures the most recent annual delta.
    # -----------------------------------------------------------------------
    ANNUAL_WINDOW = 252  # trading days ≈ 1 fiscal year

    # Prior-year values (shift by ANNUAL_WINDOW rows)
    rnd_prior = rnd.shift(ANNUAL_WINDOW)
    intensity_prior = rd_intensity.shift(ANNUAL_WINDOW)

    # Annual growth of R&D expense
    with np.errstate(divide="ignore", invalid="ignore"):
        rd_growth = np.where(
            (rnd_prior.notna()) & (rnd_prior > 0),
            (rnd - rnd_prior) / rnd_prior,
            np.nan,
        )
    rd_growth = pd.Series(rd_growth, index=df.index)

    # Annual growth of R&D / Assets
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity_growth = np.where(
            (intensity_prior.notna()) & (intensity_prior > 0),
            (rd_intensity - intensity_prior) / intensity_prior.abs(),
            np.nan,
        )
    intensity_growth = pd.Series(intensity_growth, index=df.index)

    # -----------------------------------------------------------------------
    # SurpriseRD binary flag
    # Conditions (all must hold):
    #   1. R&D / Revenue > 0  (R&D is positive and so is revenue)
    #   2. R&D / Assets > 0   (R&D and assets both positive)
    #   3. Annual R&D growth > 5%
    #   4. Annual growth of R&D/Assets > 5%
    # -----------------------------------------------------------------------
    cond1 = rd_over_rev > 0
    cond2 = rd_intensity > 0
    cond3 = rd_growth > 0.05
    cond4 = intensity_growth > 0.05

    # Any NaN in conditions → flag is NaN (not 0) to avoid false negatives
    has_data = (
        rd_over_rev.notna()
        & rd_intensity.notna()
        & rd_growth.notna()
        & intensity_growth.notna()
    )

    flag = pd.Series(np.nan, index=df.index, dtype=float)
    flag[has_data] = (cond1 & cond2 & cond3 & cond4)[has_data].astype(float)

    # -----------------------------------------------------------------------
    # Assign produced columns
    # -----------------------------------------------------------------------
    df["osap_surpriserd_flag"] = flag
    df["osap_surpriserd_rd_intensity"] = rd_intensity
    df["osap_surpriserd_rd_growth"] = rd_growth

    # Drop intermediate fund_* columns not in produces
    for col in ["fund_rnd_expense", "fund_revenue", "fund_assets"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
