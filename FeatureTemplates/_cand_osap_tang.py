"""
osap_tang — Asset Tangibility feature block.

Source: Open Source Asset Pricing (OSAP) characteristic "tang" —
  Berger, P.G., Ofek, E., & Swary, I. (1996), "Investor valuation of the
  abandonment option," Journal of Financial Economics 42(2), 257-287.
  Almeida, H., & Campello, M. (2007), "Financial constraints, asset tangibility,
  and corporate investment," Review of Financial Studies 20(5), 1429-1460.
  As catalogued by Chen, A.Y. & Zimmermann, T. (2022), "Open Source
  Cross-Sectional Asset Pricing," Critical Finance Review 11(2), 207-264.

Definition:
  tang = (cash + 0.715*receivables + 0.547*inventory + 0.535*ppe_net) / assets

Economic intuition: tangible assets can be pledged as collateral and are easier
to liquidate in distress. Higher tangibility -> lower financial constraints ->
predicts higher future returns in some literature, lower in others (positive
predictor of investment, negative in distress-risk channel).

Per-ticker proxy: uses PIT SEC fundamentals via _fundamentals.as_of().
Coverage ~84% (ETFs / foreign ADRs = NaN). Three outputs:
  - osap_tang_level  : instantaneous tangibility ratio
  - osap_tang_chg    : quarter-over-quarter change in tangibility (proxy for
                       direction of balance-sheet evolution)
  - osap_tang_zscore : 252-day rolling z-score of the level (time-series
                       normalisation so the model can see deviation from
                       the stock's own historical norm)
"""

from __future__ import annotations
import importlib.util as _ilu
from pathlib import Path as _P
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load PIT fundamentals helper
# ---------------------------------------------------------------------------
_s2 = _ilu.spec_from_file_location(
    "_fundamentals",
    _P(__file__).resolve().parent / "_fundamentals.py",
)
_fundamentals = _ilu.module_from_spec(_s2)
_s2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_tang",
    "description": (
        "Asset tangibility: (cash + 0.715*receivables + 0.547*inventory + "
        "0.535*ppe_net) / assets, following Berger et al. (1996) and "
        "Almeida & Campello (2007) as used in Open Source Asset Pricing "
        "(Chen & Zimmermann 2022). Outputs: level, QoQ change, and rolling "
        "z-score. Per-ticker proxy using PIT SEC fundamentals (~84% coverage)."
    ),
    "requires": [],   # fundamentals fetched internally; no OHLCV columns needed
    "produces": [
        "osap_tang_level",
        "osap_tang_chg",
        "osap_tang_zscore",
    ],
    "tags": ["fundamentals", "balance_sheet", "tangibility", "osap", "value"],
    "version": "1.0.0",
    "author": (
        "Berger, Ofek & Swary (1996); Almeida & Campello (2007); "
        "Chen & Zimmermann OSAP (2022). Block by Claude."
    ),
}


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asset tangibility features on a single-ticker DataFrame."""

    # Pull PIT fundamentals; returns fund_<field> columns merged backward
    fields = ["cash", "receivables", "inventory", "ppe_net", "assets"]
    df = _fundamentals.as_of(df, fields=fields)

    # -----------------------------------------------------------------------
    # Tangibility level
    # -----------------------------------------------------------------------
    cash        = df["fund_cash"].astype(float)
    receivables = df["fund_receivables"].astype(float)
    inventory   = df["fund_inventory"].astype(float)
    ppe_net     = df["fund_ppe_net"].astype(float)
    assets      = df["fund_assets"].astype(float)

    numerator = (
        cash
        + 0.715 * receivables
        + 0.547 * inventory
        + 0.535 * ppe_net
    )

    # Guard division: assets == 0 or NaN -> NaN
    denom = assets.replace(0.0, np.nan)
    tang_level = numerator / denom

    # Clamp to [0, 1] — by construction should be in range, but protect
    # against data quirks (negative receivables etc.)
    tang_level = tang_level.clip(lower=0.0, upper=1.0)

    df["osap_tang_level"] = tang_level

    # -----------------------------------------------------------------------
    # Quarter-over-quarter change in tangibility
    # The fundamentals helper fills forward (LOCF after filed_date), so
    # we detect genuine updates by watching for changes in the level.
    # A 63-day (≈1 quarter) shift captures the previous-quarter value.
    # -----------------------------------------------------------------------
    df["osap_tang_chg"] = tang_level - tang_level.shift(63)

    # -----------------------------------------------------------------------
    # Rolling 252-day z-score of the level (time-series normalisation)
    # -----------------------------------------------------------------------
    roll_mean = tang_level.rolling(252, min_periods=63).mean()
    roll_std  = tang_level.rolling(252, min_periods=63).std(ddof=1)
    # Guard std == 0 (perfectly flat series)
    roll_std = roll_std.replace(0.0, np.nan)
    df["osap_tang_zscore"] = (tang_level - roll_mean) / roll_std

    # -----------------------------------------------------------------------
    # Drop scratch fund_ columns NOT in produces
    # -----------------------------------------------------------------------
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    df = df.drop(columns=fund_cols)

    return df
