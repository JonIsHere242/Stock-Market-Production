"""
Per-ticker Revenue Surprise feature block.

SPEC ID : osap_revenuesurprise
SOURCE  : OpenSourceAP (Chen-Zimmermann), Jegadeesh and Livnat 2006

PROXY NOTE: The paper signal is inherently cross-sectional (scaled by each
stock's own trailing std, then ranked across universe). Here we implement the
per-ticker standardised version: revenue surprise = 4-quarter YoY change in
revenue-per-share, standardised (z-scored) by its own trailing 8-quarter mean
and std. This captures the same 'beat vs own history' economic signal on a
per-ticker basis. Cross-sectional ranking is left to the downstream feature
framework's --add_xs_features pass.

Price-below-$5 rows are masked to NaN per the original paper exclusion.
"""
from __future__ import annotations

import importlib.util as _ilu
import warnings
from pathlib import Path as _P

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PIT fundamentals helper
# ---------------------------------------------------------------------------
_spec2 = _ilu.spec_from_file_location(
    "_fundamentals", _P(__file__).resolve().parent / "_fundamentals.py"
)
_fundamentals = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_fundamentals)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
METADATA = {
    "name": "osap_revenuesurprise",
    "description": (
        "Per-ticker Revenue Surprise: 4-quarter YoY change in revenue-per-share "
        "standardised by the trailing 8-observation (≈2-year) mean and std of "
        "those changes.  Rows where Close < 5 are set to NaN per the paper filter. "
        "Proxy for the cross-sectional signal of Jegadeesh & Livnat (2006) / "
        "OpenSourceAP (Chen-Zimmermann)."
    ),
    "requires": ["Close"],
    "produces": [
        "osap_revenuesurprise_z",      # standardised surprise (main signal)
        "osap_revenuesurprise_chg",    # raw 4q YoY change in rev/share (unstandardised)
        "osap_revenuesurprise_accel",  # change in the z-score over past 2 quarters (momentum)
    ],
    "tags": ["fundamentals", "revenue", "surprise", "accounting", "pit"],
    "version": "1.0",
    "author": "Jegadeesh and Livnat 2006 / OpenSourceAP Chen-Zimmermann; per-ticker proxy implementation",
}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker revenue surprise features."""

    # Initialise output columns to NaN
    for col in METADATA["produces"]:
        df[col] = np.nan

    # Need at least some rows to work with
    if len(df) < 5:
        return df

    # -----------------------------------------------------------------------
    # Pull PIT fundamentals: revenue and shares_outstanding
    # -----------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _fundamentals.as_of(df, fields=["revenue_ttm", "shares_outstanding"])

    # Bail if fundamentals are missing for all rows
    if "fund_revenue_ttm" not in df.columns or "fund_shares_outstanding" not in df.columns:
        return df

    # -----------------------------------------------------------------------
    # Revenue per share (point-in-time)
    # _fundamentals delivers quarterly-cadence values forward-filled into the
    # daily frame.  We use revenue_ttm / shares_outstanding as the closest
    # readily available proxy for quarterly rev/share accumulated TTM.
    # -----------------------------------------------------------------------
    shares = df["fund_shares_outstanding"].replace(0, np.nan)
    rev_per_share = df["fund_revenue_ttm"] / shares  # NaN where either is NaN

    # -----------------------------------------------------------------------
    # 4-quarter YoY change in rev/share
    # Fundamentals update ~quarterly, so 1 quarter ≈ 63 trading days.
    # We approximate 4 quarters back as 252 trading days (1 year of trading days).
    # Use shift on the daily series; since fund_ values are forward-filled from
    # quarterly announcements this gives a meaningful year-ago reference.
    # -----------------------------------------------------------------------
    lag_252 = rev_per_share.shift(252)
    rev_chg = rev_per_share - lag_252  # YoY change in rev/share

    # -----------------------------------------------------------------------
    # Standardise: z-score of rev_chg using rolling 8-quarter window
    # ≈ 8 × 63 ≈ 504 trading days; require min 4 observations.
    # -----------------------------------------------------------------------
    window = 504
    min_periods = 4 * 63  # at least 4 quarterly data points worth of days

    roll_mean = rev_chg.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = rev_chg.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero / tiny std
    roll_std_safe = roll_std.where(roll_std > 1e-12, np.nan)

    z = (rev_chg - roll_mean) / roll_std_safe

    # Replace inf/-inf with NaN
    z = z.replace([np.inf, -np.inf], np.nan)
    rev_chg = rev_chg.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------------------------
    # Paper exclusion: price < $5
    # -----------------------------------------------------------------------
    price_ok = df["Close"] >= 5.0

    df["osap_revenuesurprise_z"]    = z.where(price_ok)
    df["osap_revenuesurprise_chg"]  = rev_chg.where(price_ok)

    # -----------------------------------------------------------------------
    # Acceleration: change in z-score over last 2 quarters (≈ 126 trading days)
    # Captures whether the surprise is improving or deteriorating.
    # -----------------------------------------------------------------------
    z_lag_126 = z.shift(126)
    accel = (z - z_lag_126).replace([np.inf, -np.inf], np.nan)
    df["osap_revenuesurprise_accel"] = accel.where(price_ok)

    # -----------------------------------------------------------------------
    # Drop scratch fund_* columns not in produces
    # -----------------------------------------------------------------------
    for scratch in ["fund_revenue_ttm", "fund_shares_outstanding"]:
        if scratch in df.columns:
            df.drop(columns=[scratch], inplace=True)

    return df
