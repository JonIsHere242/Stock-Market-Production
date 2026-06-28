"""
Overnight vs intraday reversal decomposition.

Extends the osap_streversal family by decomposing the trailing 21-day cumulative
return into its overnight (Open/prevClose) and intraday (Close/Open) components,
then takes their negatives as separate reversal signals plus the Lou-Polk-Skouras
tug-of-war difference. This is orthogonal to the parent (which uses a single
composite short-term reversal signal) because it separates *where* the past return
was earned.

Lou, D., Polk, C., & Skouras, S. (2019). "A tug of war: Overnight versus intraday
expected returns." Journal of Financial Economics, 134(1), 192–213.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

METADATA = {
    "name": "ext_reversal_overnight_intraday",
    "description": (
        "Decomposes trailing 21-day log-return into overnight (sum ln(Open/prevClose)) "
        "and intraday (sum ln(Close/Open)) components and produces their negatives as "
        "reversal signals plus the overnight-minus-intraday difference "
        "(Lou-Polk-Skouras tug-of-war). Per-ticker proxy; no cross-section needed. "
        "Orthogonal to osap_streversal which uses a composite price-impact reversal."
    ),
    "requires": ["Open", "High", "Low", "Close", "Volume"],
    "produces": [
        "ext_reversal_overnight_intraday_neg_on",    # negative cumulative overnight return
        "ext_reversal_overnight_intraday_neg_id",    # negative cumulative intraday return
        "ext_reversal_overnight_intraday_tugwar",    # overnight_ret - intraday_ret (neg, tug-of-war)
    ],
    "tags": ["reversal", "overnight", "intraday", "decomposition", "short_term"],
    "version": "1.0",
    "author": "Lou, Polk & Skouras (2019) JFE 134(1) 192-213; spec ext_reversal_overnight_intraday",
}

_WINDOW = 21


def compute(df: pd.DataFrame) -> pd.DataFrame:
    # Log returns for each component
    prev_close = df["Close"].shift(1)

    # overnight: open relative to previous close
    ln_on = np.log(df["Open"] / prev_close.replace(0, np.nan))

    # intraday: close relative to open of same day
    open_safe = df["Open"].replace(0, np.nan)
    ln_id = np.log(df["Close"] / open_safe)

    # Replace inf/-inf introduced by zero prices with NaN
    ln_on = ln_on.replace([np.inf, -np.inf], np.nan)
    ln_id = ln_id.replace([np.inf, -np.inf], np.nan)

    # Trailing 21-day sums (current day inclusive); min_periods keeps early rows as NaN
    cum_on = ln_on.rolling(_WINDOW, min_periods=_WINDOW).sum()
    cum_id = ln_id.rolling(_WINDOW, min_periods=_WINDOW).sum()

    # Reversal signals = negatives (high past return → expected mean-reversion)
    df["ext_reversal_overnight_intraday_neg_on"] = -cum_on
    df["ext_reversal_overnight_intraday_neg_id"] = -cum_id

    # Tug-of-war: overnight dominance over intraday (sign-reversed so high = strong ON reversal)
    # Original LPS: stocks with high ON and low ID returns have distinct future profiles.
    # We capture the spread as -(ON - ID) = ID - ON; positive when intraday dragged price up
    # more than overnight, signalling ON-reversal potential.
    df["ext_reversal_overnight_intraday_tugwar"] = -(cum_on - cum_id)

    return df
